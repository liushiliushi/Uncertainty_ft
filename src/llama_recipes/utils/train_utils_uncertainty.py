# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib
# from vllm import LLM
# from vllm.lora.request import LoRARequest
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
import torch.nn.functional as F
import wandb

from llama_recipes.model_checkpointing import save_fsdp_model_checkpoint_full, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_peft_checkpoint, save_model_checkpoint, save_merged_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_recipes.utils.flop_utils import FlopMeasure
from llama_recipes.utils.compute_metrics import compute_conf_metrics
from llama_recipes.utils.postprocess import postprocess_extract, confidence_replace
def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank,warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


def train(model, train_dataloader,eval_dataloader, test_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])



    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Get the ids of the numbers from 0 to 100
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    num_indices = torch.tensor(token_ids).to(model.device)

    # Start the training loop
    for epoch in range(train_config.num_epochs):
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break

                    batch.data.pop('labels')
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:

                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to(model.device)

                    y = batch.data.pop('y')
                    with autocast():
                        # get the
                        logits = model(**batch).logits


                    probabilities = F.softmax(logits, dim=-1)
                    decoded_indices = torch.argmax(probabilities, dim=-1)
                    decoded_indices = decoded_indices.tolist()
                    decoded_texts = [tokenizer.decode(indices[-1], skip_special_tokens=True) for indices in decoded_indices]

                    num_token = logits[:,-1,:] # get the logit of the confidence token
                    scores = torch.arange(0, 1.01, 0.01).view(1, 101).expand(logits.shape[0], 101).to(model.device)

                    if train_config.loss_type == 'brier':
                        num_conf = torch.index_select(num_token, 1, num_indices.squeeze(0)) # take out the logit of 0-100
                        num_conf = F.softmax(num_conf, dim=1)
                        # compute the loss
                        y_expanded = y.expand(logits.shape[0], 101)
                        squared_differences = (y_expanded - scores) ** 2
                        loss = torch.mean(torch.sum(num_conf * squared_differences, dim=1))
                    elif train_config.loss_type == 'sot':
                        norm_logit = torch.index_select(F.log_softmax(num_token, dim=1), 1, num_indices.squeeze(0))
                        smoothed = y * scores * (2 - scores) + (1 - y) * (1 - scores) * (1 + scores)
                        smoothed = smoothed / smoothed.sum(dim=1, keepdim=True)
                        loss = -torch.sum(norm_logit * smoothed, dim=1).mean()

                    #print(f"\nlabel: {y[0].item()}  {y[1].item()}") # {y[2].item()}  {y[3].item()}")#  {y[4].item()}  {y[5].item()}  {y[6].item()}  {y[7].item()}")
                    # print(f"prob: {decoded_texts[0]}  {decoded_texts[1]}")#  {decoded_texts[2]}  {decoded_texts[3]}")# {decoded_texts[4]}  {decoded_texts[5]}  {decoded_texts[6]}  {decoded_texts[7]}")
                    print(f"loss: {loss}")
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                # pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()


        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                if train_config.enable_fsdp:
                    dist.barrier()
                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"we are about to save the PEFT modules")
                    else:
                        print(f"we are about to save the PEFT modules")
                    if train_config.merge_peft and epoch == (train_config.num_epochs - 1):
                        model = model.merge_and_unload()
                        save_merged_checkpoint(model, tokenizer, train_config.output_dir)
                        if train_config.enable_fsdp:
                            if rank==0:
                                print(f"Merged modules are saved in {train_config.output_dir} directory")
                        else:
                            print(f"Merged modules are saved in {train_config.output_dir} directory")
                        
                    else:
                        save_peft_checkpoint(model, train_config.output_dir)
                        if train_config.enable_fsdp:
                            if rank==0:
                                print(f"PEFT modules are saved in {train_config.output_dir} directory")
                        else:
                            print(f"PEFT modules are saved in {train_config.output_dir} directory")

                else:
                    if not train_config.enable_fsdp:
                        save_model_checkpoint(model, train_config.output_dir)
                        
                    elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        print(" Saving the FSDP model checkpoint using FULL_STATE_DICT")
                        print("=====================================================")
                        save_fsdp_model_checkpoint_full(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                        
                        if train_config.save_optimizer:
                            print(" Saving the FSDP optimizer using FULL_STATE_DICT")
                            print("=====================================================")
                            save_optimizer_checkpoint(
                                model, optimizer, rank, train_config, epoch=epoch
                            )
                        
                    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:

                        if train_config.save_optimizer:
                            print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                            print("=====================================================")
                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                        else:
                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                            print("=====================================================")
                            save_model_and_optimizer_sharded(model, rank, train_config)

                        
                if train_config.enable_fsdp:
                    dist.barrier()
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    # eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = test(model, train_config, test_dataloader, local_rank, tokenizer, wandb_run)
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results


def train_chat(model, train_dataloader,eval_dataloader, test_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    # eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation_chat(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
    # test_ece, test_auroc = test(model, train_config, test_dataloader, local_rank, tokenizer, wandb_run)
    # test_ece, test_auroc = test_2stage(model, train_config, test_dataloader, local_rank, tokenizer, wandb_run)
    # print(f"test_ece:{test_ece} test_auroc:{test_auroc}")

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Get the ids of the numbers from 0 to 100
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    num_indices = torch.tensor(token_ids).to(model.device)
    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 80,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        # TODO:with MemoryTrace() as memtrace:  # track the memory usage
        if True:
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            # with profile(train_config,local_rank) as profile_context:
            if True:
                for step, batch in enumerate(train_dataloader): # TODO
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break

                    # batch.pop('labels')
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:

                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to(model.device)

                    y = batch.data.pop('y')

                    with autocast():
                        # get the
                        output = model(**batch)
                        logits = output.logits
                        loss_con = output.loss
                        # responses = model.generate(batch['input_ids'], **generation_kwargs) 
                        # batch_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses]


                    # decoded_indices = torch.argmax(logits, dim=-1)
                    # decoded_indices = decoded_indices.tolist()
                    # decoded_texts = [tokenizer.decode(indices[-1], skip_special_tokens=True) for indices in decoded_indices]

                    num_token = logits[:,-1,:] # get the logit of the confidence token
                    del logits
                    scores = torch.arange(0, 1.01, 0.01).view(1, 101).expand(y.shape[0], 101).to(model.device)

                    if train_config.loss_type == 'brier':
                        num_conf = torch.index_select(num_token, 1, num_indices.squeeze(0)) # take out the logit of 0-100
                        num_conf = F.softmax(num_conf, dim=1)
                        # compute the loss
                        y_expanded = y.expand(y.shape[0], 101)
                        squared_differences = (y_expanded - scores) ** 2
                        loss_cal = torch.mean(torch.sum(num_conf * squared_differences, dim=1))
                    elif train_config.loss_type == 'sot':
                        norm_logit = torch.index_select(F.log_softmax(num_token, dim=1), 1, num_indices.squeeze(0))
                        smoothed = y * scores * (2 - scores) + (1 - y) * (1 - scores) * (1 + scores)
                        smoothed = smoothed / smoothed.sum(dim=1, keepdim=True)
                        loss_cal = -torch.sum(norm_logit * smoothed, dim=1).mean()

                    #print(f"\nlabel: {y[0].item()}  {y[1].item()}") # {y[2].item()}  {y[3].item()}")#  {y[4].item()}  {y[5].item()}  {y[6].item()}  {y[7].item()}")
                    # print(f"prob: {decoded_texts[0]}  {decoded_texts[1]}")#  {decoded_texts[2]}  {decoded_texts[3]}")# {decoded_texts[4]}  {decoded_texts[5]}  {decoded_texts[6]}  {decoded_texts[7]}")
                    # loss = loss_con + loss_cal
                    loss = loss_cal
                    print(f"loss: {loss} loss_con: {loss_con} loss_cal: {loss_cal}")
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    # TODO
                    # if train_config.use_profiler or train_config.flop_counter:
                    #     profile_context.step()
                    # if train_config.flop_counter and profile_context.is_done():
                    #     TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                                'train/loss_con': loss_con.detach().float(),
                                'train/loss_cal': loss_cal.detach().float()
                            })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                    torch.cuda.empty_cache()
 

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        # TODO
        # if not train_config.enable_fsdp or rank==0:
        #     memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()


        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation_chat(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                if train_config.enable_fsdp:
                    dist.barrier()
                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"we are about to save the PEFT modules")
                    else:
                        print(f"we are about to save the PEFT modules")
                    if train_config.merge_peft and epoch == (train_config.num_epochs - 1):
                        model = model.merge_and_unload()
                        save_merged_checkpoint(model, tokenizer, train_config.output_dir)
                        if train_config.enable_fsdp:
                            if rank==0:
                                print(f"Merged modules are saved in {train_config.output_dir} directory")
                        else:
                            print(f"Merged modules are saved in {train_config.output_dir} directory")
                        
                    else:
                        save_peft_checkpoint(model, train_config.output_dir)
                        if train_config.enable_fsdp:
                            if rank==0:
                                print(f"PEFT modules are saved in {train_config.output_dir} directory")
                        else:
                            print(f"PEFT modules are saved in {train_config.output_dir} directory")

                else:
                    if not train_config.enable_fsdp:
                        save_model_checkpoint(model, train_config.output_dir)
                        
                    elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        print(" Saving the FSDP model checkpoint using FULL_STATE_DICT")
                        print("=====================================================")
                        save_fsdp_model_checkpoint_full(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                        
                        if train_config.save_optimizer:
                            print(" Saving the FSDP optimizer using FULL_STATE_DICT")
                            print("=====================================================")
                            save_optimizer_checkpoint(
                                model, optimizer, rank, train_config, epoch=epoch
                            )
                        
                    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:

                        if train_config.save_optimizer:
                            print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                            print("=====================================================")
                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                        else:
                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                            print("=====================================================")
                            save_model_and_optimizer_sharded(model, rank, train_config)

                        
                if train_config.enable_fsdp:
                    dist.barrier()
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    test_ece, test_auroc = test(model, train_config, test_dataloader, local_rank, tokenizer, wandb_run)    
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results

def train_dynamic(model, train_dataloader,eval_dataloader, test_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    # eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
    # test_ece, test_auroc = test(model, train_config, test_dataloader, local_rank, tokenizer, wandb_run)
    # print(f"test_ece:{test_ece} test_auroc:{test_auroc}")

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Get the ids of the numbers from 0 to 100
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    num_indices = torch.tensor(token_ids).to(model.device)
    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 100,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True
    }
   
    
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        model.train()
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):


                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    prompts = [json.loads(item) for item in batch["prompt"]]
                    query_tensors = tokenizer.apply_chat_template(prompts, tokenize=True, padding="longest", padding_side='left', truncation=True, return_tensors="pt", continue_final_message=True).to(model.device)
                    with torch.no_grad():
                        output = model.generate(query_tensors, **generation_kwargs) 
                    responses = output.sequences
                    logits = output.scores
                    batch_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses]
                    try:
                        prompts_new, _, y, _, _ = confidence_replace(prompts, batch_responses, batch['correct_answer'])
                    except:
                        continue
                    query_tensors_new = tokenizer(prompts_new, add_special_tokens=False, padding=True, truncation=True, return_tensors="pt")
                    white_spaces = torch.full((len(prompts_new), 1), 220)
                    attention = torch.full((len(prompts_new), 1), 1)
                    query_tensors_new['input_ids'] = torch.cat([query_tensors_new['input_ids'], white_spaces], dim=1).to(model.device)
                    query_tensors_new['attention_mask']  = torch.cat([query_tensors_new['attention_mask'], attention], dim=1).to(model.device)
                    y = torch.tensor(y).view(-1, 1).to(model.device)
                    with autocast():
                        logits = model(**query_tensors_new).logits
                
                    num_token = logits[:, -1, :]
                    scores = torch.arange(0, 1.01, 0.01).view(1, 101).expand(num_token.shape[0], 101).to(model.device)

                    if train_config.loss_type == 'brier':
                        num_conf = torch.index_select(num_token, 1, num_indices.squeeze(0)) # take out the logit of 0-100
                        num_conf = F.softmax(num_conf, dim=1)
                        # compute the loss
                        y_expanded = y.expand(num_token.shape[0], 101)
                        squared_differences = (y_expanded - scores) ** 2
                        loss = torch.mean(torch.sum(num_conf * squared_differences, dim=1))
                    elif train_config.loss_type == 'sot':
                        norm_logit = torch.index_select(F.log_softmax(num_token, dim=1), 1, num_indices.squeeze(0))
                        smoothed = y * scores * (2 - scores) + (1 - y) * (1 - scores) * (1 + scores)
                        smoothed = smoothed / smoothed.sum(dim=1, keepdim=True)
                        loss = -torch.sum(norm_logit * smoothed, dim=1).mean()

                    #print(f"\nlabel: {y[0].item()}  {y[1].item()}") # {y[2].item()}  {y[3].item()}")#  {y[4].item()}  {y[5].item()}  {y[6].item()}  {y[7].item()}")
                    # print(f"prob: {decoded_texts[0]}  {decoded_texts[1]}")#  {decoded_texts[2]}  {decoded_texts[3]}")# {decoded_texts[4]}  {decoded_texts[5]}  {decoded_texts[6]}  {decoded_texts[7]}")
                    print(f"loss: {loss}")
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
 

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()


        if train_config.run_validation:
            del logits
            torch.cuda.empty_cache()
            
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()
            if train_config.save_model: #and eval_epoch_loss < best_val_loss:
                if train_config.enable_fsdp:
                    dist.barrier()
                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"we are about to save the PEFT modules")
                    else:
                        print(f"we are about to save the PEFT modules")
                    if train_config.merge_peft and epoch == (train_config.num_epochs - 1):
                        model = model.merge_and_unload()
                        save_merged_checkpoint(model, tokenizer, train_config.output_dir)
                        if train_config.enable_fsdp:
                            if rank==0:
                                print(f"Merged modules are saved in {train_config.output_dir} directory")
                        else:
                            print(f"Merged modules are saved in {train_config.output_dir} directory")
                        
                    else:
                        save_peft_checkpoint(model, train_config.output_dir)
                        if train_config.enable_fsdp:
                            if rank==0:
                                print(f"PEFT modules are saved in {train_config.output_dir} directory")
                        else:
                            print(f"PEFT modules are saved in {train_config.output_dir} directory")

                else:
                    if not train_config.enable_fsdp:
                        save_model_checkpoint(model, train_config.output_dir)
                        
                    elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        print(" Saving the FSDP model checkpoint using FULL_STATE_DICT")
                        print("=====================================================")
                        save_fsdp_model_checkpoint_full(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                        
                        if train_config.save_optimizer:
                            print(" Saving the FSDP optimizer using FULL_STATE_DICT")
                            print("=====================================================")
                            save_optimizer_checkpoint(
                                model, optimizer, rank, train_config, epoch=epoch
                            )
                        
                    elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:

                        if train_config.save_optimizer:
                            print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                            print("=====================================================")
                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                        else:
                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                            print("=====================================================")
                            save_model_and_optimizer_sharded(model, rank, train_config)

               
                if train_config.enable_fsdp:
                    dist.barrier()
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation_dynamic(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)        
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    test_ece, test_auroc = test(model, train_config, test_dataloader, local_rank, tokenizer, wandb_run)    
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results

def evaluation_chat(model,train_config, eval_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    all_y = []
    eval_probs = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    eval_loss_con = 0.0
    eval_loss_cal = 0.0
    total_eval_steps = 0
    # Get the ids of the numbers from 0 to 100
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    num_indices = torch.tensor(token_ids).to(model.device)
    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 100,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True
    }
    with MemoryTrace() as memtrace:
        id = 0
        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            with torch.no_grad():
                output = model(**batch)
                logits = output.logits
                loss_con = output.loss
            num_token = logits[:,-1,:] # get the logit of the confidence token
            del logits
            y = batch.data.pop('y')
            scores = torch.arange(0, 1.01, 0.01).view(1, 101).expand(y.shape[0], 101).to(model.device)

            if train_config.loss_type == 'brier':
                num_conf = torch.index_select(num_token, 1, num_indices.squeeze(0)) # take out the logit of 0-100
                num_conf = F.softmax(num_conf, dim=1)
                # compute the loss
                y_expanded = y.expand(y.shape[0], 101)
                squared_differences = (y_expanded - scores) ** 2
                loss_cal = torch.mean(torch.sum(num_conf * squared_differences, dim=1))
            elif train_config.loss_type == 'sot':
                norm_logit = torch.index_select(F.log_softmax(num_token, dim=1), 1, num_indices.squeeze(0))
                smoothed = y * scores * (2 - scores) + (1 - y) * (1 - scores) * (1 + scores)
                smoothed = smoothed / smoothed.sum(dim=1, keepdim=True)
                loss_cal = -torch.sum(norm_logit * smoothed, dim=1).mean()
            if train_config.save_metrics:
                val_step_loss.append(loss.detach().float().item())
                val_step_perplexity.append(float(torch.exp(loss.detach().float())))

            # loss = loss_con + loss_cal
            loss = loss_cal
            print(f"loss: {loss} loss_con: {loss_con} loss_cal: {loss_cal}") 
            
            eval_loss += loss.detach().float()
            eval_loss_con += loss_con.detach().float()
            eval_loss_cal += loss_cal.detach().float()
            probs = 0.01 * torch.argmax(torch.index_select(num_token, 1, num_indices.squeeze(0)), dim=1)
            eval_probs.extend(probs.detach().cpu().numpy().tolist())
            all_y.extend(y.squeeze(1).detach().cpu().numpy().tolist())

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    val_metrics = compute_conf_metrics(all_y, eval_probs)
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_loss_con, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_loss_cal, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_loss_con = eval_loss_con / len(eval_dataloader)
    eval_epoch_loss_cal = eval_loss_cal / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
        eval_epoch_loss_con = eval_epoch_loss_con/world_size
        eval_epoch_loss_cal = eval_epoch_loss_cal/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        if not train_config.enable_fsdp or rank==0:
            wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                        'eval/loss_con': eval_epoch_loss_con,
                        'eval/loss_cal': eval_epoch_loss_cal,
                        'eval/ece': ece_score,
                        'eval/roc_auc': roc_auc_score,
                    }, commit=False)

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity


def evaluation_dynamic(model,train_config, eval_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    all_y = []
    eval_probs = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    # Get the ids of the numbers from 0 to 100
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    num_indices = torch.tensor(token_ids).to(model.device)
    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 100,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True
    }
    with MemoryTrace() as memtrace:
        id = 0
        model.eval()
        wan_table = wandb.Table(columns=['response','confidence', 'y'])
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            prompts = [json.loads(item) for item in batch["prompt"]]
            query_tensors = tokenizer.apply_chat_template(prompts, tokenize=True, padding="longest", padding_side='left', truncation=True, return_tensors="pt", continue_final_message=True).to(model.device)
            with torch.no_grad():
                output = model.generate(query_tensors, **generation_kwargs) 
            responses = output.sequences
            logits = output.scores
            batch_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses]
            try:
                prompts_new, confidences, y, y_None, confidences_None = confidence_replace(prompts, batch_responses, batch['correct_answer'])
            except:
                continue
            all_y.extend(y)
            eval_probs.extend(confidences)
            for response, confidence, y_item in zip(batch_responses, confidences_None, y_None):
                wan_table.add_data(response, confidence, y_item)
            query_tensors_new = tokenizer(prompts_new, add_special_tokens=False, padding=True, truncation=True, return_tensors="pt")
            white_spaces = torch.full((len(prompts_new), 1), 220)
            attention = torch.full((len(prompts_new), 1), 1)
            query_tensors_new['input_ids'] = torch.cat([query_tensors_new['input_ids'], white_spaces], dim=1).to(model.device)
            query_tensors_new['attention_mask']  = torch.cat([query_tensors_new['attention_mask'], attention], dim=1).to(model.device)
            
            with torch.no_grad():
                logits = model(**query_tensors_new).logits
            y = torch.tensor(y).view(-1, 1).to(model.device)  
            num_token = logits[:, -1, :]
            scores = torch.arange(0, 1.01, 0.01).view(1, 101).expand(num_token.shape[0], 101).to(model.device)

            if train_config.loss_type == 'brier':
                num_conf = torch.index_select(num_token, 1, num_indices.squeeze(0)) # take out the logit of 0-100
                num_conf = F.softmax(num_conf, dim=1)
                # compute the loss
                y_expanded = y.expand(num_token.shape[0], 101)
                squared_differences = (y_expanded - scores) ** 2
                loss = torch.mean(torch.sum(num_conf * squared_differences, dim=1))
            elif train_config.loss_type == 'sot':
                norm_logit = torch.index_select(F.log_softmax(num_token, dim=1), 1, num_indices.squeeze(0))
                smoothed = y * scores * (2 - scores) + (1 - y) * (1 - scores) * (1 + scores)
                smoothed = smoothed / smoothed.sum(dim=1, keepdim=True)
                loss = -torch.sum(norm_logit * smoothed, dim=1).mean()

                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()
    if wandb_run:
        if not train_config.enable_fsdp or rank==0:
            wandb_run.log({"Evaluation_samples": wan_table})
    # Compute ECE and ROC-AUC score given all_y and eval_probs
    val_metrics = compute_conf_metrics(all_y, eval_probs)
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        if not train_config.enable_fsdp or rank==0:
            wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                        'eval/ece': ece_score,
                        'eval/roc_auc': roc_auc_score,
                    }, commit=False)

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity

def test(model, train_config, test_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    # llm = LLM(model=train_config.output_dir, max_model_len=93968)
    
    all_y = []
    test_probs = []

    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 200,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True
    }

    with MemoryTrace() as memtrace:
        id = 0
        wan_table = wandb.Table(columns=['response','confidence', 'y'])
        for step, batch in enumerate(tqdm(test_dataloader,colour="green", desc="testing Epoch", dynamic_ncols=True)):
            prompts = [json.loads(item) for item in batch["prompt"]]
            query_tensors = tokenizer.apply_chat_template(prompts, tokenize=True, padding="longest", padding_side='left', truncation=True, return_tensors="pt", continue_final_message=True).to(model.device)
            # stop when the maximum number of eval steps is reached
            
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                output = model.generate(query_tensors, **generation_kwargs) 
                responses = output.sequences
                logits = output.scores
                batch_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses]
                try:
                    confidences, y, y_unfiltered, confidence_unfiltered = postprocess_extract(prompts, batch_responses, batch['correct_answer'],)
                except:
                    continue
            for response, confidence, y_item in zip(batch_responses, confidence_unfiltered, y_unfiltered):
                wan_table.add_data(response, confidence, y_item)        
            all_y.extend(y)
            test_probs.extend(confidences)

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    if wandb_run:
        if not train_config.enable_fsdp or rank==0:
            wandb_run.log({"Testing_samples": wan_table})
    number = len(all_y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(all_y, test_probs)
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    if wandb_run:
        if not train_config.enable_fsdp or rank==0:
            wandb_run.log({
                        'test/ece': ece_score,
                        'test/roc_auc': roc_auc_score,
                    }, commit=False)

    return ece_score, roc_auc_score

def test_2stage(model, train_config, test_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    # llm = LLM(model=train_config.output_dir, max_model_len=93968)
    
    all_y = []
    test_probs = []
    test_probs_stage1 = []

    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 200,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True
    }
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    num_indices = torch.tensor(token_ids).to(model.device)
    with MemoryTrace() as memtrace:
        id = 0
        wan_table = wandb.Table(columns=['response','confidence', 'y'])
        for step, batch in enumerate(tqdm(test_dataloader,colour="green", desc="testing Epoch", dynamic_ncols=True)):
            prompts = [json.loads(item) for item in batch["prompt"]]
            query_tensors = tokenizer.apply_chat_template(prompts, tokenize=True, padding="longest", padding_side='left', truncation=True, return_tensors="pt", continue_final_message=True).to(model.device)
            # stop when the maximum number of eval steps is reached
            

            with torch.no_grad():
                # Forward pass and compute loss
                output = model.generate(query_tensors, **generation_kwargs) 
                responses = output.sequences
                logits = output.scores
                batch_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses]
                try:
                    prompts_new, confidence_stage1, y, _, _ = confidence_replace(prompts, batch_responses, batch['correct_answer'])

                except:
                    continue
                query_tensors_new = tokenizer(prompts_new, add_special_tokens=False, padding=True, truncation=True, return_tensors="pt")
                white_spaces = torch.full((len(prompts_new), 1), 220)
                attention = torch.full((len(prompts_new), 1), 1)
                query_tensors_new['input_ids'] = torch.cat([query_tensors_new['input_ids'], white_spaces], dim=1).to(model.device)
                query_tensors_new['attention_mask']  = torch.cat([query_tensors_new['attention_mask'], attention], dim=1).to(model.device)
                y = torch.tensor(y).view(-1, 1).to(model.device)
                with torch.no_grad():
                    logits = model(**query_tensors_new).logits
            num_token = logits[:,-1,:]
            probs = 0.01 * torch.argmax(torch.index_select(num_token, 1, num_indices.squeeze(0)), dim=1)
            test_probs.extend(probs.detach().cpu().numpy().tolist())
            test_probs_stage1.extend(confidence_stage1)
            all_y.extend(y.squeeze(1).detach().cpu().numpy().tolist())
            # for response, confidence, y_item in zip(batch_responses, confidence_unfiltered, y_unfiltered):
            #     wan_table.add_data(response, confidence, y_item)        


    # Compute ECE and ROC-AUC score given all_y and eval_probs
    # if wandb_run:
    #     if not train_config.enable_fsdp or rank==0:
    #         wandb_run.log({"Testing_samples": wan_table})
    number = len(all_y)
    print(f"Number: {number}")
    print("Stage1:")
    val_metrics_stage1 = compute_conf_metrics(all_y, test_probs_stage1)
    print("Stage2:")
    val_metrics = compute_conf_metrics(all_y, test_probs)
    
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    if wandb_run:
        if not train_config.enable_fsdp or rank==0:
            wandb_run.log({
                        'test/ece': ece_score,
                        'test/roc_auc': roc_auc_score,
                    }, commit=False)

    return ece_score, roc_auc_score


def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")




def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""


    verify_bfloat_support = ((
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and torch.version.cuda >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    ) or
    (is_xpu_available()))


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
