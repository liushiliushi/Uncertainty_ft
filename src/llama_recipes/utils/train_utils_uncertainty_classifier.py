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
import sys
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
import torch.nn.functional as F
import wandb

from llama_recipes.model_checkpointing import save_peft_checkpoint, save_model_checkpoint, save_merged_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_recipes.utils.flop_utils import FlopMeasure
from llama_recipes.utils.compute_metrics import compute_conf_metrics, plot_confidence_histogram, plot_ece_diagram
from llama_recipes.utils.postprocess import postprocess_extract, confidence_replace, confidence_replace_3level, confidence_replace_gpt,confidence_replace_correct, confidence_replace_implicit
from vllm import LLM, SamplingParams
def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

import torch.nn as nn
class ConfidenceClassifier(nn.Module):
    def __init__(self, lm_head, token_indices, init_from_lm_head=True):
        super().__init__()
        
        # 创建新的线性层
        self.classifier = nn.Linear(lm_head.in_features, lm_head.out_features, bias=(lm_head.bias is not None))
        
        if init_from_lm_head:
            # 用lm_head的权重初始化
            selected_weights = lm_head.weight
            selected_bias = lm_head.bias if lm_head.bias is not None else None 
            
            with torch.no_grad():
                self.classifier.weight.data = selected_weights
                if selected_bias is not None:
                    self.classifier.bias.data = selected_bias
        # 如果init_from_lm_head=False，则使用PyTorch的默认随机初始化
    
    def forward(self, hidden_states):
        return self.classifier(hidden_states)
@contextlib.contextmanager
def profile(cfg, accelerator=None):
    """
    兼容 accelerate 的简易 profiler/flop 计数器上下文管理。
    """
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter simultaneously.")

    if use_profiler:
        wait_step, warmup_step, active_step = 1, 2, 3
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=wait_step, warmup=warmup_step, active=active_step, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.profiler_dir),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        warmup_step = cfg.flop_counter_start
        # 如果想在多卡时区分 rank，可用 accelerator.process_index
        with FlopMeasure(rank=0, warmup_step=warmup_step) as flop_counter:
            yield flop_counter
    else:
        yield None

def train_chat(
    model,
    confidence_classifier,
    train_dataloader,
    eval_dataloaders_dict,
    test_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    accelerator,
    wandb_run=None,
):   
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
    # 根据配置决定是否冻结模型参数
    if train_config.train_model_with_classifier:
        print("Training both model and classifier parameters...")
        for param in model.parameters():
            param.requires_grad = True
    else:
        print("Freezing model parameters, only training classifier...")
        for param in model.parameters():
            param.requires_grad = False
    
    print("Ensuring classifier parameters are trainable...")
    for param in confidence_classifier.parameters():
        param.requires_grad = True
    
    # 打印可训练参数数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    classifier_params = sum(p.numel() for p in confidence_classifier.parameters() if p.requires_grad)
    print(f"Trainable model parameters: {model_params}")
    print(f"Trainable classifier parameters: {classifier_params}")
    # 根据是否使用混合精度来设置 autocast
    if train_config.use_fp16 and torch.cuda.is_available():
        autocast = torch.cuda.amp.autocast
    else:
        autocast = contextlib.nullcontext  # 使用 nullcontext 作为默认值

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Get the ids of the numbers from 0 to 100
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    print("----------------------------")
    print(token_ids)
    num_indices = torch.tensor(token_ids).to(model.device)
    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 80,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0.1,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    for epoch in range(train_config.num_epochs):
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            if train_config.train_model_with_classifier:
                model.train()  # 同时训练model和classifier时，model也要进入训练模式
            else:
                model.eval()  # 只训练classifier时，model设为eval模式
            confidence_classifier.train()  # classifier处于训练模式
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(train_dataloader, colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True, disable=not accelerator.is_local_main_process)
            for step, batch in enumerate(pbar): 
                print(batch['input_ids'].size()[1])
                if batch['input_ids'].size()[1] > 1000:
                    continue
                total_train_steps += 1
                # stop when the maximum number of training steps is reached
                if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                    max_steps_reached = True
                    accelerator.print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)

                y = batch.data.pop('y')

                output = model(**batch, output_hidden_states=True)
                hidden = output.hidden_states[-1]
                hidden = hidden[:,-1,:]
                # num_conf = torch.index_select(hidden, 1, num_indices.squeeze(0))
                num_token = confidence_classifier(hidden)
                scores = torch.arange(0, 1.01, 0.01).view(1, 101).expand(y.shape[0], 101).to(model.device)

                if train_config.loss_type == 'brier':
                    num_conf = torch.index_select(num_token, 1, num_indices.squeeze(0))
                    num_conf = F.softmax(num_conf, dim=1)
                    # compute the loss
                    y_expanded = y.expand(y.shape[0], 101)
                    squared_differences = (y_expanded - scores) ** 2
                    loss_cal = torch.mean(torch.sum(num_conf * squared_differences, dim=1))
                    print(torch.argmax(num_conf, dim=1))
                    print(y.squeeze(1))
                elif train_config.loss_type == 'sot':
                    norm_logit = torch.index_select(F.log_softmax(num_token, dim=1), 1, num_indices.squeeze(0))
                    smoothed = y * scores * (2 - scores) + (1 - y) * (1 - scores) * (1 + scores)
                    smoothed = smoothed / smoothed.sum(dim=1, keepdim=True)
                    loss_cal = -torch.sum(norm_logit * smoothed, dim=1).mean()
                if train_config.add_loss_con:
                    loss = loss_con + loss_cal
                else:
                    loss = loss_cal
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.detach().float()

                if wandb_run and accelerator.is_main_process:
                    wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': loss.detach().float(),
                                'train/loss_cal': loss_cal.detach().float()
                        })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    torch.cuda.empty_cache()
 

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        avg_loss = total_loss / len(train_dataloader)
        avg_ppl = float(torch.exp(torch.tensor(avg_loss)))  # ppl = exp(loss)

        # Update the learning rate as needed
        lr_scheduler.step()


        if train_config.run_validation :
            # Evaluate on all 5 datasets
            total_eval_score = 0.0
            total_eval_loss = 0.0
            dataset_scores = {}
            original_dataset = train_config.dataset
            for dataset_name, eval_dataloader in eval_dataloaders_dict.items():
                # Store original dataset name for logging
                train_config.dataset = dataset_name
                
                accelerator.print(f"Evaluating on dataset: {dataset_name}")
                
                # 创建一个临时的 wandb 记录器，用于覆盖原来的 wandb 记录
                temp_wandb_run = None
                if accelerator.is_main_process and wandb_run:
                    # 创建一个字典来存储当前数据集的指标
                    temp_metrics = {}
                    def temp_log(metrics_dict, commit=True):
                        # 将指标临时存储在字典中，不直接记录到 wandb
                        temp_metrics.update(metrics_dict)
                    
                    # 保存原始的 wandb 记录函数
                    original_log = wandb_run.log
                    # 替换为临时函数
                    wandb_run.log = temp_log
                
                # 执行评估
                eval_ppl, eval_epoch_loss, _, _ = evaluation_chat(
                    model,
                    confidence_classifier,
                    train_config,
                    eval_dataloader,
                    tokenizer,
                    accelerator,
                    wandb_run,
                )
                
                # 收集并重新记录带数据集名称的指标
                if accelerator.is_main_process and wandb_run:
                    # 恢复原始的 wandb 记录函数
                    wandb_run.log = original_log
                    
                    # 重新格式化指标，添加数据集名称前缀
                    dataset_metrics = {}
                    for key, value in temp_metrics.items():
                        if key.startswith('eval/'):
                            # 将 'eval/' 替换为 'eval/{dataset_name}/'
                            new_key = f'eval/{dataset_name}/{key[5:]}'
                            dataset_metrics[new_key] = value
                        elif key.startswith('plots/'):
                            dataset_metrics[key] = value
                    
                    # 记录特定数据集的指标
                    wandb_run.log(dataset_metrics)
                    
                    # 提取关键指标用于计算总分
                    ece_score = temp_metrics.get('eval/ece', 0)
                    roc_auc_score = temp_metrics.get('eval/roc_auc', 0)
                    
                    # 计算数据集得分
                    dataset_score = 2 * roc_auc_score - ece_score
                    dataset_scores[dataset_name] = dataset_score
                    total_eval_score += dataset_score
                    
                    # 记录该数据集的总体得分和损失
                    wandb_run.log({
                        f'eval/{dataset_name}/score': dataset_score,
                    })
                    
                    accelerator.print(f"Dataset {dataset_name} - Score: {dataset_score:.4f}, Loss: {eval_epoch_loss:.4f}")

            train_config.dataset = original_dataset
            
            # Calculate average scores and log them
            if accelerator.is_main_process and wandb_run:
                # Log aggregate metrics
                wandb_run.log({
                    'eval/summary/total_score': total_eval_score,
                })
                accelerator.print(f"Evaluation Total Score: {total_eval_score:.4f}")

            
            # Save model if validation performance improved
            if accelerator.is_main_process:
                if True:
                    if train_config.save_model:
                        if train_config.use_peft:
                            if hasattr(model, "module"):
                                model = model.module
                            if train_config.merge_peft and epoch == (train_config.num_epochs - 1):
                                model = model.to(dtype=torch.float32)
                                model = model.merge_and_unload()
                                model = model.to(dtype=torch.float16)
                                save_merged_checkpoint(model, tokenizer, train_config.output_dir)
                                accelerator.print(f"Merged modules are saved in {train_config.output_dir} directory")
                            else:
                                save_peft_checkpoint(model, train_config.output_dir)
                                accelerator.print(f"PEFT modules are saved in {train_config.output_dir} directory")
                        else:
                            save_model_checkpoint(model, train_config.output_dir)
                            accelerator.print(f"Model is saved in {train_config.output_dir} directory")


    results = None
    sys.exit(0)

    return results

def evaluation_chat(
    model,
    confidence_classifier,
    train_config,
    eval_dataloader,
    tokenizer,
    accelerator,
    wandb_run=None,
    original=False
):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
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
    with torch.no_grad():
        model.eval()
        confidence_classifier.eval()  # 设置classifier为评估模式
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            y = batch.data.pop('y')
            output = model(**batch, output_hidden_states=True)
            hidden = output.hidden_states[-1]
            hidden = hidden[:,-1,:]  # 获取最后一个token的hidden state
            loss_con = output.loss
            num_token = confidence_classifier(hidden)  # 使用classifier获取置信度logits
            scores = torch.arange(0, 1.01, 0.01).view(1, 101).expand(y.shape[0], 101).to(model.device)

            if train_config.loss_type == 'brier':
                num_conf = torch.index_select(num_token, 1, num_indices.squeeze(0))
                num_conf = F.softmax(num_conf, dim=1)
                # compute the loss
                y_expanded = y.expand(y.shape[0], 101)
                squared_differences = (y_expanded - scores) ** 2
                loss_cal = torch.mean(torch.sum(num_conf * squared_differences, dim=1))
            elif train_config.loss_type == 'sot':
                norm_logit = F.log_softmax(num_token, dim=1)  # 直接使用101维的logits
                smoothed = y * scores * (2 - scores) + (1 - y) * (1 - scores) * (1 + scores)
                smoothed = smoothed / smoothed.sum(dim=1, keepdim=True)
                loss_cal = -torch.sum(norm_logit * smoothed, dim=1).mean()
            if train_config.save_metrics:
                val_step_loss.append(loss.detach().float().item())
                val_step_perplexity.append(float(torch.exp(loss.detach().float())))

            if train_config.add_loss_con:
                loss = loss_con + loss_cal
            else:
                loss = loss_cal
            
            eval_loss += loss.detach().float()
            eval_loss_con += loss_con.detach().float()
            eval_loss_cal += loss_cal.detach().float()
            probs = 0.01 * torch.argmax(num_conf, dim=1)  # 直接从101维logits计算概率
            print(probs)
            print(y.squeeze(1).detach().cpu().numpy().tolist())
            # TODO:
            # probs, y = accelerator.gather_for_metrics((probs, y))
            eval_probs.extend(probs.detach().cpu().numpy().tolist())
            all_y.extend(y.squeeze(1).detach().cpu().numpy().tolist())

    total_num = len(all_y)
    
    # 收集所有设备上的数据
    gathered_all_y = all_y
    gathered_eval_probs = eval_probs

    # 计算平均损失和困惑度
    eval_epoch_loss = eval_loss / len(eval_dataloader) / accelerator.num_processes
    eval_epoch_loss_con = eval_loss_con / len(eval_dataloader) / accelerator.num_processes
    eval_epoch_loss_cal = eval_loss_cal / len(eval_dataloader) / accelerator.num_processes
    eval_ppl = torch.exp(eval_epoch_loss)

    # 只在主进程上计算指标并进行可视化
    if accelerator.is_main_process:
        # 将收集到的数据转换为列表
        all_gathered_y = gathered_all_y
        all_gathered_probs = gathered_eval_probs
        
        # 计算指标
        accelerator.print(f"====== {train_config.dataset}")
        accelerator.print(f"Number: {total_num}")
        val_metrics = compute_conf_metrics(all_gathered_y, all_gathered_probs, total_num)
        ece_score = val_metrics['ece']
        roc_auc_score = val_metrics['auroc']

        # 在主进程上记录和可视化结果
        if train_config.use_wandb:
            plot_confidence_histogram(
                all_gathered_y, all_gathered_probs, "evaluation", 
                val_metrics['acc'], val_metrics['auroc'], val_metrics['ece'], 
                wandb_run, original, train_config.dataset, use_annotation=True
            )
            plot_ece_diagram(
                all_gathered_y, all_gathered_probs, "evaluation", 
                wandb_run, original, train_config.dataset
            )
        
        # 打印评估指标
        accelerator.print(f" {eval_ppl=} {eval_epoch_loss=}")

        # 记录到wandb
        if wandb_run:
            wandb_run.log({
                'eval/perplexity': eval_ppl,
                'eval/loss': eval_epoch_loss,
                'eval/loss_con': eval_epoch_loss_con,
                'eval/loss_cal': eval_epoch_loss_cal,
                'eval/ece': ece_score,
                'eval/roc_auc': roc_auc_score,
            }, commit=False)
    else:
        # 非主进程不需要计算指标
        ece_score = 0
        roc_auc_score = 0

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity

def test_2stage(model, train_config, test_dataloader, local_rank, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    # llm = LLM(model=train_config.output_dir, max_model_len=93968)
    
    all_y = []
    test_probs = []
    test_probs_stage1 = []

    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 200, 
        "top_k": 0.0,
        "top_p": 1.0,
        "repetition_penalty": train_config.repetition_penalty,
        "temperature": train_config.temperature,
        "do_sample": train_config.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True,
        # "pad_token_id": tokenizer.pad_token_id
    }
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    num_indices = torch.tensor(token_ids).to(model.device)
    with MemoryTrace() as memtrace:
        id = 0
        wan_table = wandb.Table(columns=['response','confidence', 'y'])
        for step, batch in enumerate(tqdm(test_dataloader,colour="green", desc="testing Epoch", dynamic_ncols=False)):
            
            prompts = [json.loads(item) for item in batch["prompt"]]
            query_tensors = tokenizer.apply_chat_template(prompts, tokenize=True, padding="longest", padding_side='left', truncation=True, return_dict=True, return_tensors="pt", continue_final_message=True).to(model.device)
            # stop when the maximum number of eval steps is reached
            

            with torch.no_grad():
                output = model.generate(**query_tensors, **generation_kwargs, ) 
                # output = model(**query_tensors, **generation_kwargs)
                responses = output.sequences
                batch_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses]
                prompts_new, out_responses, questions, confidence_stage1, y, y_unfiltered, confidence_unfiltered,_ = confidence_replace(prompts, batch_responses, batch['correct_answer'], train_config.dataset)
                if len(prompts_new) == 0:
                    continue
                y = torch.tensor(y).view(-1, 1).to(model.device)

                # include attention mask
                query_tensors_new = tokenizer.apply_chat_template(prompts_new, tokenize=True, padding="longest", padding_side='left', truncation=True, return_dict=True, return_tensors="pt", continue_final_message=True).to(model.device)
                white_spaces = torch.full((len(prompts_new), 1), 220).to(model.device)
                white_attention = torch.full((len(prompts_new), 1), 1).to(model.device)
                query_tensors_new['input_ids'] = torch.cat([query_tensors_new['input_ids'], white_spaces], dim=1).to(model.device)
                query_tensors_new['attention_mask'] = torch.cat([query_tensors_new['attention_mask'], white_attention], dim=1).to(model.device)
                logits = model(**query_tensors_new).logits

                # exclude attention mask
                # query_tensors_new = tokenizer.apply_chat_template(prompts_new, tokenize=True, padding="longest", padding_side='left', truncation=True, return_tensors="pt", continue_final_message=True).to(model.device)
                # white_spaces = torch.full((len(prompts_new), 1), 220).to(model.device)
                # query_tensors_new = torch.cat([query_tensors_new, white_spaces], dim=1).to(model.device)
                # logits = model(query_tensors_new).logits

            num_token = logits[:,-1,:]
            probs = 0.01 * torch.argmax(torch.index_select(num_token, 1, num_indices.squeeze(0)), dim=1)
            # num_token2 = logits2[:,-1,:]
            # probs2 = 0.01 * torch.argmax(torch.index_select(num_token2, 1, num_indices.squeeze(0)), dim=1)
            test_probs.extend(probs.detach().cpu().numpy().tolist())
            test_probs_stage1.extend(confidence_stage1)
            # print(probs)
            # print(confidence_stage1)
            all_y.extend(y.squeeze(1).detach().cpu().numpy().tolist())
            for response, confidence, y_item in zip(batch_responses, confidence_unfiltered, y_unfiltered):
                wan_table.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    if wandb_run:
        if original == True:
            wandb_run.log({"Testing_samples/original": wan_table})
        else:
            wandb_run.log({"Testing_samples/fine-tuned": wan_table})

    number = len(all_y)
    print(f"Number: {number}")
    print("Stage1:")
    val_metrics_stage1 = compute_conf_metrics(all_y, test_probs_stage1)
    if train_config.use_wandb:
        plot_confidence_histogram(all_y, test_probs_stage1, "stage1", val_metrics_stage1['acc'], val_metrics_stage1['auroc'], val_metrics_stage1['ece'], wandb_run, original, use_annotation=True)
        plot_ece_diagram(all_y, test_probs_stage1, "stage1", wandb_run, original)

    print("Stage2:")
    val_metrics = compute_conf_metrics(all_y, test_probs)
    if train_config.use_wandb:
        plot_confidence_histogram(all_y, test_probs, "stage2", val_metrics['acc'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, use_annotation=True)
        plot_ece_diagram(all_y, test_probs, "stage2", wandb_run,original)
    
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']

    if wandb_run:
        wandb_run.log({
                        'test/ece_stage1': ece_score,
                        'test/roc_auc_stage1': roc_auc_score,
                        'test/ece_stage2': ece_score,
                        'test/roc_auc_stage2': roc_auc_score,
                    }, commit=False)

    return ece_score, roc_auc_score

def test_vllm(train_config, test_dataset, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    all_y = []
    test_probs = []
    test_probs_stage1 = []
    llm = LLM(
        model=train_config.model_name if original else train_config.output_dir,
        tensor_parallel_size=1,
        dtype="float16",
        seed=train_config.seed,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    if train_config.test_linguistic:
        responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace_3level(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    elif train_config.test_correct:
        responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace_correct(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    else:
        responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    if wandb_run:
        if original == True:
            wandb_run.log({f"Testing_{train_config.dataset}/original": wan_table})
        else:
            wandb_run.log({f"Testing_{train_config.dataset}/fine-tuned": wan_table})

    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences, len(prompts))
    if train_config.use_wandb:
        plot_confidence_histogram(y, out_confidences, "stage1", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, out_confidences, "stage1", wandb_run, original, train_config.dataset)

    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    if wandb_run:
        wandb_run.log({
                        f'test/number_{train_config.dataset}': number,
                        f'test/acc_{train_config.dataset}': val_metrics['acc'],
                        f'test/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'test/ece_{train_config.dataset}': ece_score,
                        f'test/auroc_{train_config.dataset}': roc_auc_score,
                        f'test/score_{train_config.dataset}': score,
                    }, commit=False)

    return ece_score, roc_auc_score, val_metrics['acc2']


def test_implicit(train_config, test_dataset, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    all_y = []
    test_probs = []
    test_probs_stage1 = []
    llm = LLM(
        model=train_config.model_name if original else train_config.output_dir,
        tensor_parallel_size=1,
        dtype="float16",
        seed=train_config.seed,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace_implicit(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    if wandb_run:
        if original == True:
            wandb_run.log({f"Testing_{train_config.dataset}/original": wan_table})
        else:
            wandb_run.log({f"Testing_{train_config.dataset}/fine-tuned": wan_table})

    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences, len(prompts))
    if train_config.use_wandb:
        plot_confidence_histogram(y, out_confidences, "stage1", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, out_confidences, "stage1", wandb_run, original, train_config.dataset)

    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    if wandb_run:
        wandb_run.log({
                        f'test/number_{train_config.dataset}': number,
                        f'test/acc_{train_config.dataset}': val_metrics['acc'],
                        f'test/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'test/ece_{train_config.dataset}': ece_score,
                        f'test/auroc_{train_config.dataset}': roc_auc_score,
                        f'test/score_{train_config.dataset}': score,
                    }, commit=False)

    return ece_score, roc_auc_score, val_metrics['acc2']


def test_gpt(train_config, test_dataset, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader using OpenAI API

    Args:
        train_config: The training configuration
        test_dataset: The dataset containing the test data
        tokenizer: The tokenizer used to process prompts
        wandb_run: The wandb run object for logging
        original: Whether to use the original model or the fine-tuned model

    Returns: ece_score, roc_auc_score, accuracy
    """
    all_y = []
    test_probs = []
    test_probs_stage1 = []
    original = True
    # Initialize OpenAI client
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            api_version=os.environ['OPENAI_API_VERSION'],
            azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        )
    except:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
        )
    # Create a wandb table for logging
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    
    # Process prompts
    prompts = [json.loads(item) for item in test_dataset["prompt"]]

    # Generate responses using OpenAI API
    outputs = []
    for prompt in prompts:
        try:
            response = client.chat.completions.create(
                model=os.environ['OPENAI_DEPLOYMENT_NAME'],
                messages=prompt,
                temperature=train_config.temperature,
                max_tokens=400,
                top_p=1.0,
                n=1
            )
            outputs.append(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating response: {e}")
            outputs.append("")
    
    responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace_gpt(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=False)
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table.add_data(response, confidence, y_item)        

    # Log to wandb
    if wandb_run:
        if original == True:
            wandb_run.log({f"Testing_{train_config.dataset}/original": wan_table})
        else:
            wandb_run.log({f"Testing_{train_config.dataset}/fine-tuned": wan_table})

    # Compute metrics
    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences, len(prompts))
    
    # Plot metrics if wandb is enabled
    if train_config.use_wandb:
        plot_confidence_histogram(y, out_confidences, "stage1", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, out_confidences, "stage1", wandb_run, original, train_config.dataset)

    # Calculate scores
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    # Log metrics to wandb
    if wandb_run:
        wandb_run.log({
                        f'gpt/number_{train_config.dataset}': number,
                        f'gpt/acc_{train_config.dataset}': val_metrics['acc'],
                        f'gpt/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'gpt/ece_{train_config.dataset}': ece_score,
                        f'gpt/auroc_{train_config.dataset}': roc_auc_score,
                        f'gpt/score_{train_config.dataset}': score,
                    }, commit=False)
    prompts2 = []
    for prompt, response in zip(prompts, responses):
        prompt[2]['content'] += (' ' + response)
        prompts2.append(prompt)
    
    original = False
    prompts2 = tokenizer.apply_chat_template(prompts2, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    llm = LLM(
        model=train_config.model_name if original else train_config.output_dir,
        tensor_parallel_size=1,
        dtype="float16",
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    outputs = llm.generate(prompts=prompts2, sampling_params=sampling_params)
    confidences = []
    import re
    for output in outputs:
        percent_str = output.outputs[0].text
        match = re.search(r"(\d+\.?\d*)%", percent_str)
        if match:
            percent = float(match.group(1)) / 100
            confidences.append(percent)
        else:
            # 处理无效值（例如设为 0 或记录警告）
            confidences.append(0.0)
            print(f"Warning: Invalid confidence format: {percent_str}")
    val_metrics = compute_conf_metrics(y, confidences, len(prompts2))
    
    # Plot metrics if wandb is enabled
    if train_config.use_wandb:
        plot_confidence_histogram(y, confidences, "stage2", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, confidences, "stage2", wandb_run, original, train_config.dataset)

    # Calculate scores
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    # Log metrics to wandb
    if wandb_run:
        wandb_run.log({
                        f'test/number_{train_config.dataset}': number,
                        f'test/acc_{train_config.dataset}': val_metrics['acc'],
                        f'test/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'test/ece_{train_config.dataset}': ece_score,
                        f'test/auroc_{train_config.dataset}': roc_auc_score,
                        f'test/score_{train_config.dataset}': score,
                    }, commit=False)
    
    return responses


def test_cross(train_config, test_dataset, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader using OpenAI API

    Args:
        train_config: The training configuration
        test_dataset: The dataset containing the test data
        tokenizer: The tokenizer used to process prompts
        wandb_run: The wandb run object for logging
        original: Whether to use the original model or the fine-tuned model

    Returns: ece_score, roc_auc_score, accuracy
    """
    
    llm = LLM(
        model=train_config.model_name,
        tensor_parallel_size=1,
        dtype="float16",
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    if wandb_run:
        if original == True:
            wandb_run.log({f"Testing_{train_config.dataset}/original": wan_table})
        else:
            wandb_run.log({f"Testing_{train_config.dataset}/fine-tuned": wan_table})

    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences, len(prompts))
    if train_config.use_wandb:
        plot_confidence_histogram(y, out_confidences, "stage1", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, out_confidences, "stage1", wandb_run, original, train_config.dataset)

    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    if wandb_run:
        wandb_run.log({
                        f'origin/number_{train_config.dataset}': number,
                        f'origin/acc_{train_config.dataset}': val_metrics['acc'],
                        f'origin/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'origin/ece_{train_config.dataset}': ece_score,
                        f'origin/auroc_{train_config.dataset}': roc_auc_score,
                        f'origin/score_{train_config.dataset}': score,
                    }, commit=False)
    del llm
    prompts2 = []
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    for prompt, response in zip(prompts, out_response_cleans):
        prompt[2]['content'] += response
        prompts2.append(prompt)
    
    original = False
    prompts2 = tokenizer.apply_chat_template(prompts2, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    llm = LLM(
        model=train_config.model_name if original else train_config.output_dir,
        tensor_parallel_size=1,
        dtype="float16",
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    outputs = llm.generate(prompts=prompts2, sampling_params=sampling_params)
    confidences = []
    import re
    for output in outputs:
        percent_str = output.outputs[0].text
        match = re.search(r"(\d+\.?\d*)%", percent_str)
        if match:
            percent = float(match.group(1)) / 100
            confidences.append(percent)
        else:
            # 处理无效值（例如设为 0 或记录警告）
            confidences.append(0.0)
            print(f"Warning: Invalid confidence format: {percent_str}")
    val_metrics = compute_conf_metrics(y, confidences, len(prompts2))
    
    # Plot metrics if wandb is enabled
    if train_config.use_wandb:
        plot_confidence_histogram(y, confidences, "stage2", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, confidences, "stage2", wandb_run, original, train_config.dataset)

    # Calculate scores
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    # Log metrics to wandb
    if wandb_run:
        wandb_run.log({
                        f'test/number_{train_config.dataset}': number,
                        f'test/acc_{train_config.dataset}': val_metrics['acc'],
                        f'test/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'test/ece_{train_config.dataset}': ece_score,
                        f'test/auroc_{train_config.dataset}': roc_auc_score,
                        f'test/score_{train_config.dataset}': score,
                    }, commit=False)
    
    return responses


def test_reflection(train_config, test_dataset, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader using OpenAI API

    Args:
        train_config: The training configuration
        test_dataset: The dataset containing the test data
        tokenizer: The tokenizer used to process prompts
        wandb_run: The wandb run object for logging
        original: Whether to use the original model or the fine-tuned model

    Returns: ece_score, roc_auc_score, accuracy
    """
    reflection_prompt =  """For the question, response, and confidence, if the confidence is less than 50%, please revise your response and provide a better one. Otherwise, please repeat the response and the confidence.

            Here is the example:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was Percy Bysshe Shelley.
            Confidence: 40%
            If the confidence is less than 50%, analyze the answer and provide a better one. 
            Reflection: The response is less than 50%. 
            Response: The author of Paradise Lost wasn't Percy Bysshe Shelley, it was John Milton, who published the book in 1667.
            Confidence: 90%
            """
    original = True
    llm = LLM(
        model=train_config.model_name,
        tensor_parallel_size=1,
        dtype="float16",
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    wandb_run.log({f"Testing_{train_config.dataset}/original": wan_table})

    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences, len(prompts))
    if train_config.use_wandb:
        plot_confidence_histogram(y, out_confidences, "stage1", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, out_confidences, "stage1", wandb_run, original, train_config.dataset)

    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    if wandb_run:
        wandb_run.log({
                        f'origin/number_{train_config.dataset}': number,
                        f'origin/acc_{train_config.dataset}': val_metrics['acc'],
                        f'origin/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'origin/ece_{train_config.dataset}': ece_score,
                        f'origin/auroc_{train_config.dataset}': roc_auc_score,
                        f'origin/score_{train_config.dataset}': score,
                    }, commit=False)
    del llm
    original = False
    prompts2 = []
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    for prompt, response, confidence in zip(prompts, out_response_cleans, out_confidences):
        if confidence < 0.5:
            prompt[0]['content'] = reflection_prompt
            prompt[1]['content'] += ('\nResponse:' + response + str(int(confidence * 100)) + '%\n' + 'The confidence is less than 50%, analyze the answer step by step and provide a better one.')
            prompt[2]['content'] = 'Reflection: The response is '
            prompts2.append(prompt)
        else:
            prompts2.append(prompt)
    
    wan_table2 = wandb.Table(columns=['response','confidence', 'y'])
    prompts2 = tokenizer.apply_chat_template(prompts2, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    llm = LLM(
        model=train_config.model_name if original else train_config.output_dir,
        tensor_parallel_size=1,
        dtype="float16",
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    outputs = llm.generate(prompts=prompts2, sampling_params=sampling_params)
    responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table2.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    wandb_run.log({f"Testing_{train_config.dataset}/fine-tuned": wan_table2})

    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences, len(prompts)) 
    # Plot metrics if wandb is enabled
    if train_config.use_wandb:
        plot_confidence_histogram(y, out_confidences, "stage2", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, out_confidences, "stage2", wandb_run, original, train_config.dataset)

    # Calculate scores
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    # Log metrics to wandb
    if wandb_run:
        wandb_run.log({
                        f'test/number_{train_config.dataset}': number,
                        f'test/acc_{train_config.dataset}': val_metrics['acc'],
                        f'test/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'test/ece_{train_config.dataset}': ece_score,
                        f'test/auroc_{train_config.dataset}': roc_auc_score,
                        f'test/score_{train_config.dataset}': score,
                    }, commit=False)
    
    return responses


def test_reflection_gpt(train_config, test_dataset, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader using OpenAI API with different numbers of low-confidence samples
    being regenerated by GPT. Optimized to call the API only once for the maximum sample count.

    Args:
        train_config: The training configuration
        test_dataset: The dataset containing the test data
        tokenizer: The tokenizer used to process prompts
        wandb_run: The wandb run object for logging
        original: Whether to use the original model or the fine-tuned model

    Returns: ece_score, roc_auc_score, accuracy
    """
    reflection_prompt =  """For the question, response, and confidence, if the confidence is less than 50%, please revise your response and provide a better one. Otherwise, please repeat the response and the confidence.

            Here is the example:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was Percy Bysshe Shelley.
            Confidence: 40%
            If the confidence is less than 50%, analyze the answer and provide a better one. 
            Reflection: The response is less than 50%. 
            Response: The author of Paradise Lost wasn't Percy Bysshe Shelley, it was John Milton, who published the book in 1667.
            Confidence: 90%
            """
    original = True
    llm = LLM(
        model=train_config.model_name,
        tensor_parallel_size=1,
        dtype="float16",
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    responses_stage1, out_response_cleans_stage1, questions_stage1, out_confidences_stage1, y_stage1, y_None_stage1, confidences_None_stage1, correct_answer_cleans_stage1 = confidence_replace(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    for response, confidence, y_item in zip(responses_stage1, confidences_None_stage1, y_None_stage1):
        wan_table.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score for stage 1
    wandb_run.log({f"Testing_{train_config.dataset}/original": wan_table})

    number = len(y_stage1)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y_stage1, out_confidences_stage1, len(prompts))
    if train_config.use_wandb:
        plot_confidence_histogram(y_stage1, out_confidences_stage1, "stage1", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y_stage1, out_confidences_stage1, "stage1", wandb_run, original, train_config.dataset)

    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    if wandb_run:
        wandb_run.log({
                        f'origin/number_{train_config.dataset}': number,
                        f'origin/acc_{train_config.dataset}': val_metrics['acc'],
                        f'origin/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'origin/ece_{train_config.dataset}': ece_score,
                        f'origin/auroc_{train_config.dataset}': roc_auc_score,
                        f'origin/score_{train_config.dataset}': score,
                    }, commit=False)
    del llm
    
    # 获取所有样本的置信度和对应的索引
    confidence_with_indices = [(conf, i) for i, conf in enumerate(out_confidences_stage1)]
    # 按置信度升序排序
    confidence_with_indices.sort(key=lambda x: x[0])
    
    # 定义要测试的样本数量
    sample_counts = [10, 50, 80, 100, 200, 300, 400]
    # 确保最大数量不超过总样本数
    max_count = min(max(sample_counts), len(confidence_with_indices))
    original = False
    
    # 创建一个表格来存储不同样本数量的指标
    metrics_table = wandb.Table(columns=["sample_count", "acc", "acc2", "ece", "auroc", "score"])
    
    # 只生成一次最大数量的样本
    print(f"\n===== Generating responses for {max_count} lowest confidence samples =====")
    
    # 选择置信度最低的指定数量样本的索引
    lowest_conf_indices = [idx for _, idx in confidence_with_indices[:max_count]]
    
    # 为置信度最低的样本准备提示
    prompts_low_conf = []
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    for idx in lowest_conf_indices:
        prompt = prompts[idx].copy()
        # prompt[0]['content'] = reflection_prompt
        # prompt[1]['content'] += ('\nResponse:' + out_response_cleans_stage1[idx] + str(int(out_confidences_stage1[idx] * 100)) + '%\n' + 'The confidence is less than 50%, analyze the answer step by step and provide a better one.')
        # prompt[2]['content'] = 'Reflection: The response is '
        prompts_low_conf.append(prompt)
    
    # 使用GPT API生成置信度低的样本的新回答
    gpt_responses = []
    gpt_confidences = []
    gpt_y = []
    
    # 调用GPT API生成置信度最低的样本的新回答
    if prompts_low_conf:
        try:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.environ['OPENAI_API_KEY'],
                api_version=os.environ['OPENAI_API_VERSION'],
                azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
            )
        except:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],
            )
        
        outputs_low_conf = []
        for i, prompt in enumerate(prompts_low_conf):
            try:
                print(f"Generating response for sample {i+1}/{len(prompts_low_conf)}")
                response = client.chat.completions.create(
                    model=os.environ.get('OPENAI_DEPLOYMENT_NAME'),
                    messages=prompt,
                    temperature=train_config.temperature,
                    max_tokens=400,
                    top_p=1.0,
                    n=1
                )
                outputs_low_conf.append(response.choices[0].message.content)
            except Exception as e:
                print(f"Error generating response with GPT API: {e}")
                outputs_low_conf.append("")
        
        # 提取低置信度样本对应的问题和正确答案
        questions_low_conf = [test_dataset['question'][idx] for idx in lowest_conf_indices]
        correct_answers_low_conf = [test_dataset['correct_answer'][idx] for idx in lowest_conf_indices]
        
        # 处理GPT生成的结果
        responses_stage2, out_response_cleans_stage2, questions_stage2, out_confidences_stage2, y_stage2, y_None_stage2, confidences_None_stage2, correct_answer_cleans_stage2 = confidence_replace_gpt(questions_low_conf, outputs_low_conf, correct_answers_low_conf, dataset_name=train_config.dataset, vllm=False)
        
        # 存储GPT生成的结果
        for i in range(len(responses_stage2)):
            gpt_responses.append(responses_stage2[i])
            gpt_confidences.append(out_confidences_stage2[i])
            gpt_y.append(y_stage2[i])
    
    # 为每个样本数量进行评估
    for count in sample_counts:
        # 确保样本数量不超过总样本数和最大生成数量
        actual_count = min(count, max_count, len(confidence_with_indices))
        print(f"\n===== Evaluating with {actual_count} lowest confidence samples =====")
        
        # 选择置信度最低的指定数量样本的索引
        current_lowest_indices = [idx for _, idx in confidence_with_indices[:actual_count]]
        # 其余样本的索引
        other_indices = [idx for idx in range(len(out_confidences_stage1)) if idx not in lowest_conf_indices[:actual_count]]
        
        # 创建一个表格记录当前样本数量的结果
        wan_table2 = wandb.Table(columns=['response', 'confidence', 'y', 'source'])
        
        # 记录两个阶段的所有结果
        final_responses = []
        final_confidences = []
        final_y = []
        
        # 首先添加其他样本的原始结果
        for idx in other_indices:
            final_responses.append(responses_stage1[idx])
            final_confidences.append(out_confidences_stage1[idx])
            final_y.append(y_stage1[idx])
            # wan_table2.add_data(responses_stage1[idx], confidences_None_stage1[idx], y_None_stage1[idx], "stage1-original")
        
        # 添加GPT生成的结果（仅使用当前样本数量对应的部分）
        for i in range(min(actual_count, len(gpt_responses))):
            final_responses.append(gpt_responses[i])
            final_confidences.append(gpt_confidences[i])
            final_y.append(gpt_y[i])
            # wan_table2.add_data(gpt_responses[i], gpt_confidences[i], gpt_y[i], "stage2-gpt")
        
        # 记录这个特定样本数量的结果
        # wandb_run.log({f"Testing_{train_config.dataset}/samples_{actual_count}": wan_table2})

        # 计算合并后的指标
        val_metrics = compute_conf_metrics(final_y, final_confidences, len(final_y))
        
        # 可视化合并后的结果
        if train_config.use_wandb:
            plot_confidence_histogram(final_y, final_confidences, f"combined_{actual_count}", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
            plot_ece_diagram(final_y, final_confidences, f"combined_{actual_count}", wandb_run, original, train_config.dataset)

        # 计算最终分数
        ece_score = val_metrics['ece']
        roc_auc_score = val_metrics['auroc']
        score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

        # 记录这个样本数量的指标
        metrics_dict = {
            f'test_{actual_count}/number_{train_config.dataset}': len(final_y),
            f'test_{actual_count}/acc_{train_config.dataset}': val_metrics['acc'],
            f'test_{actual_count}/acc2_{train_config.dataset}': val_metrics['acc2'],
            f'test_{actual_count}/ece_{train_config.dataset}': ece_score,
            f'test_{actual_count}/auroc_{train_config.dataset}': roc_auc_score,
            f'test_{actual_count}/score_{train_config.dataset}': score,
            f'test_{actual_count}/low_conf_count': actual_count,
            f'test_{actual_count}/high_conf_count': len(other_indices),
        }
        wandb_run.log(metrics_dict, commit=False)
        
        # 将指标添加到汇总表格
        metrics_table.add_data(actual_count, val_metrics['acc'], val_metrics['acc2'], ece_score, roc_auc_score, score)
        
        # 打印此次评估的指标
        print(f"Sample count: {actual_count}")
        print(f"Accuracy: {val_metrics['acc']:.4f}")
        print(f"Accuracy (correct with conf): {val_metrics['acc2']:.4f}")
        print(f"ECE: {ece_score:.4f}")
        print(f"AUROC: {roc_auc_score:.4f}")
        print(f"Score: {score:.4f}")
    
    # 记录汇总表格
    wandb_run.log({f"metrics_summary_{train_config.dataset}": metrics_table})
    
    # 打印不同修改数目对应的准确率汇总表格
    print("\n====== 不同修改数目对应的指标汇总 ======")
    print("| 修改样本数 | Accuracy | Accuracy2 | ECE  | AUROC | 总分数 |")
    print("|------------|----------|-----------|------|-------|--------|")
    for count, acc, acc2, ece, auroc, score in metrics_table.data:
        print(f"| {count:10d} | {acc:.4f}  | {acc2:.4f}   | {ece:.4f} | {auroc:.4f} | {score:.4f} |")
    print("================================================")
    
    # 返回最后一次评估的结果
    return final_responses


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
