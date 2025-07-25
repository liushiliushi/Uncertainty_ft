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
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
import torch.nn.functional as F
import torch.nn as nn
import wandb
import sys

from llama_recipes.model_checkpointing import save_peft_checkpoint, save_model_checkpoint, save_merged_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_recipes.utils.flop_utils import FlopMeasure
from llama_recipes.utils.compute_metrics import compute_conf_metrics, plot_confidence_histogram, plot_ece_diagram
from llama_recipes.utils.postprocess import postprocess_extract, confidence_replace
from vllm import LLM, SamplingParams

class ConfidenceClassifier(nn.Module):
    """
    置信度分类器：将token概率分布转换为0-100的置信度预测
    """
    def __init__(self, vocab_size, hidden_dim=512, num_classes=101):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes  # 0-100，共101个类别
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 分类器头
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, token_logits):
        """
        Args:
            token_logits: [batch_size, vocab_size] - 最后一个token的logits
        Returns:
            confidence_logits: [batch_size, num_classes] - 置信度分类logits
        """
        # 将logits转换为概率分布作为特征
        token_probs = F.softmax(token_logits, dim=-1)
        
        # 特征提取
        features = self.feature_extractor(token_probs)
        
        # 分类预测
        confidence_logits = self.classifier(features)
        
        return confidence_logits

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

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
    train_dataloader,
    eval_dataloader,
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
    
    # 初始化置信度分类器
    vocab_size = len(tokenizer)
    confidence_classifier = ConfidenceClassifier(vocab_size=vocab_size).to(model.device)
    
    # 分类器优化器
    classifier_optimizer = torch.optim.AdamW(
        confidence_classifier.parameters(), 
        lr=train_config.lr, 
        weight_decay=train_config.weight_decay
    )
    
    # 准备分类器参与accelerate
    confidence_classifier, classifier_optimizer = accelerator.prepare(
        confidence_classifier, classifier_optimizer
    )
    
    accelerator.print(f"Confidence Classifier initialized with vocab_size={vocab_size}")
    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 80,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0.1,
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
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(train_dataloader, colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True, disable=not accelerator.is_local_main_process)
            for step, batch in enumerate(pbar): 
                total_train_steps += 1
                # stop when the maximum number of training steps is reached
                if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                    max_steps_reached = True
                    accelerator.print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)

                y = batch.data.pop('y')

                with autocast():
                    # get the
                    output1 = model(**batch)
                    logits1 = output1.logits
                    loss_con = output1.loss

                # 获取最后一个token的logits，用作分类器的输入特征
                num_token1 = logits1[:,-1,:] # [batch_size, vocab_size]
                del logits1
                
                # 使用分类器预测置信度
                confidence_logits = confidence_classifier(num_token1)  # [batch_size, 101]
                
                # 将logits转换为概率分布
                confidence_probs = F.softmax(confidence_logits, dim=1)  # [batch_size, 101]
                
                # 创建置信度值数组 [0.00, 0.01, 0.02, ..., 1.00]
                confidence_values = torch.arange(0, 1.01, 0.01).view(1, 101).to(model.device)  # [1, 101]
                
                # 计算期望的置信度预测值
                predicted_confidence = torch.sum(confidence_probs * confidence_values, dim=1)  # [batch_size]
                
                # 计算Brier Score损失
                squared_differences = (predicted_confidence - y.squeeze()) ** 2
                loss_cal = torch.mean(squared_differences)
                
                if train_config.add_loss_con:
                    loss = loss_con + loss_cal
                else:
                    loss = loss_cal
                accelerator.backward(loss)
                optimizer.step()
                classifier_optimizer.step()
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                total_loss += loss.detach().float()

                if wandb_run and accelerator.is_main_process:
                    wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': loss.detach().float(),
                            'train/loss_con': loss_con.detach().float(),
                                'train/loss_cal': loss_cal.detach().float()
                        })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    # torch.cuda.empty_cache()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        avg_loss = total_loss / len(train_dataloader)
        avg_ppl = float(torch.exp(torch.tensor(avg_loss)))  # ppl = exp(loss)

        # Update the learning rate as needed
        lr_scheduler.step()
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, _, _ = evaluation_chat(
                model,
                train_config,
                eval_dataloader,
                tokenizer,
                accelerator,
                wandb_run,
                confidence_classifier=confidence_classifier,
            )
            if True:
                if accelerator.is_main_process:
                    best_val_loss = eval_epoch_loss
                if train_config.save_model and accelerator.is_main_process:
                    # 保存分类器参数
                    classifier_path = os.path.join(train_config.output_dir, "confidence_classifier.pt")
                    os.makedirs(train_config.output_dir, exist_ok=True)
                    
                    # 获取分类器的状态字典（处理accelerate包装）
                    if hasattr(confidence_classifier, "module"):
                        classifier_state_dict = confidence_classifier.module.state_dict()
                    else:
                        classifier_state_dict = confidence_classifier.state_dict()
                    
                    torch.save({
                        'model_state_dict': classifier_state_dict,
                        'optimizer_state_dict': classifier_optimizer.state_dict() if hasattr(classifier_optimizer, 'state_dict') else None,
                        'vocab_size': len(tokenizer),
                        'epoch': epoch,
                    }, classifier_path)
                    accelerator.print(f"Confidence classifier saved to {classifier_path}")

    results = None
   

    return results

def train_gpt(
    model,
    train_dataloader,
    eval_dataloaders_dict,  # Changed to dictionary of dataset_name -> eval_dataloader
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
    numbers = [str(i) for i in range(10)]
    token_ids1 = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    token_ids2 = [tokenizer.encode('0', add_special_tokens=False)[0], tokenizer.eos_token_id]
    print("----------------------------")
    print(token_ids1)
    num_indices1 = torch.tensor(token_ids1).to(model.device)
    num_indices2 = torch.tensor(token_ids2).to(model.device)
    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 80,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0.1,
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
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(train_dataloader, colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True, disable=not accelerator.is_local_main_process)
            for step, batch in enumerate(pbar): 
                total_train_steps += 1
                # stop when the maximum number of training steps is reached
                if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                    max_steps_reached = True
                    accelerator.print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)

                y = batch.data.pop('y')

                with autocast():
                    # get the
                    output1 = model(**batch)
                    logits1 = output1.logits
                    loss_con = output1.loss

                num_token1 = logits1[:,-1,:] # get the logit of the confidence token
                del logits1
                scores = torch.arange(0, 1, 0.1).view(1, 10).expand(y.shape[0], 10).to(model.device)

                if train_config.loss_type == 'brier':
                    num_conf1 = torch.index_select(num_token1, 1, num_indices1.squeeze(0)) # take out the logit of 0-9+1
                    num_conf1 = F.softmax(num_conf1, dim=1)
                    y_expanded = y.expand(y.shape[0], 10)
                    squared_differences = (y_expanded - scores) ** 2
                    loss_cal = torch.mean(torch.sum(num_conf1 * squared_differences, dim=1))
                elif train_config.loss_type == 'sot':
                    norm_logit = torch.index_select(F.log_softmax(num_token1, dim=1), 1, num_indices1.squeeze(0))
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
                            'train/loss_con': loss_con.detach().float(),
                                'train/loss_cal': loss_cal.detach().float()
                        })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    # torch.cuda.empty_cache()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        avg_loss = total_loss / len(train_dataloader)
        avg_ppl = float(torch.exp(torch.tensor(avg_loss)))  # ppl = exp(loss)

        # Update the learning rate as needed
        lr_scheduler.step()
        if train_config.run_validation and epoch == (train_config.num_epochs - 1):
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
            if accelerator.is_main_process:
                # Log aggregate metrics
                if wandb_run:
                    wandb_run.log({
                        'eval/summary/total_score': total_eval_score,
                    })
                    accelerator.print(f"Evaluation Total Score: {total_eval_score:.4f}")

            

                if True:
                    if train_config.save_model:
                        if train_config.use_peft:
                            if hasattr(model, "module"):
                                model = model.module
                            if train_config.merge_peft and epoch == (train_config.num_epochs - 1):
                                # 清理GPU缓存以释放内存
                                clear_gpu_cache()
                                try:
                                    accelerator.print("开始合并PEFT模型...请耐心等待")
                                    # 先转到CPU合并，减少GPU内存使用
                                    model = model.to(dtype=torch.float32)
                                    model = model.cpu()
                                    model = model.merge_and_unload()
                                    # 合并后再转回原始精度和设备
                                    model = model.to(dtype=torch.float16)
                                    if torch.cuda.is_available():
                                        model = model.to("cuda")
                                    accelerator.print("PEFT模型合并完成，开始保存...")
                                    save_merged_checkpoint(model, tokenizer, train_config.output_dir)
                                    accelerator.print(f"Merged modules are saved in {train_config.output_dir} directory")
                                except Exception as e:
                                    accelerator.print(f"Error during model merge and save: {str(e)}")
                            else:
                                save_peft_checkpoint(model, train_config.output_dir)
                                accelerator.print(f"PEFT modules are saved in {train_config.output_dir} directory")
                        else:
                            save_model_checkpoint(model, train_config.output_dir)
                            accelerator.print(f"Model is saved in {train_config.output_dir} directory")
                    # 移除sys.exit(0)调用
    
    results = None
   

    return results


def evaluation_chat(
    model,
    train_config,
    eval_dataloader,
    tokenizer,
    accelerator,
    wandb_run=None,
    original=False,
    confidence_classifier=None
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
    
    # 如果没有提供分类器，创建一个默认的分类器用于评估
    if confidence_classifier is None:
        vocab_size = len(tokenizer)
        confidence_classifier = ConfidenceClassifier(vocab_size=vocab_size).to(model.device)
        accelerator.print("Warning: Using default classifier for evaluation")
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
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            y = batch.data.pop('y')
            output = model(**batch)
            logits = output.logits
            loss_con = output.loss
            num_token = logits[:,-1,:] # get the logit of the confidence token
            del logits
            
            # 使用分类器预测置信度
            confidence_logits = confidence_classifier(num_token)  # [batch_size, 101]
            
            # 将logits转换为概率分布
            confidence_probs = F.softmax(confidence_logits, dim=1)  # [batch_size, 101]
            
            # 创建置信度值数组 [0.00, 0.01, 0.02, ..., 1.00]
            confidence_values = torch.arange(0, 1.01, 0.01).view(1, 101).to(model.device)  # [1, 101]
            
            # 计算期望的置信度预测值
            predicted_confidence = torch.sum(confidence_probs * confidence_values, dim=1)  # [batch_size]
            
            # 计算Brier Score损失
            squared_differences = (predicted_confidence - y.squeeze()) ** 2
            loss_cal = torch.mean(squared_differences)
            
            if train_config.save_metrics:
                val_step_loss.append(loss.detach().float().item())
                val_step_perplexity.append(float(torch.exp(loss.detach().float())))

            if train_config.add_loss_con:
                loss = loss_con + loss_cal
            else:
                loss = loss_cal
            print(f"loss: {loss} loss_con: {loss_con} loss_cal: {loss_cal}") 
               
            eval_loss += loss.detach().float()
            eval_loss_con += loss_con.detach().float()
            eval_loss_cal += loss_cal.detach().float()
            
            # 预测置信度：使用期望值（与训练时一致）
            probs = predicted_confidence  # 已经是0-1范围的置信度

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

def test_vllm(train_config, test_dataset, tokenizer, wandb_run, original=False, classifier_path=None):
    """
    Evaluates the model using VLLM and classifier for confidence prediction

    Args:
        train_config: training configuration
        test_dataset: test dataset 
        tokenizer: tokenizer for the model
        wandb_run: wandb run for logging
        original: whether to use original model
        classifier_path: path to confidence classifier checkpoint

    Returns: ece_score, roc_auc_score, acc2_score
    """
    from transformers import AutoModelForCausalLM
    
    all_y = []
    test_probs = []
    all_responses = []
    
    # 加载分类器（如果提供路径）
    confidence_classifier = None
    if classifier_path and os.path.exists(classifier_path):
        confidence_classifier, _ = load_confidence_classifier(classifier_path, device='cuda')
        confidence_classifier.eval()
        print(f"Loaded confidence classifier from {classifier_path}")
        
        # 加载模型用于获取logits
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name if original else train_config.output_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print("Using classifier for confidence prediction")
    else:
        print("No classifier provided, using original confidence extraction method")
    
    # 使用VLLM生成回答
    llm = LLM(
        model=train_config.model_name if original else train_config.output_dir,
        tensor_parallel_size=1,
        dtype="float16",
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,  # 降低一点以留出内存给分类器
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=train_config.temperature,
        top_k=-1,
        top_p=1.0,
        max_tokens=400
    )

    wan_table = wandb.Table(columns=['response', 'confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    prompts_text = tokenizer.apply_chat_template(
        prompts, tokenize=False, padding="longest", 
        truncation=True, return_tensors="pt", continue_final_message=True
    )
    
    # VLLM生成回答
    outputs = llm.generate(prompts=prompts_text, sampling_params=sampling_params)
    
    # 提取回答文本
    responses = [output.outputs[0].text for output in outputs]
    
    # 处理每个回答以获取置信度标签
    responses_filtered, out_response_cleans, questions, confidence_stage1, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(
        test_dataset['question'], responses, test_dataset['correct_answer'], 
        dataset_name=train_config.dataset, vllm=True
    )
    
    if confidence_classifier is not None and len(responses_filtered) > 0:
        print("Using classifier to predict confidence...")
        
        # 为带有置信度问题的回答重新获取logits
        batch_size = 8
        classifier_confidences = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(responses_filtered), batch_size), desc="Classifier inference"):
                batch_end = min(i + batch_size, len(responses_filtered))
                batch_prompts = responses_filtered[i:batch_end]
                
                # 构建带有置信度问题的prompts
                confidence_prompts = []
                for j, response in enumerate(batch_prompts):
                    # 获取对应的问题和回答
                    question = questions[i+j] if i+j < len(questions) else ""
                    
                    # 构建置信度问题的prompt
                    confidence_prompt = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": f"{response}\n\nHow confident are you in this answer? Please provide a confidence level from 0 to 100: "}
                    ]
                    confidence_prompts.append(confidence_prompt)
                
                # Tokenize
                query_tensors = tokenizer.apply_chat_template(
                    confidence_prompts,
                    tokenize=True,
                    padding="longest",
                    padding_side='left',
                    truncation=True,
                    return_dict=True,
                    return_tensors="pt",
                    continue_final_message=True
                ).to(model.device)
                
                # 添加空格token
                white_spaces = torch.full((len(confidence_prompts), 1), 220).to(model.device)
                white_attention = torch.full((len(confidence_prompts), 1), 1).to(model.device)
                query_tensors['input_ids'] = torch.cat([query_tensors['input_ids'], white_spaces], dim=1)
                query_tensors['attention_mask'] = torch.cat([query_tensors['attention_mask'], white_attention], dim=1)
                
                # 获取logits
                logits = model(**query_tensors).logits
                
                # 使用分类器预测置信度
                num_token = logits[:, -1, :]  # [batch_size, vocab_size]
                confidence_logits = confidence_classifier(num_token)  # [batch_size, 101]
                
                # 计算期望置信度
                confidence_probs = F.softmax(confidence_logits, dim=1)
                confidence_values = torch.arange(0, 1.01, 0.01).view(1, 101).to(model.device)
                predicted_confidence = torch.sum(confidence_probs * confidence_values, dim=1)
                
                classifier_confidences.extend(predicted_confidence.detach().cpu().numpy().tolist())
        
        # 使用分类器的置信度预测
        out_confidences = classifier_confidences
        print(f"Generated {len(out_confidences)} confidence predictions using classifier")
        
    else:
        # 使用原始的置信度提取方法
        out_confidences = confidence_stage1
        print("Using original confidence extraction method")
    
    # 记录到wandb表格
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table.add_data(response, confidence, y_item)
    
    if wandb_run:
        if original:
            wandb_run.log({f"Testing_{train_config.dataset}/original": wan_table})
        else:
            classifier_suffix = "_classifier" if confidence_classifier is not None else ""
            wandb_run.log({f"Testing_{train_config.dataset}/fine-tuned{classifier_suffix}": wan_table})

    # 计算指标
    if len(y) == 0:
        print("No valid samples for evaluation")
        return None, None, None
    
    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences, len(prompts))
    
    if train_config.use_wandb:
        method_name = "classifier" if confidence_classifier is not None else "stage1"
        plot_confidence_histogram(
            y, out_confidences, method_name, 
            val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], 
            wandb_run, original, train_config.dataset, use_annotation=True
        )
        plot_ece_diagram(y, out_confidences, method_name, wandb_run, original, train_config.dataset)

    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    acc2_score = val_metrics['acc2']
    score = 3 * acc2_score + 2 * roc_auc_score - ece_score
    
    print(f"Results ({'Classifier' if confidence_classifier else 'Original'}):")
    print(f"  ECE: {ece_score:.4f}")
    print(f"  ROC-AUC: {roc_auc_score:.4f}")
    print(f"  Accuracy: {acc2_score:.4f}")
    print(f"  Score: {score:.4f}")

    if wandb_run:
        method_suffix = "_classifier" if confidence_classifier is not None else ""
        wandb_run.log({
            f'test/number_{train_config.dataset}{method_suffix}': number,
            f'test/acc_{train_config.dataset}{method_suffix}': val_metrics['acc'],
            f'test/acc2_{train_config.dataset}{method_suffix}': acc2_score,
            f'test/ece_{train_config.dataset}{method_suffix}': ece_score,
            f'test/auroc_{train_config.dataset}{method_suffix}': roc_auc_score,
            f'test/score_{train_config.dataset}{method_suffix}': score,
        }, commit=False)

    return ece_score, roc_auc_score, acc2_score

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

def load_confidence_classifier(checkpoint_path, device='cuda'):
    """
    加载保存的置信度分类器
    
    Args:
        checkpoint_path: 分类器检查点路径
        device: 设备类型
    
    Returns:
        confidence_classifier: 加载的分类器模型
        checkpoint_info: 检查点信息字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab_size = checkpoint['vocab_size']
    confidence_classifier = ConfidenceClassifier(vocab_size=vocab_size)
    confidence_classifier.load_state_dict(checkpoint['model_state_dict'])
    confidence_classifier.to(device)
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', -1),
        'vocab_size': vocab_size
    }
    
    print(f"Loaded confidence classifier from {checkpoint_path}")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Epoch: {checkpoint_info['epoch']}")
    
    return confidence_classifier, checkpoint_info
