# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib
import re
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
from llama_recipes.utils.postprocess import postprocess_extract, confidence_replace, confidence_replace_classifier
from vllm import LLM, SamplingParams

def balance_batch_data(batch, y, target_ratio=0.6):
    """
    平衡batch数据，确保每个batch都包含正负样本
    
    Args:
        batch: 当前batch数据
        y: 标签张量 [batch_size, 1]
        target_ratio: 目标主要类别的最大比例（默认0.6，即60%）
    
    Returns:
        (balanced_batch, balanced_y) 或 None（如果无法平衡）
    """
    try:
        # 获取标签和索引
        unique_labels, counts = torch.unique(y, return_counts=True)
        
        if len(unique_labels) == 1:
            # 如果只有一种标签，无法平衡
            return None
        
        batch_size = len(y)
        label_ratios = counts.float() / batch_size
        
        # 找出主要类别和次要类别
        majority_label = unique_labels[torch.argmax(counts)]
        minority_label = unique_labels[torch.argmin(counts)]
        
        majority_indices = (y.squeeze() == majority_label).nonzero(as_tuple=False).squeeze()
        minority_indices = (y.squeeze() == minority_label).nonzero(as_tuple=False).squeeze()
        
        # 确保indices是一维张量
        if majority_indices.dim() == 0:
            majority_indices = majority_indices.unsqueeze(0)
        if minority_indices.dim() == 0:
            minority_indices = minority_indices.unsqueeze(0)
            
        # 计算目标数量
        target_majority_count = min(int(batch_size * target_ratio), len(majority_indices))
        target_minority_count = batch_size - target_majority_count
        
        # 如果次要类别样本不够，调整
        if len(minority_indices) < target_minority_count:
            target_minority_count = len(minority_indices)
            target_majority_count = batch_size - target_minority_count
        
        # 随机选择样本
        if len(majority_indices) > target_majority_count:
            majority_selected = majority_indices[torch.randperm(len(majority_indices))[:target_majority_count]]
        else:
            majority_selected = majority_indices
            
        if len(minority_indices) > target_minority_count:
            minority_selected = minority_indices[torch.randperm(len(minority_indices))[:target_minority_count]]
        else:
            minority_selected = minority_indices
        
        # 合并选中的索引
        selected_indices = torch.cat([majority_selected, minority_selected])
        
        # 如果选中的样本数太少，返回None
        if len(selected_indices) < max(4, batch_size // 2):
            return None
        
        # 重新排列选中的索引
        selected_indices = selected_indices[torch.randperm(len(selected_indices))]
        
        # 创建平衡的batch
        balanced_y = y[selected_indices]
        
        # 创建平衡的batch数据
        balanced_batch = type(batch)()
        balanced_batch.data = {}
        
        for key, value in batch.data.items():
            if key != 'y':  # y已经被pop出来了
                if isinstance(value, torch.Tensor):
                    balanced_batch.data[key] = value[selected_indices]
                elif isinstance(value, list):
                    balanced_batch.data[key] = [value[i] for i in selected_indices.cpu().numpy()]
                else:
                    # 对于其他类型，尝试索引操作
                    try:
                        balanced_batch.data[key] = [value[i] for i in selected_indices.cpu().numpy()]
                    except:
                        # 如果无法索引，保持原值
                        balanced_batch.data[key] = value
        
        # 复制其他属性
        for attr_name in dir(batch):
            if not attr_name.startswith('_') and attr_name != 'data' and hasattr(batch, attr_name):
                attr_value = getattr(batch, attr_name)
                if not callable(attr_value):
                    setattr(balanced_batch, attr_name, attr_value)
        
        return balanced_batch, balanced_y
        
    except Exception as e:
        # 如果平衡过程出错，返回None
        print(f"Error in balance_batch_data: {str(e)}")
        return None

class ConfidenceClassifier(nn.Module):
    """
    置信度分类器：将token概率分布转换为0-100的置信度预测
    """
    def __init__(self, vocab_size, hidden_dim=1024, num_classes=101):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes  # 0-100，共101个类别
        
        # 更深的特征提取层，增加网络容量
        self.feature_extractor = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加层归一化
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 增加一层
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 分类器头
        self.classifier = nn.Linear(hidden_dim // 4, num_classes)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重，使用更积极的初始化策略"""
        # 初始化特征提取层 - 使用标准的Xavier初始化
        for i, module in enumerate(self.feature_extractor):
            if isinstance(module, nn.Linear):
                # 使用标准的Xavier/Glorot初始化
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # 分类器层使用He初始化，并添加小的随机偏置来打破对称性
        nn.init.kaiming_uniform_(self.classifier.weight, mode='fan_in', nonlinearity='linear')
        # 给偏置添加小的随机值，而不是全零，有助于打破对称性
        nn.init.uniform_(self.classifier.bias, -0.1, 0.1)
        
    def forward(self, token_logits):
        """
        Args:
            token_logits: [batch_size, vocab_size] - 最后一个token的logits
        Returns:
            confidence_logits: [batch_size, num_classes] - 置信度分类logits
        """
        # 检查输入是否有异常值
        if torch.isnan(token_logits).any() or torch.isinf(token_logits).any():
            print("Warning: NaN or Inf in input token_logits")
            token_logits = torch.where(torch.isnan(token_logits) | torch.isinf(token_logits), 
                                     torch.zeros_like(token_logits), token_logits)
        
        # 数值稳定的softmax计算
        token_logits_clamped = torch.clamp(token_logits, min=-50, max=50)
        token_probs = F.softmax(token_logits_clamped, dim=-1)
        
        # 检查概率是否有异常值
        if torch.isnan(token_probs).any():
            print("Warning: NaN in token_probs, using uniform distribution")
            token_probs = torch.ones_like(token_probs) / token_probs.size(-1)
        
        # 特征提取
        features = self.feature_extractor(token_probs)
        
        # 检查特征是否有异常值
        if torch.isnan(features).any():
            print("Warning: NaN in features")
            features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
        
        # 分类预测
        confidence_logits = self.classifier(features)
        
        # 限制输出范围，防止极端值
        confidence_logits = torch.clamp(confidence_logits, min=-10, max=10)
        
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
    confidence_classifier = ConfidenceClassifier(vocab_size=vocab_size)
    
    # 为了数值稳定性，分类器使用float32，而不是float16
    model_dtype = next(model.parameters()).dtype
    classifier_dtype = torch.float32  # 强制使用float32以获得更好的数值稳定性
    confidence_classifier = confidence_classifier.to(classifier_dtype)
    
    confidence_classifier = confidence_classifier.to(model.device)
    
    accelerator.print(f"Model dtype: {model_dtype}, Classifier dtype: {classifier_dtype}")
    
    # 分类器优化器 - 使用更积极的学习率
    # 给分类器设置一个更合理的学习率范围
    min_classifier_lr = 1e-4  # 提高最小学习率
    max_classifier_lr = 1e-3  # 设置最大学习率，防止过大
    classifier_lr = min(max(train_config.lr * 2.0, min_classifier_lr), max_classifier_lr)  # 分类器学习率为主模型的2倍，但有上下限
    
    classifier_optimizer = torch.optim.AdamW(
        confidence_classifier.parameters(), 
        lr=classifier_lr, 
        weight_decay=train_config.weight_decay * 0.1,  # 降低权重衰减
        eps=1e-8,  # 增加数值稳定性
        betas=(0.9, 0.999)
    )
    
    accelerator.print(f"Classifier learning rate: {classifier_lr}")
    accelerator.print(f"Main model learning rate: {train_config.lr}")
    accelerator.print(f"Classifier weight decay: {train_config.weight_decay * 0.1}")
    
    # 分类器学习率调度器：使用带预热的余弦调度
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    # 预热阶段：前10%的steps使用线性增长
    warmup_steps = max(1, int(0.1 * train_config.num_epochs * len(train_dataloader)))
    # 主要阶段：使用余弦衰减
    main_steps = train_config.num_epochs * len(train_dataloader) - warmup_steps
    
    warmup_scheduler = LinearLR(classifier_optimizer, start_factor=0.1, total_iters=warmup_steps)
    main_scheduler = CosineAnnealingLR(classifier_optimizer, T_max=main_steps, eta_min=classifier_lr * 0.1)
    
    classifier_lr_scheduler = SequentialLR(
        classifier_optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_steps]
    )
    
    # 准备分类器参与accelerate
    confidence_classifier, classifier_optimizer = accelerator.prepare(
        confidence_classifier, classifier_optimizer
    )
    
    accelerator.print(f"Confidence Classifier initialized with vocab_size={vocab_size}, dtype={model_dtype}")
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
        current_lr = classifier_optimizer.param_groups[0]['lr']
        accelerator.print(f"Epoch {epoch+1}/{train_config.num_epochs} - Current classifier learning rate: {current_lr:.6f}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            # 设置模型状态：LLM为eval模式（不训练），分类器为train模式
            model.eval()  # LLM不训练，只用于推理
            confidence_classifier.train()  # 只训练分类器
            
            # 冻结LLM参数
            for param in model.parameters():
                param.requires_grad = False
            
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

                # LLM推理：不计算梯度
                with torch.no_grad():
                    with autocast():
                        output1 = model(**batch)
                        logits1 = output1.logits
                        loss_con = output1.loss

                # 获取最后一个token的logits，用作分类器的输入特征
                num_token1 = logits1[:,-1,:] # [batch_size, vocab_size]
                del logits1
                
                # 调试信息：检查数据类型匹配和输入特征
                if total_train_steps == 1:  # 只在第一步打印，避免刷屏
                    accelerator.print(f"Input dtype: {num_token1.dtype}, Classifier dtype: {next(confidence_classifier.parameters()).dtype}")
                    accelerator.print(f"LLM requires_grad: {next(model.parameters()).requires_grad}")
                    accelerator.print(f"Classifier requires_grad: {next(confidence_classifier.parameters()).requires_grad}")
                    accelerator.print(f"Input token logits stats: min={num_token1.min():.4f}, max={num_token1.max():.4f}, mean={num_token1.mean():.4f}")
                    accelerator.print(f"Input has NaN: {torch.isnan(num_token1).any()}, Input has Inf: {torch.isinf(num_token1).any()}")
                    accelerator.print(f"Y values stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.float().mean():.4f}")
                    accelerator.print(f"Y has NaN: {torch.isnan(y).any()}")
                    
                    # 检查分类器权重
                    for name, param in confidence_classifier.named_parameters():
                        if 'classifier' in name:  # 只检查最后的分类器层
                            accelerator.print(f"Classifier layer {name}: min={param.min():.4f}, max={param.max():.4f}, mean={param.mean():.4f}")
                            accelerator.print(f"Classifier layer {name} has NaN: {torch.isnan(param).any()}")
                
                # 分类器训练：计算梯度（不使用autocast，因为分类器是float32）
                # 将输入转换为float32以匹配分类器
                num_token1_float32 = num_token1.detach().float()  # 转换为float32并断开梯度
                y_float32 = y.float()  # 转换标签为float32
                
                # 使用分类器预测置信度
                confidence_logits = confidence_classifier(num_token1_float32)
                    
                # 检查训练时的logits异常值
                if total_train_steps <= 3:  # 只在前几步打印调试信息
                    accelerator.print(f"Train step {total_train_steps} - Confidence logits stats: min={confidence_logits.min():.4f}, max={confidence_logits.max():.4f}")
                    accelerator.print(f"Train step {total_train_steps} - Has NaN in logits: {torch.isnan(confidence_logits).any()}")
                
                # 数值稳定的softmax计算
                confidence_logits = torch.clamp(confidence_logits, min=-50, max=50)  # 限制logits范围
                confidence_probs = F.softmax(confidence_logits, dim=1)  # [batch_size, 101]
                
                # 计算预测的置信度值（期望值）
                confidence_values = torch.arange(0, 1.01, 0.01, dtype=torch.float32).view(1, 101).to(model.device)
                predicted_confidence = torch.sum(confidence_probs * confidence_values, dim=1)  # [batch_size]
                
                # 输出每个batch的预测置信度和真实标签（控制输出频率）
                should_print = (total_train_steps <= 5) or (total_train_steps % 20 == 0)  # 前5步 + 每20步输出一次
                
                if should_print:
                    accelerator.print(f"\n=== Step {total_train_steps} Batch Predictions ===")
                    # 最多显示前8个样本，避免输出过长
                    max_samples_to_show = min(8, len(predicted_confidence))
                    for i in range(max_samples_to_show):
                        pred_conf = predicted_confidence[i].item()
                        true_label = y_float32[i].item()
                        accelerator.print(f"Sample {i+1}: Predicted={pred_conf:.4f}, True={true_label:.4f}, Diff={abs(pred_conf-true_label):.4f}")
                    
                    if len(predicted_confidence) > max_samples_to_show:
                        accelerator.print(f"... and {len(predicted_confidence) - max_samples_to_show} more samples")
                    
                    # 计算batch统计信息
                    batch_mae = torch.mean(torch.abs(predicted_confidence - y_float32.squeeze())).item()
                    batch_mse = torch.mean((predicted_confidence - y_float32.squeeze()) ** 2).item()
                    accelerator.print(f"Batch Stats: MAE={batch_mae:.4f}, MSE={batch_mse:.4f}")
                    accelerator.print(f"Pred range: [{predicted_confidence.min():.4f}, {predicted_confidence.max():.4f}]")
                    accelerator.print(f"True range: [{y_float32.min():.4f}, {y_float32.max():.4f}]")
                    accelerator.print("=" * 50)
                
                # 计算改进的损失函数（Brier Score + Diversity Loss + Focal Loss）
                y_expanded = y_float32.expand(y_float32.shape[0], 101)  # 使用float32版本的y
                scores = torch.arange(0, 1.01, 0.01, dtype=torch.float32).view(1, 101).expand(y_float32.shape[0], 101).to(model.device)
                squared_differences = (y_expanded - scores) ** 2
                
                # 1. Brier Score损失
                brier_losses = torch.sum(confidence_probs * squared_differences, dim=1)  # [batch_size]
                brier_loss = torch.mean(brier_losses)
                
                # 2. 多样性损失：鼓励批次内预测的多样性
                batch_diversity_loss = 0.0
                if len(predicted_confidence) > 1:
                    # 计算预测置信度的方差，方差越大越好（多样性越高）
                    pred_var = torch.var(predicted_confidence)
                    # 将方差转换为损失（方差小时损失大）
                    diversity_loss = torch.exp(-pred_var * 10)  # 10是调节因子
                    batch_diversity_loss = diversity_loss
                
                # 3. Focal Loss：对于困难样本给予更多关注
                # 计算每个类别的真实概率（one-hot编码）
                y_indices = (y_float32.squeeze() * 100).long().clamp(0, 100)  # 转换为类别索引
                y_one_hot = torch.zeros_like(confidence_probs)
                y_one_hot.scatter_(1, y_indices.unsqueeze(1), 1.0)
                
                # Focal loss with gamma=2
                gamma = 2.0
                pt = torch.sum(confidence_probs * y_one_hot, dim=1)  # 真实类别的预测概率
                focal_losses = -torch.pow(1 - pt, gamma) * torch.log(pt + 1e-8)
                focal_loss = torch.mean(focal_losses)
                
                # 组合损失
                alpha_brier = 1.0      # Brier Score权重
                alpha_diversity = 0.1  # 多样性损失权重  
                alpha_focal = 0.5      # Focal Loss权重
                
                loss_cal = (alpha_brier * brier_loss + 
                           alpha_diversity * batch_diversity_loss + 
                           alpha_focal * focal_loss)
                
                # 计算每个样本的损失（用于显示）
                individual_losses = brier_losses  # 仍然显示Brier Score作为个体损失
                
                # 输出每个数据对应的损失
                should_print_losses = (total_train_steps <= 5) or (total_train_steps % 10 == 0)  # 前5步 + 每50步输出一次
                
                if should_print_losses:
                    accelerator.print(f"\n=== Step {total_train_steps} Individual Losses ===")
                    # 最多显示前10个样本的损失，避免输出过长
                    max_samples_to_show = min(10, len(individual_losses))
                    for i in range(max_samples_to_show):
                        sample_loss = individual_losses[i].item()
                        pred_conf = predicted_confidence[i].item()
                        true_label = y_float32[i].item()
                        accelerator.print(f"Sample {i+1}: Loss={sample_loss:.6f}, Pred={pred_conf:.4f}, True={true_label:.4f}")
                    
                    if len(individual_losses) > max_samples_to_show:
                        accelerator.print(f"... and {len(individual_losses) - max_samples_to_show} more samples")
                    
                    # 统计损失分布
                    loss_min = individual_losses.min().item()
                    loss_max = individual_losses.max().item()
                    loss_std = individual_losses.std().item()
                    
                    # 显示各个损失组件
                    accelerator.print(f"Loss Components:")
                    accelerator.print(f"  Brier Loss: {brier_loss.item():.6f}")
                    accelerator.print(f"  Diversity Loss: {batch_diversity_loss:.6f}")
                    accelerator.print(f"  Focal Loss: {focal_loss.item():.6f}")
                    accelerator.print(f"  Total Loss: {loss_cal.item():.6f}")
                    
                    # 统计预测多样性
                    pred_min = predicted_confidence.min().item()
                    pred_max = predicted_confidence.max().item()
                    pred_std = predicted_confidence.std().item()
                    accelerator.print(f"Prediction Stats: min={pred_min:.4f}, max={pred_max:.4f}, std={pred_std:.4f}")
                    accelerator.print(f"Brier Loss Distribution: min={loss_min:.6f}, max={loss_max:.6f}, mean={brier_loss.item():.6f}, std={loss_std:.6f}")
                    accelerator.print("=" * 50)
                
                if True:
                    # 检查损失是否为NaN或Inf
                    if torch.isnan(loss_cal) or torch.isinf(loss_cal):
                        accelerator.print(f"Warning at step {total_train_steps}: Loss is NaN/Inf ({loss_cal.item()}), skipping this step")
                        continue  # 跳过这个批次
                    
                    # 限制损失范围，防止极端值
                    if loss_cal > 100:
                        accelerator.print(f"Warning at step {total_train_steps}: Loss is very large ({loss_cal.item()}), clipping to 100")
                        loss_cal = torch.clamp(loss_cal, max=100)
                    
                    # 只使用分类器损失，不使用LLM损失
                    loss = loss_cal
                    
                    # 再次检查最终损失
                    if torch.isnan(loss) or torch.isinf(loss):
                        accelerator.print(f"Warning at step {total_train_steps}: Final loss is NaN/Inf, skipping this step")
                        continue
                
                # 只对分类器进行反向传播和参数更新
                accelerator.backward(loss)
                
                # 检查梯度
                if total_train_steps <= 3:
                    total_grad_norm = 0
                    for name, param in confidence_classifier.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                            if torch.isnan(param.grad).any():
                                accelerator.print(f"Warning: NaN gradient in {name}")
                            if torch.isinf(param.grad).any():
                                accelerator.print(f"Warning: Inf gradient in {name}")
                    total_grad_norm = total_grad_norm ** (1. / 2)
                    accelerator.print(f"Step {total_train_steps} - Total gradient norm: {total_grad_norm:.4f}")
                
                # 梯度裁剪，防止梯度爆炸
                if hasattr(confidence_classifier, "module"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(confidence_classifier.module.parameters(), max_norm=1.0)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(confidence_classifier.parameters(), max_norm=1.0)
                
                if total_train_steps <= 3:
                    accelerator.print(f"Step {total_train_steps} - Gradient norm after clipping: {grad_norm:.4f}")
                
                classifier_optimizer.step()
                
                # 更新学习率调度器（每step调用）
                classifier_lr_scheduler.step()
                
                # 检查权重更新后是否有NaN，如果有则重新初始化
                nan_detected = False
                for name, param in confidence_classifier.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        accelerator.print(f"Error: NaN/Inf detected in {name} after optimizer step, reinitializing...")
                        nan_detected = True
                        
                        # 直接修改参数的data，而不是重新创建参数对象
                        with torch.no_grad():
                            if 'weight' in name:
                                param.data.normal_(mean=0.0, std=0.01)
                            elif 'bias' in name:
                                param.data.fill_(0.0)
                        
                        # 清除该参数在优化器中的状态
                        if param in classifier_optimizer.state:
                            del classifier_optimizer.state[param]
                
                if nan_detected:
                    accelerator.print("Parameters reinitialized and optimizer state cleaned due to NaN/Inf detection")
                
                classifier_optimizer.zero_grad()
                total_loss += loss.detach().float()

                if wandb_run and accelerator.is_main_process:
                    wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': loss.detach().float(),
                            'train/loss_classifier': loss_cal.detach().float()
                        })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (classifier loss: {loss.detach().float()})")

                    # torch.cuda.empty_cache()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        avg_loss = total_loss / len(train_dataloader)
        avg_ppl = float(torch.exp(torch.tensor(avg_loss)))  # ppl = exp(loss)

        # 注意：学习率调度器现在在每个step调用，不需要在epoch结束时调用
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
                        'model_dtype': model_dtype,  # 主模型的数据类型
                        'classifier_dtype': classifier_dtype,  # 分类器的数据类型
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
    eval_loss_cal = 0.0
    total_eval_steps = 0
    
    # 如果没有提供分类器，创建一个默认的分类器用于评估
    if confidence_classifier is None:
        vocab_size = len(tokenizer)
        confidence_classifier = ConfidenceClassifier(vocab_size=vocab_size)
        
        # 为了数值稳定性，分类器使用float32
        model_dtype = next(model.parameters()).dtype
        classifier_dtype = torch.float32
        confidence_classifier = confidence_classifier.to(classifier_dtype)
        
        confidence_classifier = confidence_classifier.to(model.device)
        accelerator.print("Warning: Using default classifier for evaluation")
        accelerator.print(f"Model dtype: {model_dtype}, Classifier dtype: {classifier_dtype}")
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
        confidence_classifier.eval()  # 确保分类器也在eval模式
        
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            y = batch.data.pop('y')
            
            # LLM推理：不计算梯度
            output = model(**batch)
            logits = output.logits
            num_token = logits[:,-1,:] # get the logit of the confidence token
            del logits
            
            # 将输入转换为float32以匹配分类器
            num_token_float32 = num_token.float()  # 转换为float32
            y_float32 = y.float()  # 转换标签为float32
            
            # 使用分类器预测置信度
            confidence_logits = confidence_classifier(num_token_float32)  # [batch_size, 101]
            
            # 检查logits是否有异常值
            if step == 0:  # 只在第一步打印调试信息
                print(f"Confidence logits stats: min={confidence_logits.min():.4f}, max={confidence_logits.max():.4f}, mean={confidence_logits.mean():.4f}")
                print(f"Has NaN in logits: {torch.isnan(confidence_logits).any()}")
                print(f"Has Inf in logits: {torch.isinf(confidence_logits).any()}")
            
            # 数值稳定的softmax计算
            confidence_logits = torch.clamp(confidence_logits, min=-50, max=50)  # 限制logits范围
            confidence_probs = F.softmax(confidence_logits, dim=1)  # [batch_size, 101]
            
            # 检查概率是否有异常值
            if step == 0:
                print(f"Confidence probs stats: min={confidence_probs.min():.4f}, max={confidence_probs.max():.4f}, sum={confidence_probs.sum(dim=1).mean():.4f}")
                print(f"Has NaN in probs: {torch.isnan(confidence_probs).any()}")
            
            # 创建置信度值数组 [0.00, 0.01, 0.02, ..., 1.00] - 使用float32
            confidence_values = torch.arange(0, 1.01, 0.01, dtype=torch.float32).view(1, 101).to(model.device)  # [1, 101]
            
            # 计算期望的置信度预测值
            predicted_confidence = torch.sum(confidence_probs * confidence_values, dim=1)  # [batch_size]
            
            # 检查预测置信度是否有异常值
            if step == 0:
                print(f"Predicted confidence stats: min={predicted_confidence.min():.4f}, max={predicted_confidence.max():.4f}, mean={predicted_confidence.mean():.4f}")
                print(f"Has NaN in predicted confidence: {torch.isnan(predicted_confidence).any()}")
            
            # 处理NaN值：如果有NaN，替换为0.5（中性置信度）
            if torch.isnan(predicted_confidence).any():
                print(f"Warning: Found NaN in predicted confidence, replacing with 0.5")
                predicted_confidence = torch.where(torch.isnan(predicted_confidence), 
                                                 torch.tensor(0.5, device=predicted_confidence.device, dtype=predicted_confidence.dtype), 
                                                 predicted_confidence)
            
            # 计算Brier Score损失（只针对分类器）- 使用float32
            squared_differences = (predicted_confidence - y_float32.squeeze()) ** 2
            loss_cal = torch.mean(squared_differences)
            
            # 只使用分类器损失
            loss = loss_cal
            
            if train_config.save_metrics:
                val_step_loss.append(loss.detach().float().item())
                val_step_perplexity.append(float(torch.exp(loss.detach().float())))

            print(f"classifier loss: {loss_cal:.4f}") 
               
            eval_loss += loss.detach().float()
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
                'eval/loss_classifier': eval_epoch_loss_cal,
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
        # 加载模型用于获取logits
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name if original else train_config.output_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        
        # 获取模型的数据类型
        model_dtype = next(model.parameters()).dtype
        confidence_classifier, _ = load_confidence_classifier(classifier_path, device='cuda', dtype=model_dtype)
        confidence_classifier.eval()
        print(f"Loaded confidence classifier from {classifier_path}")
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
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    
    
    # 处理每个回答以获取置信度标签
    responses_filtered, out_response_cleans, questions, confidence_stage1, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(
        test_dataset['question'], outputs, test_dataset['correct_answer'], 
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
    
    # 记录到wandb表格 - 从outputs提取文本
    responses = [output.outputs[0].text for output in outputs]
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

def test_classifier(train_config, test_dataset, tokenizer, wandb_run, original=False, classifier_path=None):
    from transformers import AutoModelForCausalLM
    
    all_y = []
    test_probs = []
    all_responses = []
    
    # 加载分类器（如果提供路径）
    confidence_classifier = None
    if classifier_path and os.path.exists(classifier_path):
        # 加载模型用于获取logits
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name if original else train_config.output_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        
        # 获取模型的数据类型
        model_dtype = next(model.parameters()).dtype
        confidence_classifier, _ = load_confidence_classifier(classifier_path, device='cuda', dtype=model_dtype)
        confidence_classifier.eval()
        print(f"Loaded confidence classifier from {classifier_path}")
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
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    
    
    # 处理每个回答以获取置信度标签
    responses_filtered, out_response_cleans, questions, confidence_stage1, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(
        test_dataset['question'], outputs, test_dataset['correct_answer'], 
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
    
    # 记录到wandb表格 - 从outputs提取文本
    responses = [output.outputs[0].text for output in outputs]
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

def load_confidence_classifier(checkpoint_path, device='cuda', dtype=None):
    """
    加载保存的置信度分类器
    
    Args:
        checkpoint_path: 分类器检查点路径
        device: 设备类型
        dtype: 数据类型，如果为None则使用保存时的类型
    
    Returns:
        confidence_classifier: 加载的分类器模型
        checkpoint_info: 检查点信息字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab_size = checkpoint['vocab_size']
    confidence_classifier = ConfidenceClassifier(vocab_size=vocab_size)
    confidence_classifier.load_state_dict(checkpoint['model_state_dict'])
    confidence_classifier.to(device)
    
    # 使用保存时的数据类型，如果没有指定的话
    if dtype is None:
        dtype = checkpoint.get('classifier_dtype', torch.float32)  # 默认为float32
    
    confidence_classifier = confidence_classifier.to(dtype)
    
    # 检查加载的权重是否有异常值并重新初始化
    for name, param in confidence_classifier.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Warning: Found NaN/Inf in parameter {name}, reinitializing...")
            with torch.no_grad():
                if 'weight' in name:
                    fan_in = param.size(1) if param.dim() > 1 else param.size(0)
                    fan_out = param.size(0)
                    std = (2.0 / (fan_in + fan_out)) ** 0.5 * 0.01
                    param.data.normal_(mean=0.0, std=std)
                elif 'bias' in name:
                    param.data.fill_(0.0)
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', -1),
        'vocab_size': vocab_size,
        'model_dtype': checkpoint.get('model_dtype', torch.float16),
        'classifier_dtype': checkpoint.get('classifier_dtype', torch.float32)
    }
    
    print(f"Loaded confidence classifier from {checkpoint_path}")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Epoch: {checkpoint_info['epoch']}")
    print(f"  - Device: {device}")
    print(f"  - Classifier dtype: {dtype}")
    print(f"  - Model dtype: {checkpoint_info['model_dtype']}")
    
    return confidence_classifier, checkpoint_info
