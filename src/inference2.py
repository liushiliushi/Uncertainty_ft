# uncertainty_sft.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
import os
import dataclasses
import fire
import random
import numpy as np
import torch.optim as optim
from peft import get_peft_model, PeftModel
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
)
from accelerate import Accelerator
from accelerate.utils import is_xpu_available
from warnings import warn

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import quantization_config as QUANTIZATION_CONFIG
from llama_recipes.data.concatenator import ConcatDataset2
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset2
from llama_recipes.utils.train_utils_uncertainty import (
    train_chat,
    test_vllm,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from warnings import warn


def setup_wandb(train_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    if "Ministral" in train_config.model_name:
        from llama_recipes.configs import wandb_config_mini as WANDB_CONFIG_MINI
        wandb_config = WANDB_CONFIG_MINI()
    elif "Qwen" in train_config.model_name:
        from llama_recipes.configs import wandb_config_qwen as WANDB_CONFIG_QWEN
        wandb_config = WANDB_CONFIG_QWEN()
    else:
        from llama_recipes.configs import wandb_config_llama as WANDB_CONFIG_Llama
        wandb_config = WANDB_CONFIG_Llama()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(dataclasses.asdict(train_config), allow_val_change=True)
    return run



def main(**kwargs):

    # 加载 config（原先有 FSDP_CONFIG，这里省略）
    train_config = TRAIN_CONFIG()
    update_config(train_config, **kwargs)
    
    # 设置随机种子列表
    seeds = [42, 123, 456, 789, 1024]
    
    accelerator = Accelerator()

    # 只有在主进程时才进行 wandb 初始化
    wandb_run = None
    if train_config.use_wandb and accelerator.is_main_process:
        wandb_run = setup_wandb(train_config, **kwargs)

    # 加载 tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name, 
        padding_side='left'
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_test = get_preprocessed_dataset2(tokenizer, 'test', train_config)

    # 存储每次运行的结果
    test_ece_scores = []
    test_auroc_scores = []
    test_acc2_scores = []
    
    if train_config.test_original_model:
        accelerator.print("==============original test================")
        for seed in seeds:
            if is_xpu_available():
                torch.xpu.manual_seed(seed)
            else:
                torch.manual_seed(seed)
            random.seed(seed)
            accelerator.print(f"Testing with seed {seed}")
            test_vllm(train_config, dataset_test, tokenizer, wandb_run, original=True)

    if accelerator.is_main_process:
        accelerator.print("==============finetuned test2stage================")
        for seed in seeds:
            if is_xpu_available():
                torch.xpu.manual_seed(seed)
            else:
                torch.manual_seed(seed)
            random.seed(seed)
            
            accelerator.print(f"Testing with seed {seed}")
            train_config.seed = seed
            test_ece, test_auroc, test_acc2 = test_vllm(train_config, dataset_test, tokenizer, wandb_run, original=False)
            test_ece_scores.append(test_ece)
            test_auroc_scores.append(test_auroc)
            test_acc2_scores.append(test_acc2)

        # 计算平均值和方差
        mean_ece = np.mean(test_ece_scores)
        mean_auroc = np.mean(test_auroc_scores)
        mean_acc2 = np.mean(test_acc2_scores)
        var_ece = np.var(test_ece_scores)
        var_auroc = np.var(test_auroc_scores)
        var_acc2 = np.var(test_acc2_scores)
        std_ece = np.std(test_ece_scores)
        std_auroc = np.std(test_auroc_scores)
        std_acc2 = np.std(test_acc2_scores)

        # 输出结果
        print(f"Mean ECE: {mean_ece}")
        print(f"Mean AUROC: {mean_auroc}")
        print(f"Mean Accuracy2: {mean_acc2}")
        print(f"Variance ECE: {var_ece}")
        print(f"Variance AUROC: {var_auroc}")
        print(f"Variance Accuracy2: {var_acc2}")
        print(f"Std Dev ECE: {std_ece}")
        print(f"Std Dev AUROC: {std_auroc}")
        print(f"Std Dev Accuracy2: {std_acc2}")

        # 记录到wandb
        if wandb_run:
            wandb_run.log({
                f'test/mean_ece_{train_config.dataset}': mean_ece,
                f'test/mean_auroc_{train_config.dataset}': mean_auroc,
                f'test/mean_acc2_{train_config.dataset}': mean_acc2,
                f'test/var_ece_{train_config.dataset}': var_ece,
                f'test/var_auroc_{train_config.dataset}': var_auroc,
                f'test/var_acc2_{train_config.dataset}': var_acc2,
                f'test/std_ece_{train_config.dataset}': std_ece,
                f'test/std_auroc_{train_config.dataset}': std_auroc,
                f'test/std_acc2_{train_config.dataset}': std_acc2,
            })


if __name__ == "__main__":
    fire.Fire(main)