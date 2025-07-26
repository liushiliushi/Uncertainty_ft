# uncertainty_sft.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
import os
import dataclasses
import fire
import random
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
from llama_recipes.utils.dataset_utils import get_dataset_confidence_answer
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
    
    # 设置随机种子
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    else:
        torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    # 若有可用 GPU，设置一下可见设备（如果你使用 accelerate launcher，通常无需手动设置）
    # os.environ["CUDA_VISIBLE_DEVICES"] = train_config.cuda

    accelerator = Accelerator()

    # 只有在主进程时才进行 wandb 初始化
    wandb_run = None
    if train_config.use_wandb and True:
        wandb_run = setup_wandb(train_config, **kwargs)

    # 加载 tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name, 
        padding_side='left'
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_test = get_dataset_confidence_answer(tokenizer, 'test', train_config)

    if train_config.test_original_model:
        accelerator.print("==============original test================")
        test_vllm(train_config, dataset_test, tokenizer, wandb_run, original=True)

    if True:
        print("==============finetuned test2stage================")
        test_ece, test_auroc, test_acc2 = test_vllm(train_config, dataset_test, tokenizer, wandb_run, original=False)


if __name__ == "__main__":
    fire.Fire(main)