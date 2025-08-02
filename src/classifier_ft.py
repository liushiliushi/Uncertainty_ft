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
    AutoModelForCausalLM
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
from llama_recipes.utils.train_utils_uncertainty_classifier import (
    clear_gpu_cache,
    print_model_size,
    ConfidenceClassifier
)
from warnings import warn
import sys

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
    run.config.update(train_config, allow_val_change=True)
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
    if train_config.use_wandb and accelerator.is_main_process:
        wandb_run = setup_wandb(train_config, **kwargs)

    # 设置量化配置
    bnb_config = None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
                FutureWarning)
            train_config.quantization = "8bit"
        # 4bit / 8bit
        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name,
        padding_side='left'
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 准备数据集
    dataset_train = get_preprocessed_dataset2(tokenizer, 'train', train_config).shuffle(seed=42)
    accelerator.print(f"--> Training Set Length = {len(dataset_train)}")

    # 若使用 packing strategy
    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset2(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    if train_config.on_policy:
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=0,
            pin_memory=True,
            batch_sampler=train_dl_kwargs['batch_sampler'],
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=0,
            pin_memory=True,
            **train_dl_kwargs,
        )

    dataset_test = get_preprocessed_dataset2(tokenizer, 'test', train_config)
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        num_workers=0,
        batch_size=train_config.batch_size_testing
    )

    if train_config.run_validation:
        if train_config.train_gpt:
            eval_dataloaders_dict = {}
            eval_datasets = ['hotpot_qa', 'trivia_qa', 'gsm8k_dataset', 'truthful_qa', 'strategy_qa']
            original_dataset = train_config.dataset
            for dataset_name in eval_datasets:
                train_config.dataset = dataset_name     
                dataset_val = get_preprocessed_dataset2(tokenizer, 'val', train_config)
                if train_config.batching_strategy == "packing":
                    dataset_val = ConcatDataset2(dataset_val, chunk_size=train_config.context_length)
                val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
                eval_dataloader = torch.utils.data.DataLoader(
                    dataset_val,
                    num_workers=train_config.num_workers_dataloader,
                    pin_memory=True,
                    **val_dl_kwargs,
                )
                accelerator.print(f"--> Validation Set Length = {len(dataset_val)}")

                if len(eval_dataloader) == 0:
                    raise ValueError("Validation set size too small to form a single batch.")
                eval_dataloaders_dict[dataset_name] = eval_dataloader   
            train_config.dataset = original_dataset
        else:
            dataset_val = get_preprocessed_dataset2(tokenizer, 'val', train_config)
            if train_config.batching_strategy == "packing":
                dataset_val = ConcatDataset2(dataset_val, chunk_size=train_config.context_length)
            val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
            eval_dataloader = torch.utils.data.DataLoader(
                dataset_val,
                num_workers=train_config.num_workers_dataloader,
                pin_memory=True,
                **val_dl_kwargs,
            )
            accelerator.print(f"--> Validation Set Length = {len(dataset_val)}")
            if len(eval_dataloader) == 0:
                raise ValueError("Validation set size too small to form a single batch.")

    # 加载/初始化模型
    use_cache = None  # accelerate 下不需要专门设置
    # 加载/初始化模型
    if "Llama-3.1" in train_config.model_name:
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map=None,  # accelerate 内部会处理
            torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name, 
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map=None
        )

    # embedding resize
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        accelerator.print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    # 看一下模型大小
    if accelerator.is_main_process:
        print_model_size(model, train_config, rank=0)

    # PEFT
    if train_config.use_peft:
        if train_config.from_peft_checkpoint:
            print(f"Loading peft from {train_config.from_peft_checkpoint}")
            model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
            peft_config = model.peft_config
        else:
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
        if accelerator.is_main_process:
            if wandb_run:
                wandb_run.config.update(peft_config, allow_val_change=True)
            model.print_trainable_parameters()

    if is_xpu_available():
        model.to("xpu:0")
    elif torch.cuda.is_available():
        model.to("cuda")

    # Get the ids of the numbers from 0 to 100
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    print("----------------------------")
    print(token_ids)
    num_indices = torch.tensor(token_ids).to(model.device)
    confidence_classifier = ConfidenceClassifier(model.lm_head, num_indices).to(model.device)
    optimizer = optim.AdamW(confidence_classifier.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    
    # Start the training loop
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, 
    )
    if True:
            eval_dataloaders_dict = {}
            eval_datasets = ['trivia_qa', 'gsm8k_dataset']
            original_dataset = train_config.dataset
            for dataset_name in eval_datasets:
                train_config.dataset = dataset_name     
                dataset_val = get_preprocessed_dataset2(tokenizer, 'val', train_config)
                if train_config.batching_strategy == "packing":
                    dataset_val = ConcatDataset2(dataset_val, chunk_size=train_config.context_length)
                val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
                eval_dataloader = torch.utils.data.DataLoader(
                    dataset_val,
                    num_workers=train_config.num_workers_dataloader,
                    pin_memory=True,
                    **val_dl_kwargs,
                )
                accelerator.print(f"--> Validation Set Length = {len(dataset_val)}")

                if len(eval_dataloader) == 0:
                    raise ValueError("Validation set size too small to form a single batch.")
                eval_dataloaders_dict[dataset_name] = eval_dataloader   
            train_config.dataset = original_dataset
    if train_config.run_validation:
        for dataset_name in eval_dataloaders_dict:
            eval_dataloaders_dict[dataset_name] = accelerator.prepare(eval_dataloaders_dict[dataset_name])
    else:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    clear_gpu_cache()  # 清理一次缓存

    # 开始训练
    if train_config.train_coarse:
        from llama_recipes.utils.train_utils_uncertainty_coarse import (
            train_chat,
            train_gpt
        )
        if train_config.train_gpt:
            results = train_gpt(
                model,
                train_dataloader,
                eval_dataloaders_dict,  # Pass dictionary of dataloaders
                test_dataloader,
                tokenizer,
                optimizer,
                scheduler,
                train_config.gradient_accumulation_steps,
                train_config,
                accelerator,         # 传入 accelerator
                wandb_run,
            )
        else:
            results = train_chat(
                model,
                train_dataloader,
                eval_dataloader,
                dataset_test,
                tokenizer,
                optimizer,
                scheduler,
                train_config.gradient_accumulation_steps,
                train_config,
                accelerator,         # 传入 accelerator
                wandb_run,
            )
    else:
        from llama_recipes.utils.train_utils_uncertainty_classifier import (
            train_chat,
        )
        if train_config.train_gpt:
            results = train_gpt(
            model,
            train_dataloader,
            eval_dataloaders_dict,  # Pass dictionary of dataloaders
            test_dataloader,
            tokenizer,
            optimizer,
            scheduler,
            train_config.gradient_accumulation_steps,
            train_config,
            accelerator,         # 传入 accelerator
            wandb_run,
        )
        else:
            results = train_chat(
                model,
                confidence_classifier,
                train_dataloader,
                eval_dataloaders_dict,  # Pass dictionary of dataloaders
                dataset_test,
                tokenizer,
                optimizer,
                scheduler,
                train_config.gradient_accumulation_steps,
                train_config,
                accelerator,         # 传入 accelerator
                wandb_run,
            )
    print("training ended")
    sys.exit(0)


if __name__ == "__main__":
    fire.Fire(main)
    
    sys.exit(0)
