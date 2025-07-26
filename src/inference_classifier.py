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
    AutoModelForCausalLM,
)
import torch.nn.functional as F
import json
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
    train_chat,
    test_vllm,
    test_classifier,
    clear_gpu_cache,
    print_model_size,
    get_policies,
    ConfidenceClassifier,
    load_confidence_classifier,
)
from llama_recipes.utils.postprocess import confidence_replace
from llama_recipes.utils.compute_metrics import compute_conf_metrics, plot_confidence_histogram, plot_ece_diagram
from tqdm import tqdm
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
    
    # 如果没有指定output_dir，设置默认值
    if not hasattr(train_config, 'output_dir') or train_config.output_dir is None:
        train_config.output_dir = "checkpoints/classifier"
        print(f"No output_dir specified, using default: {train_config.output_dir}")
    
    # 检查模型路径是否存在
    if not os.path.exists(train_config.output_dir):
        print(f"Warning: Model directory {train_config.output_dir} does not exist!")
        print("This might cause issues with finetuned model testing.")
        # 创建目录以避免后续错误
        os.makedirs(train_config.output_dir, exist_ok=True)
    
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

    dataset_test = get_preprocessed_dataset2(tokenizer, 'test', train_config)

    # 设置分类器路径
    classifier_path = kwargs.get('classifier_path', os.path.join(train_config.output_dir, "confidence_classifier.pt"))
    
    # 如果没有指定test_original_model，且没有找到微调模型，则默认测试原始模型
    if not hasattr(train_config, 'test_original_model'):
        train_config.test_original_model = False
    
    if train_config.test_original_model:
        accelerator.print("==============original test================")
        test_classifier(train_config, dataset_test, tokenizer, wandb_run, original=True)

    # 检查微调模型是否存在
    model_files_exist = (
        os.path.exists(os.path.join(train_config.output_dir, "adapter_config.json")) or
        os.path.exists(os.path.join(train_config.output_dir, "config.json")) or
        any(os.path.exists(os.path.join(train_config.output_dir, f)) for f in 
            ["pytorch_model.bin", "model.safetensors", "adapter_model.bin", "adapter_model.safetensors"])
    )
    
    test_ece, test_auroc, test_acc2 = None, None, None  # 初始化变量
    
    if model_files_exist:
        print("==============finetuned test (traditional method)================")
        test_ece, test_auroc, test_acc2 = test_classifier(train_config, dataset_test, tokenizer, wandb_run, original=False)
    else:
        print(f"Finetuned model not found in {train_config.output_dir}. Skipping finetuned model test.")
        print("Expected files: adapter_config.json, config.json, pytorch_model.bin, model.safetensors, etc.")
    
    # 使用分类器进行测试
    if os.path.exists(classifier_path):
        print("==============finetuned test (with classifier)================")
        classifier_ece, classifier_auroc, classifier_acc2 = test_classifier(
            train_config, dataset_test, tokenizer, wandb_run, 
            original=False, classifier_path=classifier_path
        )
        
        if classifier_ece is not None:
            print(f"\n=== Performance Results ===")
            print(f"Method                | ECE      | ROC-AUC  | Accuracy | Score")
            print(f"---------------------|----------|----------|----------|----------")
            
            if test_ece is not None:
                print(f"Traditional          | {test_ece:.4f}   | {test_auroc:.4f}   | {test_acc2:.4f}   | {3*test_acc2 + 2*test_auroc - test_ece:.4f}")
                print(f"Classifier           | {classifier_ece:.4f}   | {classifier_auroc:.4f}   | {classifier_acc2:.4f}   | {3*classifier_acc2 + 2*classifier_auroc - classifier_ece:.4f}")
                
                # 计算改进
                ece_improvement = test_ece - classifier_ece
                auroc_improvement = classifier_auroc - test_auroc
                acc_improvement = classifier_acc2 - test_acc2
                
                print(f"\n=== Improvements ===")
                print(f"ECE improvement: {ece_improvement:+.4f} (lower is better)")
                print(f"ROC-AUC improvement: {auroc_improvement:+.4f} (higher is better)")
                print(f"Accuracy improvement: {acc_improvement:+.4f} (higher is better)")
            else:
                print(f"Classifier (only)    | {classifier_ece:.4f}   | {classifier_auroc:.4f}   | {classifier_acc2:.4f}   | {3*classifier_acc2 + 2*classifier_auroc - classifier_ece:.4f}")
                print(f"\nNote: Traditional method was skipped (no finetuned model found)")
        
    else:
        print(f"Classifier not found at {classifier_path}. Make sure you have trained the classifier first.")
        print("You can specify a custom classifier path using --classifier_path argument")
        print(f"\nTo run this script properly, you should specify:")
        print(f"  --output_dir: Path to the directory containing your trained model")
        print(f"  --classifier_path: Path to the trained classifier (optional, defaults to output_dir/confidence_classifier.pt)")
        print(f"\nExample:")
        print(f"  python inference_classifier.py --output_dir checkpoints/classifier --dataset trivia_qa ...")


if __name__ == "__main__":
    fire.Fire(main)