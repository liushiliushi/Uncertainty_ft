# uncertainty_sft.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from json import JSONDecodeError
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
from vllm import LLM, SamplingParams
from accelerate import Accelerator
from accelerate.utils import is_xpu_available
from warnings import warn
import json
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import quantization_config as QUANTIZATION_CONFIG
from llama_recipes.data.concatenator import ConcatDataset2
from llama_recipes.utils.postprocess import postprocess_extract, confidence_replace, confidence_replace_cons
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.compute_metrics import compute_conf_metrics, plot_confidence_histogram, plot_ece_diagram
from llama_recipes.utils.dataset_utils import get_dataset_cons
from llama_recipes.utils.train_utils_uncertainty import (
    train_chat,
    test_vllm,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from warnings import warn
import wandb

def setup_wandb(train_config, **kwargs):
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    return run

def test_cons(train_config, test_dataset, tokenizer, wandb_run, original=False):
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
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=10,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400,)

    
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    prompts = prompts
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace_cons(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table.add_data(response, confidence, y_item)        


    # Compute ECE and ROC-AUC score given all_y and eval_probs
    if wandb_run:
        if original == True:
            wandb_run.log({"Testing_samples/original": wan_table})
        else:
            wandb_run.log({"Testing_samples/fine-tuned": wan_table})

    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences)
    if train_config.use_wandb:
        plot_confidence_histogram(y, out_confidences, "stage1", val_metrics['acc'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, use_annotation=True)
        plot_ece_diagram(y, out_confidences, "stage1", wandb_run, original)

    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']


    if wandb_run:
        wandb_run.log({
                        'test/ece_stage1': ece_score,
                        'test/roc_auc_stage1': roc_auc_score,
                    }, commit=False)

    return ece_score, roc_auc_score

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

    # 加载 tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name, 
        padding_side='left'
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_test = get_dataset_cons(tokenizer, train_config.dataset, 'test', True)

    test_cons(train_config, dataset_test, tokenizer, wandb_run, original=True)



if __name__ == "__main__":
    fire.Fire(main)