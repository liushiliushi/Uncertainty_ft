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
import string
from vllm import LLM, SamplingParams
from accelerate import Accelerator
from accelerate.utils import is_xpu_available
from llama_recipes.utils.gpt_answer_scoring import GPTAnswerScoring
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
    run.config.update(dataclasses.asdict(train_config), allow_val_change=True)
    return run

def test_cons(train_config, test_dataset, tokenizer, wandb_run, original=False):
    """
    Evaluates the model using two-stage generation:
    1. Generate one response with temperature=0
    2. Generate 10 responses with temperature=0.6
    Confidence is calculated as the proportion of high-temperature responses matching the deterministic response
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
    
    # First stage: Generate one response with temperature=0
    sampling_params_deterministic = SamplingParams(
        n=1,
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        max_tokens=400,
    )

    # Second stage: Generate 10 responses with temperature=0.6
    sampling_params_diverse = SamplingParams(
        n=10,
        temperature=0.2,
        top_k=-1,
        top_p=1.0,
        max_tokens=400,
    )
    
    wan_table = wandb.Table(columns=['response', 'confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]

    # Generate deterministic responses
    outputs_deterministic = llm.generate(prompts=prompts, sampling_params=sampling_params_deterministic)
    # Generate diverse responses
    outputs_diverse = llm.generate(prompts=prompts, sampling_params=sampling_params_diverse)

    # Process deterministic responses to get base answers and correctness
    responses_det, out_response_cleans_det, questions, _, y_det, y_None_det, _, correct_answer_cleans = confidence_replace_cons(
        test_dataset['question'], 
        outputs_deterministic, 
        test_dataset['correct_answer'], 
        dataset_name=train_config.dataset,
        vllm=True
    )

    # Process diverse responses
    responses_div, out_response_cleans_div, _, _, _, _, _, _ = confidence_replace_cons(
        test_dataset['question'], 
        outputs_diverse, 
        test_dataset['correct_answer'], 
        dataset_name=train_config.dataset,
        vllm=True
    )
    # print("*************")
    # print(len(responses_det))
    # print(len(responses_div))
    # Calculate confidences based on agreement with deterministic response
    if train_config.dataset == "hotpot_qa" or train_config.dataset == "truthful_qa":
        answer_scorer = GPTAnswerScoring()
    confidences = []
    for i in range(len(responses_det)):
        det_response = responses_det[i]
        div_responses = responses_div[i*10:(i+1)*10]
        if det_response == None:
            continue
        # print("===============")
        # print(det_response)
        # print(div_responses)
        matching_count = 0
        # Compare answers based on dataset type
        if train_config.dataset == 'gsm8k_dataset':
            for div_resp in div_responses:
                if det_response == div_resp:
                    matching_count += 1
        
        elif train_config.dataset in ["trivia_qa", "strategy_qa"]:
            for div_resp in div_responses:
                if det_response == div_resp:
                    matching_count += 1
        
        elif train_config.dataset == "hotpot_qa":
            for div_resp in div_responses:
                if answer_scorer.score_same(questions[i], div_resp, det_response):
                    matching_count += 1
        
        elif train_config.dataset == "truthful_qa":
            for div_resp in div_responses:
                if answer_scorer.score_same(questions[i], div_resp, det_response):
                    matching_count += 1
        
        else:  # Default case: direct string comparison
            for div_resp in div_responses:
                if div_resp.strip() == det_response.strip():
                    matching_count += 1
        
        confidence = matching_count / 10.0
        print(confidence)
        confidences.append(confidence)

    # Add results to wandb table
    for response, confidence, y_item in zip(responses_det, confidences, y_det):
        wan_table.add_data(response, confidence, y_item)

    # Log to wandb
    if wandb_run:
        if original:
            wandb_run.log({"Testing_samples/original": wan_table})
        else:
            wandb_run.log({"Testing_samples/fine-tuned": wan_table})

    number = len(y_det)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y_det, confidences)
    
    if train_config.use_wandb:
        plot_confidence_histogram(y_det, confidences, "stage1", val_metrics['acc'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, use_annotation=True)
        plot_ece_diagram(y_det, confidences, "stage1", wandb_run, original)

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