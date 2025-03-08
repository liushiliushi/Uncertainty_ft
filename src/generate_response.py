# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import torch
from collections import Counter
import os
import json
import dataclasses
import fire
import random

import torch.optim as optim
from peft import get_peft_model, PeftModel, AutoPeftModelForCausalLM
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)
from tqdm import tqdm
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from llama_recipes.utils.train_utils_uncertainty import (
    setup,
    setup_environ_flags,
    clear_gpu_cache,
)
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import quantization_config as QUANTIZATION_CONFIG
from llama_recipes.data.concatenator import ConcatDataset2
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
    check_fsdp_config,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset, get_raw_dataset, get_preprocessed_dataset2
from llama_recipes.utils.postprocess import postprocess_extract, confidence_replace

from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from accelerate.utils import is_xpu_available
from warnings import warn
from vllm import LLM, SamplingParams



def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # setting quantization configs
    bnb_config = None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
                FutureWarning)
            train_config.quantization = "8bit"

        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError("8bit quantization is not supported with FSDP, please use 4bit quantization")

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None

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

    # model = AutoPeftModelForCausalLM.from_pretrained("checkpoints0921")

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name, padding_side='left')
    # TODO
    tokenizer.pad_token_id = tokenizer.eos_token_id




    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16 and not train_config.quantization:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        # Load the pre-trained peft model checkpoint and setup its configuration
        if train_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
            peft_config = model.peft_config
        # Generate the peft config and start fine-tuning from original model
        else:
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    hsdp_device_mesh_plan = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh_plan = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size,
                                                 sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    dataset_config = None

    dataset = get_raw_dataset(
        tokenizer,
        train_config.dataset,
        kwargs['split'],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers= 0,
        batch_size=train_config.batch_size_testing
    )

    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])


    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                     max_tokens=2048)
    prompts = dataset['prompt']
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    _, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(dataset['question'], outputs, dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    print(y)
    with open(train_config.output_dir, "w") as f1:
        for query_ids in range(len(questions)):
            if train_config.dataset == "hotpot_qa":
                item= {"question": questions[query_ids],
                                            "response_clean": out_response_cleans[query_ids], 
                                            "correct_answer": correct_answer_cleans[query_ids],
                                            "y": y[query_ids]}
            else:
                item= {"question": questions[query_ids],
                                            "response_clean": out_response_cleans[query_ids], 
                                            "correct_answer": correct_answer_cleans[query_ids]}
            json_line = json.dumps(item)  
            f1.write(json_line + "\n")   
                # f1.flush()




if __name__ == "__main__":
    fire.Fire(main)
