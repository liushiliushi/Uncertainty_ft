# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.datasets2 import DATASET_PREPROC, DATASET_RAW, DATASET_YES, DATASET_CONS, DATASET_REFLECTION, DATASET_CONFIDENCE_ANSWER, DATASET_IMPLICIT, get_custom_dataset
from llama_recipes.utils.config_utils import get_dataloader_kwargs


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    # def get_split():
    #     return (
    #         dataset_config.train_split
    #         if split == "train"
    #         else dataset_config.test_split
    #     )

    return DATASET_PREPROC[dataset_config.dataset](
        # dataset_config,
        tokenizer,
        'train',
    )

def get_preprocessed_dataset2(
    tokenizer, split, train_config
) -> torch.utils.data.Dataset:
    return DATASET_PREPROC[train_config.dataset](
            tokenizer, split, train_config, train_config.on_policy
        )

def get_dataset_yes(
    tokenizer, dataset, split, vllm
) -> torch.utils.data.Dataset:
    return DATASET_YES[dataset](
            tokenizer, split, vllm
        )

def get_dataset_reflection(
    tokenizer, split, train_config
) -> torch.utils.data.Dataset:
    return DATASET_REFLECTION[train_config.dataset](
            tokenizer, split, 
        )



def get_dataset_confidence_answer(
    tokenizer, split, train_config
) -> torch.utils.data.Dataset:
    return DATASET_CONFIDENCE_ANSWER[train_config.dataset](
            tokenizer, split, train_config, train_config.on_policy
        )

def get_dataset_implicit(
    tokenizer, split, train_config
) -> torch.utils.data.Dataset:
    return DATASET_IMPLICIT[train_config.dataset](
            tokenizer, split, train_config, train_config.on_policy
        )

def get_dataset_cons(
    tokenizer, dataset, split, vllm
) -> torch.utils.data.Dataset:
    return DATASET_CONS[dataset](
            tokenizer, split, vllm
        )


def get_raw_dataset(
    tokenizer, train_config, split
) -> torch.utils.data.Dataset:
    return DATASET_RAW[train_config.dataset](
            tokenizer, split, train_config
        )


def get_dataloader(tokenizer, dataset_config, train_config, split: str = "train"):
    dataset = get_preprocessed_dataset(tokenizer, dataset_config, split)
    dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)
    
    if split == "train" and train_config.batching_strategy == "packing":
        dataset = ConcatDataset(dataset, chunk_size=train_config.context_length)

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **dl_kwargs,
    )
    return dataloader
    