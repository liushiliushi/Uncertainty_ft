# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets

def load_samsum(split, _):
    try:
        ds = datasets.load_dataset("Samsung/samsum", split=split)
    except ValueError as e:
        if "trust_remote_code" in str(e):
          raise ValueError("Loading Samsung/samsum requires you to execute the dataset script in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set HF_DATASETS_TRUST_REMOTE_CODE env variable to True.") from e
        else:
          raise e
    return ds

def get_preprocessed_samsum(tokenizer, split):
    dataset = datasets.load_dataset("samsum", split=split)#, trust_remote_code=dataset_config.trust_remote_code)
    #dataset = datasets.load_from_disk("/home/lyb/workspace/llama-recipes/src/data/samsum")

    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset



def get_preprocessed_samsum2(dataset_config, tokenizer, split):
    # if not hasattr(dataset_config, "trust_remote_code") or not dataset_config.trust_remote_code:
    #     raise ValueError("The repository for samsum contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/samsum. To activate `trust_remote_code` option use this config: --samsum_dataset.trust_remote_code=True")
    # # dataset = datasets.load_dataset("samsum", split=split, trust_remote_code=dataset_config.trust_remote_code)
    dataset = datasets.load_from_disk("/home/lyb/workspace/llama-recipes/src/data/samsum")

    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
