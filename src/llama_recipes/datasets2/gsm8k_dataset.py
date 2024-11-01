
import copy
import datasets
import os
import json
import os
import re
import torch
from datasets import concatenate_datasets

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def get_gsm8k_dataset(tokenizer, split, generate="vanilla"):
    if split == 'train':
        path = '../dataset/grade_school_math/data/train.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    else:
        path = '../dataset/grade_school_math/data/test.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')


    def apply_prompt_template(sample):
        return {
            "prompt": f"Question: {sample['question']}\n Answer: {extract_answer(sample['answer'])} \n Provide the probability that the answer for the question is correct (0% to 100%). The response should follow the format:\nP: <The probability that the answer is correct>.\nP: ",
            "y": 1,
        }
    def apply_prompt_template_neg(sample):
        return {
            "prompt": f"Question: {sample['question']}\n Answer: {int(extract_answer(sample['answer'])) - 1} \n Provide the probability that the answer for the question is correct (0% to 100%). The response should follow the format:\nP: <The probability that the answer is correct>.\nP: ",
            "y": 0,
        }

    dataset_pos = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset_neg = dataset.map(apply_prompt_template_neg, remove_columns=list(dataset.features))
    dataset = concatenate_datasets([dataset_pos, dataset_neg])

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)

        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * (len(prompt)),
            'y': [sample['y']]
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset


