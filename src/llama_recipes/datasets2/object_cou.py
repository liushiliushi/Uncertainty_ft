
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

def get_object_cou(tokenizer, split, generate="vanilla"):
    if split == 'train':
        if generate == 'llm':
            path = '../dataset/ObjectCou/train_neg.jsonl'
        else:
            path = '../dataset/ObjectCou/train.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    else:
        if generate == 'llm':
            path = '../dataset/ObjectCou/test_neg.jsonl'
        else:
            path = '../dataset/ObjectCou/test.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    def apply_prompt_template(qa):

        question = qa['input']
        pos_answer =qa['target'][1]
        return {
            "prompt": f"Provide the probability that the answer to the question is correct (0% to 100%). Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nP:<ONLY the probability that the answer is correct, without any extra commentary whatsoever; just the probability!>\n\nQuestion: {question}\nAnswer: {pos_answer}\nP: ",
            "y": 1,
        }
    def apply_prompt_template_neg(qa):
        question = qa['input']
        pos_answer = qa['target'][1]
        if generate == "vanilla":
            neg_answer = str(int(pos_answer) - 1)
        else:
            neg_answer = qa['neg_llm']
        return {
            "prompt": f"Provide the probability that the answer to the question is correct (0% to 100%). Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nP:<ONLY the probability that the answer is correct, without any extra commentary whatsoever; just the probability!>\n\nQuestion: {question}\nAnswer: {neg_answer}\nP: ",
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

def get_object_cou2(tokenizer, split, generate="vanilla"):
    if split == 'train':
        path = '../dataset/ObjectCou/train_neg2.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    else:
        path = '../dataset/ObjectCou/test_neg2.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    def apply_prompt_template(sample):
        if sample['label'] == 1:
            answer = sample['real_answer']
        else:
            answer = sample['neg_llm2']
        question = sample['question']
        return {
            "prompt": f"Provide the probability that the answer to the question is correct (0% to 100%). Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nP:<ONLY the probability that the answer is correct, without any extra commentary whatsoever; just the probability!>\n\nQuestion: {question}\nAnswer: {answer}\nP: ",
            "y": sample['label'],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

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

def get_object_cou3(tokenizer, split, generate="vanilla"):
    if split == 'train':
        path = '../dataset/ObjectCou/train_neg2.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    else:
        path = '../dataset/ObjectCou/test_neg2.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    def apply_prompt_template(sample):
        if sample['label'] == 1:
            answer = sample['real_answer']
        else:
            answer = sample['neg_llm2']
        question = sample['question']
        return {
            "prompt": f"Provide the probability that the answer to the question is correct (0% to 100%). Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nP:<ONLY the probability that the answer is correct, without any extra commentary whatsoever; just the probability!>\n\nQuestion: {question}\nAnswer: {answer}\nP: ",
            "y": sample['label'],
        }

    def apply_prompt_template2(sample):
        if sample['label'] == 0:
            answer = sample['real_answer']
        else:
            answer = sample['neg_llm2']
        question = sample['question']
        return {
            "prompt": f"Provide the probability that the answer to the question is correct (0% to 100%). Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nP:<ONLY the probability that the answer is correct, without any extra commentary whatsoever; just the probability!>\n\nQuestion: {question}\nAnswer: {answer}\nP: ",
            "y": (1-sample['label']),
        }

    dataset1 = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset2 = dataset.map(apply_prompt_template2, remove_columns=list(dataset.features))
    dataset = concatenate_datasets([dataset1, dataset2])

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

