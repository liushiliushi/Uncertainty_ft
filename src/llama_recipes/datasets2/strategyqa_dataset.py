
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

def get_strategyqa(tokenizer, split):
    if split == 'train':
        path = '../dataset/StrategyQA/train.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    else:
        path = '../dataset/StrategyQA/test.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    character = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)", "(M)", "(N)",
                 "(O)", "(P)", "(Q)", "(R)", "(S)", "(T)", "(U)", "(V)", "(W)", "(X)", "(Y)", "(Z)"]

    def apply_prompt_template(qa):

        question = qa['input'] + '\n' + 'Options: '
        j = 0
        for key, value in qa['target_scores'].items():
            option = character[j] + " " + key + "\t"
            question += option
            if value == 1:
                answer_index = j
            j += 1

        pos_answer = character[answer_index]
        return {
            "prompt": f"Provide the probability that the answer to the question is correct (0% to 100%). Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nP:<ONLY the probability that the answer is correct, without any extra commentary whatsoever; just the probability!>\n\nQuestion: {question}\nAnswer: {pos_answer}\nP: ",
            "y": 1,
        }
    def apply_prompt_template_neg(qa):
        question = qa['input'] + '\n' + 'Options: '
        j = 0
        for key, value in qa['target_scores'].items():
            option = character[j] + " " + key + "\t"
            question += option
            if value == 1:
                answer_index = j
            j += 1
        neg_answer = character[1-answer_index]
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

