
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
system_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
            Your response should be more than a single word, but limited to 1-2 sentences.
            Then please provide the final answer of yes or no. If no answer is present, please write "NONE".
            Finally, please provide your confidence (0%-100%) to your answer.

            Here are some examples:

            Question: Can plants survive without sunlight?
            Response: No. Photosynthesis depends on sunlight.
            Final answer: No.
            Confidence: 95%

            Question: Do all mammals lay eggs?
            Response: No. Only monotremes (e.g., platypus) do.
            Final answer: No.
            Confidence: 97%

            Question: Is water a good conductor of electricity?
            Response: No. Pure water lacks conductive ions.
            Final answer: No.
            Confidence: 90%"""
system_prompt_yes = """You will be asked trivia questions. Please respond to the best of your ability.
            Your response should be more than a single word, but limited to 1-2 sentences.
            Then please provide the final answer of yes or no. If no answer is present, please write "NONE".
            Finally, please judge whether your answer is correct with "yes" or "no".

            Here are some examples:

            Question: Can plants survive without sunlight?
            Response: No. Photosynthesis depends on sunlight.
            Final answer: No.
            Correct: yes

            Question: Do all mammals lay eggs?
            Response: No. Only monotremes (e.g., platypus) do.
            Final answer: No.
            Correct: yes

            Question: Is water a good conductor of electricity?
            Response: No. Pure water lacks conductive ions.
            Final answer: No.
            Correct: no"""

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def get_strategyqa_yes(tokenizer, split, vllm=True):
    path = '../dataset/StrategyQA/task.jsonl'
    dataset = datasets.load_dataset('json', data_files=path, split='train')

    def apply_prompt_template(sample):
        prompt = [{'role': 'system', 'content': system_prompt_yes},
            {"role": "user", "content":  f"Question: {sample['input']}"},
            {"role": "assistant", "content": f"Response:"},
            ]
        if vllm:
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True)
        else:
            prompt = json.dumps(prompt)
        correct_answer = "yes" if sample['target_scores']["Yes"] == 1 else "no"
        return {
            'question': json.dumps(sample['input']),
            "prompt": json.dumps(prompt),
            "correct_answer": json.dumps(correct_answer),
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset

def get_strategyqa(tokenizer, split, train_config, on_policy=False):
    if split == 'train':
        print("Error! Strategy QA is not used for training.")
    else:
        path = '../dataset/StrategyQA/task.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')

    def apply_prompt_template(sample):
        if "Ministral" in train_config.model_name:
            prompt = [
                {"role": "user", "content":  f"{system_prompt}\n\nQuestion: {sample['input']}"},
                {"role": "assistant", "content": f"Response:"},
                ]
        else:
            prompt = [{'role': 'system', 'content': system_prompt},
            {"role": "user", "content":  f"Question: {sample['input']}"},
            {"role": "assistant", "content": f"Response:"},
            ]
        correct_answer = "yes" if sample['target_scores']["Yes"] == 1 else "no"
        return {
            'question': json.dumps(sample['input']),
            "prompt": json.dumps(prompt),
            "correct_answer": json.dumps(correct_answer),
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset

