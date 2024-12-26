
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


def extract_number(text):
    # 去掉文本中的逗号
    text = text.replace(',', '')

    # 使用正则表达式提取数字（支持小数和负号）
    match = re.search(r'[-+]?\d*\.\d+|\d+', text)

    if match:
        # 提取到的数值并返回
        return float(match.group(0))
    else:
        # 如果没有找到数字，返回 None
        return None

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


def get_gsm8k_dataset2(tokenizer, split, on_policy=False):
    if split == 'train':
        path = '../dataset/grade_school_math/data/train_response2.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:2000]')
    elif split == 'val':
        path = '../dataset/grade_school_math/data/train_response2.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[5000:5100]')
    else:
        # path = '../dataset/grade_school_math/data/test_negllm.jsonl'
        # dataset = datasets.load_dataset('json', data_files=path, split='train')
        path = '../dataset/grade_school_math/data/train_response2.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[5000:5100]')


    def apply_prompt_template(sample):
        prompt = [{'role': 'system', 'content': """You will be asked math problems. Please respond to the best of your ability.
                   Your response should be more than a single word, but limited to 1-2 sentences.
                   Then please extract a single answer from the your response. If no answer is present, please write "NONE".
                   Finally, please provide your confidence (0%-100%) to your answer.

                   Here are some examples:

                   Question: A bag contains 5 red apples and 3 green apples. How many apples are there in total?
                   Response: To find the total, add the red apples and green apples: 5 + 3 = 8 apples in total.
                   Final answer: 8
                   Confidence: 100%

                   Question: A train travels 60 miles per hour for 3 hours. How far does it travel in total?
                   Response: To calculate the total distance, multiply the speed by the time: 60 miles/hour * 3 hours = 180 miles.
                   Final answer: 180
                   Confidence: 100%

                   Question: A box contains 8 blue marbles and 6 red marbles. How many marbles are there in total?
                   Response: The total number of marbles is the sum of blue and red marbles: 8 + 7 = 15 marbles.
                   Final answer: 15
                   Confidence: 0%"""},
                  {"role": "user", "content": f"Question: {sample['question']}"},
                  {"role": "assistant", "content": f"Response: {sample['response_clean']}"}
                  ]

        matches = re.findall("Final answer: (.*)", sample['response_clean'])
        if matches:
            answer = re.findall("Final answer: (.*)", sample['response_clean'])[-1]
            y  = 1 if extract_number(answer) == float(sample['correct_answer']) else 0
            return {
                "prompt": prompt,
                "y": y,
            }
        else:
            return {
                "prompt": prompt,
                "y": 0,
            }

    def apply_prompt_template_test(sample):
        prompt = [{'role': 'system', 'content': """You will be asked math problems. Please respond to the best of your ability.
                   Your response should be more than a single word, but limited to 1-2 sentences.
                   Then please extract a single answer from the your response. If no answer is present, please write "NONE".
                   Finally, please provide your confidence (0%-100%) to your answer.

                   Here are some examples:

                   Question: A bag contains 5 red apples and 3 green apples. How many apples are there in total?
                   Response: To find the total, add the red apples and green apples: 5 + 3 = 8 apples in total.
                   Final answer: 8
                   Confidence: 100%

                   Question: A train travels 60 miles per hour for 3 hours. How far does it travel in total?
                   Response: To calculate the total distance, multiply the speed by the time: 60 miles/hour * 3 hours = 180 miles.
                   Final answer: 180
                   Confidence: 100%

                   Question: A box contains 8 blue marbles and 6 red marbles. How many marbles are there in total?
                   Response: The total number of marbles is the sum of blue and red marbles: 8 + 7 = 15 marbles.
                   Final answer: 15
                   Confidence: 0%"""},
                  {"role": "user", "content": f"Question: {sample['question']}"},
                  {"role": "assistant", "content": f"Response:"}
                  ]

        return {
            "prompt": json.dumps(prompt),
            "correct_answer": sample['correct_answer'],
        }

    if on_policy == False:
        if split == 'test':
            dataset = dataset.map(apply_prompt_template_test, remove_columns=list(dataset.features))
        else:
            dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    else:
        if split == 'val':
            dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
        else:
            dataset = dataset.map(apply_prompt_template_test, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        # prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        prompt = tokenizer.apply_chat_template(sample['prompt'], tokenize=True, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True).squeeze(0)
        prompt = torch.cat((prompt, torch.tensor([220])))
        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * (len(prompt)),
            # 'conf_index': torch.tensor([len(prompt) - 1]),
            'label': prompt,
            'y': [sample['y']]
            }

        return sample
    if on_policy == False:
        if split == 'test':
            dataset = dataset
        else:
            dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    else:
        if split == 'val':
            dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    
    return dataset



