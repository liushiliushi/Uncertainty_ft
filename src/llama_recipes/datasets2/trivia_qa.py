
import copy
import datasets
import os
import json
import os
import re
import torch
from datasets import concatenate_datasets
from functools import partial
import string

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
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()



def get_trivia_qa(tokenizer, split, generate="vanilla"):
    if split == 'train':
        path = '/home/lyb/workspace/pragmatic_calibration/data/trivia_qa/tqa_train3.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:7500]')
    elif split == 'val':
        path = '/home/lyb/workspace/pragmatic_calibration/data/trivia_qa/tqa_train3.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[7500:]')
    else:
        path = '/home/lyb/workspace/pragmatic_calibration/data/trivia_qa/tqa_val.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')

    def apply_prompt_template(sample):
        sample['prompt'][2] = {'role': 'assistant', 'content': f"Answer: {sample['response_clean']}"}
        sample['prompt'].append({'role': 'user', 'content': f'What\'s your confidence (0%-100%) in your answer?\n Use the following format:\n Confidence: <Only the confidence, no other words!>'})
        sample['prompt'].append({'role': 'assistant', 'content': f'Confidence: '})
        y  = 1 if normalize_answer(sample['answer']).lower().strip() in eval(sample['correct_answer']) else 0
        return {
            "prompt": sample['prompt'],
            "y": y,
            # 'answer': sample['answer'],
            # 'correct_answer': sample['correct_answer']
        }
    def apply_prompt_template_test(sample):
        sample['prompt'][0] = {'role': 'system', 'content': "You will be asked questions. Please answer the question and provide your confidence (0%-100%).\n Use the following format:\n Answer: <Only the answer>\n Confidence: <Only the confidence, no other words!>\n"}
        sample['prompt'][2] = {'role': 'assistant', 'content': f"Answer:"}
        return {
            "prompt": json.dumps(sample['prompt']),
            "correct_answer": sample['correct_answer'],
            # 'answer': sample['answer'],
            # 'correct_answer': sample['correct_answer']
        }

    if split == 'test':
        dataset = dataset.map(apply_prompt_template_test, remove_columns=list(dataset.features))
    else:
        dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        # prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        prompt = tokenizer.apply_chat_template(sample['prompt'], tokenize=True, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True).squeeze(0)
        prompt = torch.cat((prompt, torch.tensor([220])))
        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * (len(prompt)),
            # 'conf_index': torch.tensor([len(prompt) - 1]),
            'y': [sample['y']]
            }

        return sample
    def tokenize_add_label_test(sample):
        # prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        prompt = tokenizer.apply_chat_template(sample['prompt'], tokenize=True, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True).squeeze(0)
        prompt = torch.cat((prompt, torch.tensor([220])))
        correct_answer = json.loads(sample['correct_answer'])
        correct_answer = json.dumps(correct_answer)
        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * (len(prompt)),
            # 'conf_index': torch.tensor([len(prompt) - 1]),
            # 'correct_answer': sample['correct_answer']
            }

        return sample
    if split == 'test':
        # dataset = dataset.map(tokenize_add_label_test, remove_columns=list(dataset.features))
        dataset = dataset
    else:
        dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset



