
import copy
import datasets
import os
import json
import os
import re
import torch
from datasets import concatenate_datasets
from functools import partial


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

def get_professional_law(tokenizer, split, generate="vanilla"):
    if split == 'train':
        if generate == 'llm':
            path = '../dataset/data/test/professional_law_test_neg.csv'
        else:
            path = '../dataset/data/test/professional_law_test.csv'
        dataset = datasets.load_dataset('csv', data_files=path, split='train')
    else:
        if generate == 'llm':
            path = '../dataset/data/val/professional_law_val_neg.csv'
        else:
            path = '../dataset/data/val/professional_law_val.csv'
        dataset = datasets.load_dataset('csv', data_files=path, split='train')
    character = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)", "(M)", "(N)",
                 "(O)", "(P)", "(Q)", "(R)", "(S)", "(T)", "(U)", "(V)", "(W)", "(X)", "(Y)", "(Z)"]

    def apply_prompt_template(sample):
        raw_question = sample['0']
        pos_answer = character[ord(sample['5']) - ord("A")]
        options = [sample[str(i)] for i in range(1, 5)]
        question = raw_question + '\n' + 'Options: '
        for i in range(4):
            question += character[i] + " " + options[i] + "  "
        prompt = f"Provide the probability that the answer to the question is correct (0% to 100%). Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nP:<ONLY the probability that the answer is correct, without any extra commentary whatsoever; just the probability!>\n\nQuestion: {question}\nAnswer: {pos_answer}\nP: "
        return {
            "prompt": prompt,
            "y": 1,
        }
    def apply_prompt_template_neg(sample):
        raw_question = sample['0']
        if generate == "vanilla":
            neg_answer = character[((ord(sample['5']) - ord("A")) - 1) % 4]
        else:
            neg_answer = character[((ord(sample['6']) - ord("A"))) % 4]
        options = [sample[str(i)] for i in range(1, 5)]
        question = raw_question + '\n' + 'Options: '
        for i in range(4):
            question += character[i] + " " + options[i] + "  "
        prompt = f"Provide the probability that the answer to the question is correct (0% to 100%). Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nP:<ONLY the probability that the answer is correct, without any extra commentary whatsoever; just the probability!>\n\nQuestion: {question}\nAnswer: {neg_answer}\nP: "
        return {
            "prompt": prompt,
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
            # 'conf_index': torch.tensor([len(prompt) - 1]),
            'y': [sample['y']]
            }

        return sample
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

# add test split (require both answer and confidence for the test dataset)
# def get_professional_law(tokenizer, split, generate="vanilla"):
#     if split == 'train':
#         if generate == 'llm':
#             path = '../dataset/data/test/professional_law_test_neg.csv'
#         else:
#             path = '../dataset/data/test/professional_law_test.csv'
#         dataset = datasets.load_dataset('csv', data_files=path, split='train[:4]')
#     elif split == "val":
#         if generate == 'llm':
#             path = '../dataset/data/test/professional_law_test_neg.csv'
#         else:
#             path = '../dataset/data/test/professional_law_test.csv'
#         dataset = datasets.load_dataset('csv', data_files=path, split='train[-4:]')
#     else:
#         if generate == 'llm':
#             path = '../dataset/data/val/professional_law_val_neg.csv'
#         else:
#             path = '../dataset/data/val/professional_law_val.csv'
#         dataset = datasets.load_dataset('csv', data_files=path, split='train')
#     character = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)", "(M)", "(N)",
#                  "(O)", "(P)", "(Q)", "(R)", "(S)", "(T)", "(U)", "(V)", "(W)", "(X)", "(Y)", "(Z)"]

#     def apply_prompt_template(sample, split):
#         raw_question = sample['0']
#         pos_answer = character[ord(sample['5']) - ord("A")]
#         options = [sample[str(i)] for i in range(1, 5)]
#         question = raw_question + '\n' + 'Options: '
#         for i in range(4):
#             question += character[i] + " " + options[i] + "  "
#         if split == "test":
#             prompt = f"Provide your answer and the probability that the answer is correct (0% to 100%) for the following question. Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nG: <ONLY the numerical answer of first most likely guess; not a complete sentence, just the guess!>\nP: <ONLY the probability that G is correct, without any extra commentary whatsoever; just the probability!>\nQuestion:{question}\n"            
#             return {
#                 "prompt": prompt,
#                 "answer": ord(sample['5']) - ord("A"),
#             }
#         else:
#             prompt = f"Provide the probability that the answer to the question is correct (0% to 100%). Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nP:<ONLY the probability that the answer is correct, without any extra commentary whatsoever; just the probability!>\n\nQuestion: {question}\nAnswer: {pos_answer}\nP: "
#             return {
#                 "prompt": prompt,
#                 "y": 1,
#             }
#     def apply_prompt_template_neg(sample, split):
#         raw_question = sample['0']
#         if generate == "vanilla":
#             neg_answer = character[((ord(sample['5']) - ord("A")) - 1) % 4]
#         else:
#             neg_answer = character[((ord(sample['6']) - ord("A"))) % 4]
#         options = [sample[str(i)] for i in range(1, 5)]
#         question = raw_question + '\n' + 'Options: '
#         for i in range(4):
#             question += character[i] + " " + options[i] + "  "
#         if split == "test":
#             prompt = f"Provide your answer and the probability that the answer is correct (0% to 100%) for the following question. Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nG: <ONLY the numerical answer of first most likely guess; not a complete sentence, just the guess!>\nP: <ONLY the probability that G is correct, without any extra commentary whatsoever; just the probability!>\nQuestion:{question}\n"            
#             return {
#                 "prompt": prompt,
#                 "answer": ord(sample['5']) - ord("A"),
#             }
#         else:
#             prompt = f"Provide the probability that the answer to the question is correct (0% to 100%). Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nP:<ONLY the probability that the answer is correct, without any extra commentary whatsoever; just the probability!>\n\nQuestion: {question}\nAnswer: {neg_answer}\nP: "
#             return {
#                 "prompt": prompt,
#                 "y": 0,
#             }

#     apply_prompt_template_with_split = partial(apply_prompt_template, split=split)
#     apply_prompt_template_neg_with_split = partial(apply_prompt_template_neg, split=split)
#     dataset_pos = dataset.map(apply_prompt_template_with_split, remove_columns=list(dataset.features))
#     dataset_neg = dataset.map(apply_prompt_template_neg_with_split, remove_columns=list(dataset.features))
#     dataset = concatenate_datasets([dataset_pos, dataset_neg])

#     def tokenize_add_label(sample, split):
#         if split == "test":
#             prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], padding=True, add_special_tokens=False)
#             sample = {
#                 "input_ids": prompt,
#                 "attention_mask" : [1] * (len(prompt)),
#                 # 'conf_index': torch.tensor([len(prompt) - 1]),
#                 'answer': [sample['answer']]
#                 }
#         else:
#             prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
#             sample = {
#                 "input_ids": prompt,
#                 "attention_mask" : [1] * (len(prompt)),
#                 # 'conf_index': torch.tensor([len(prompt) - 1]),
#                 'y': [sample['y']]
#                 }

#         return sample
#     tokenize_add_label_with_split = partial(tokenize_add_label, split=split)
#     dataset = dataset.map( tokenize_add_label_with_split, remove_columns=list(dataset.features))

#     return dataset


def get_professional_law2(tokenizer, split, generate="vanilla"):
    if split == 'train':
        path = '../dataset/data/test/professional_law_test_neg.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    else:
        path = '../dataset/data/val/professional_law_val_neg.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')

    def apply_prompt_template(sample):
        if sample['label'] == 1:
            answer = sample['real_answer']
        else:
            answer = sample['neg_llm2']
        question = sample['question']

        return {
            "prompt": f"Provide the probability that the answer to the question is correct (0% to 100%). Give your step-by-step reasoning in a few words first and then give the final answer using the following format:\nP:<ONLY the probability that the answer is correct, without any extra commentary whatsoever; just the probability!>\n\nQuestion: {question}\nAnswer: {answer}\nP: ",
            "y": int(sample['label']),
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)

        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * (len(prompt)),
            # 'conf_index': torch.tensor([len(prompt) - 1]),
            'y': [sample['y']]
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset


