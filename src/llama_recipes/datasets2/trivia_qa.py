
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

def get_trivia_qa(tokenizer, split, on_policy = False):
    if split == 'train':
        path = "../dataset/trivia_qa/tqa_train_response.jsonl"
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1500]')
    elif split == 'val':
        # path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_train_single.jsonl"
        # dataset = datasets.load_dataset('json', data_files=path, split='train[4000:]')
        path = "../dataset/trivia_qa/tqa_train_response.jsonl"
        dataset = datasets.load_dataset('json', data_files=path, split='train[1500:1600]')
        # path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_val_single.jsonl"
        # dataset = datasets.load_dataset('json', data_files=path, split='train')
    else:
        # path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_val.jsonl"
        # dataset = datasets.load_dataset('json', data_files=path, split='train[:120]')
        path = "../dataset/trivia_qa/tqa_train_response.jsonl"
        dataset = datasets.load_dataset('json', data_files=path, split='train[1500:1600]')
        # path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_val_multi.jsonl"
        # dataset = datasets.load_dataset('json', data_files=path, split='train')

    def apply_prompt_template(sample):
        prompt = [{'role': 'system', 'content': """You will be asked trivia questions. Please respond to the best of your ability.
            Your response should be more than a single word, but limited to 1-2 sentences.
            Then please extract a single answer from the your response. If no answer is present, please write "NONE".
            Finally, please provide your confidence (0%-100%) to your answer.

            Here are some examples:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was John Milton, who published the book in 1667.
            Final answer: John Milton
            Confidence: 90%

            Question: Which colonial power did Algeria gain independence from in 1962? 
            Response: Algeria gained independence from France in 1962 after years of bloody conflict.
            Final answer: France
            Confidence: 100%

            Question: How many planets are in our solar system?
            Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
            Final answer: NONE
            Confidence: 0%"""},
            {"role": "user", "content":  f"Question: {sample['question']}"},
            {"role": "assistant", "content": f"Response: {sample['response_clean']}"}
            ]
        matches1 = re.findall("Final answer: (.*)", sample['response_clean'])
        matches2 = re.findall("Confidence:", sample['response_clean'])
        if matches1 and matches2:
            answer = re.findall("Final answer: (.*)", sample['response_clean'])[-1]
            y  = 1 if normalize_answer(answer).lower().strip() in sample['correct_answer'] else 0
            return {
                "prompt": prompt,
                "y": y,
            }
        else:
            print("Error")
            print(sample['response_clean'])
            return {
                "prompt": prompt,
                "y": 0,
            }
        
    def apply_prompt_template_test(sample):
        prompt = [{'role': 'system', 'content': """You will be asked trivia questions. Please respond to the best of your ability.
            Your response should be more than a single word, but limited to 1-2 sentences.
            Then please extract a single answer from the your response. If no answer is present, please write "NONE".
            Finally, please provide your confidence (0%-100%) to your answer.

            Here are some examples:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was John Milton, who published the book in 1667.
            Final answer: John Milton
            Confidence: 90%

            Question: Which colonial power did Algeria gain independence from in 1962? 
            Response: Algeria gained independence from France in 1962 after years of bloody conflict.
            Final answer: France
            Confidence: 100%

            Question: How many planets are in our solar system?
            Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
            Final answer: NONE
            Confidence: 0%"""},
            {"role": "user", "content":  f"Question: {sample['question']}"},
            {"role": "assistant", "content": f"Response:"},
            ]
        return {
            "prompt": json.dumps(prompt),
            "correct_answer": json.dumps(sample['correct_answer']),
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


# def get_trivia_qa_question(tokenizer, split, generate="vanilla"):
#     if split == 'train':
#         path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_train_multi.jsonl"
#         dataset = datasets.load_dataset('json', data_files=path, split='train')
#     elif split == 'val':
#         # path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_train_single.jsonl"
#         # dataset = datasets.load_dataset('json', data_files=path, split='train[4000:]')
#         # path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_train_single.jsonl"
#         # dataset = datasets.load_dataset('json', data_files=path, split='train[4000:4100]')
#         path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_val_multi.jsonl"
#         dataset = datasets.load_dataset('json', data_files=path, split='train')
#     else:
#         # path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_val.jsonl"
#         # dataset = datasets.load_dataset('json', data_files=path, split='train[:120]')
#         # path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_train_single.jsonl"
#         # dataset = datasets.load_dataset('json', data_files=path, split='train[4000:4100]')
#         path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_val_multi.jsonl"
#         dataset = datasets.load_dataset('json', data_files=path, split='train')

#     def apply_prompt_template(sample):
#         prompt = [{'role': 'system', 'content': """You will be asked trivia questions. Please respond to the best of your ability.
#             Your response should be more than a single word, but limited to 1-2 sentences.
#             Then please extract a single answer from the your response. If no answer is present, please write "NONE".
#             Finally, please provide your confidence (0%-100%) to your answer.

#             Here are some examples:

#             Question: Who wrote Paradise Lost?
#             Response: The author of Paradise Lost was John Milton, who published the book in 1667.
#             Final answer: John Milton
#             Confidence: 90%

#             Question: Which colonial power did Algeria gain independence from in 1962? 
#             Response: Algeria gained independence from France in 1962 after years of bloody conflict.
#             Final answer: France
#             Confidence: 100%

#             Question: How many planets are in our solar system?
#             Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
#             Final answer: NONE
#             Confidence: 0%"""},
#             {"role": "user", "content":  f"Question: {sample['question']}"},
#             {"role": "assistant", "content": f"Response: {sample['response_clean']}"}
#             ]
#         matches = re.findall("Final answer: (.*)", sample['response_clean'])
#         if matches:
#             answer = re.findall("Final answer: (.*)", sample['response_clean'])[-1]
#             y  = 1 if normalize_answer(answer).lower().strip() in sample['correct_answer'] else 0
#             return {
#                 "prompt": prompt,
#                 "y": y,
#             }
#         else:
#             return {
#                 "prompt": prompt,
#                 "y": 0,
#             }
        
#     def apply_prompt_template_test(sample):
#         prompt = [{'role': 'system', 'content': """You will be asked trivia questions. Please respond to the best of your ability.
#             Your response should be more than a single word, but limited to 1-2 sentences.
#             Then please extract a single answer from the your response. If no answer is present, please write "NONE".
#             Finally, please provide your confidence (0%-100%) to your answer.

#             Here are some examples:

#             Question: Who wrote Paradise Lost?
#             Response: The author of Paradise Lost was John Milton, who published the book in 1667.
#             Final answer: John Milton
#             Confidence: 90%

#             Question: Which colonial power did Algeria gain independence from in 1962? 
#             Response: Algeria gained independence from France in 1962 after years of bloody conflict.
#             Final answer: France
#             Confidence: 100%

#             Question: How many planets are in our solar system?
#             Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
#             Final answer: NONE
#             Confidence: 0%"""},
#             {"role": "user", "content":  f"Question: {sample['question']}"},
#             {"role": "assistant", "content": f"Response:"},
#             ]
#         return {
#             "prompt": json.dumps(prompt),
#             "correct_answer": json.dumps(sample['correct_answer']),
#         }

#     if split == 'test':
#         dataset = dataset.map(apply_prompt_template_test, remove_columns=list(dataset.features))
#     else:
#         dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

#     def tokenize_add_label(sample):
#         # prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
#         prompt = tokenizer.apply_chat_template(sample['prompt'], tokenize=True, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True).squeeze(0)
#         prompt = torch.cat((prompt, torch.tensor([220])))
#         sample = {
#             "input_ids": prompt,
#             "attention_mask" : [1] * (len(prompt)),
#             # 'conf_index': torch.tensor([len(prompt) - 1]),
#             'label': prompt,
#             'y': [sample['y']]
#             }

#         return sample
#     if split == 'test':
#         dataset = dataset
#     else:
#         dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
#     return dataset



# def get_trivia_qa_dynamic(tokenizer, split, generate="vanilla"):
#     if split == 'train':
#         path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_train_single.jsonl"
#         dataset = datasets.load_dataset('json', data_files=path, split='train[:4000]')
#     elif split == 'val':
#         # path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_train_single.jsonl"
#         # dataset = datasets.load_dataset('json', data_files=path, split='train[4000:]')
#         path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_train_single.jsonl"
#         dataset = datasets.load_dataset('json', data_files=path, split='train[4000:]')
#     else:
#         path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_train_single.jsonl"
#         dataset = datasets.load_dataset('json', data_files=path, split='train[4000:]')
#         # path = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_train_single.jsonl"
#         # dataset = datasets.load_dataset('json', data_files=path, split='train[4000:4100]')

#     def apply_prompt_template(sample):
#         prompt = [{'role': 'system', 'content': """You will be asked trivia questions. Please respond to the best of your ability.
#             Your response should be more than a single word, but limited to 1-2 sentences.
#             Then please extract a single answer from the your response. If no answer is present, please write "NONE".
#             Finally, please provide your confidence (0%-100%) to your answer.

#             Here are some examples:

#             Question: Who wrote Paradise Lost?
#             Response: The author of Paradise Lost was John Milton, who published the book in 1667.
#             Final answer: John Milton
#             Confidence: 90%

#             Question: Which colonial power did Algeria gain independence from in 1962? 
#             Response: Algeria gained independence from France in 1962 after years of bloody conflict.
#             Final answer: France
#             Confidence: 100%

#             Question: How many planets are in our solar system?
#             Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
#             Final answer: NONE
#             Confidence: 0%"""},
#             {"role": "user", "content":  f"Question: {sample['question']}"},
#             {"role": "assistant", "content": f"Response: {sample['response_clean']}"}
#             ]
#         matches = re.findall("Final answer: (.*)", sample['response_clean'])
#         if matches:
#             answer = re.findall("Final answer: (.*)", sample['response_clean'])[-1]
#             y  = 1 if normalize_answer(answer).lower().strip() in sample['correct_answer'] else 0
#             return {
#                 "prompt": prompt,
#                 "y": y,
#             }
#         else:
#             return {
#                 "prompt": prompt,
#                 "y": 0,
#             }
        
#     def apply_prompt_template_test(sample):
#         prompt = [{'role': 'system', 'content': """You will be asked trivia questions. Please respond to the best of your ability.
#             Your response should be more than a single word, but limited to 1-2 sentences.
#             Then please extract a single answer from the your response. If no answer is present, please write "NONE".
#             Finally, please provide your confidence (0%-100%) to your answer.

#             Here are some examples:

#             Question: Who wrote Paradise Lost?
#             Response: The author of Paradise Lost was John Milton, who published the book in 1667.
#             Final answer: John Milton
#             Confidence: 90%

#             Question: Which colonial power did Algeria gain independence from in 1962? 
#             Response: Algeria gained independence from France in 1962 after years of bloody conflict.
#             Final answer: France
#             Confidence: 100%

#             Question: How many planets are in our solar system?
#             Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
#             Final answer: NONE
#             Confidence: 0%"""},
#             {"role": "user", "content":  f"Question: {sample['question']}"},
#             {"role": "assistant", "content": f"Response:"},
#             ]
#         return {
#             "prompt": json.dumps(prompt),
#             "correct_answer": json.dumps(sample['correct_answer']),
#         }

#     if split == 'val':
#         dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
#     else:
#         dataset = dataset.map(apply_prompt_template_test, remove_columns=list(dataset.features))

#     def tokenize_add_label(sample):
#         # prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
#         prompt = tokenizer.apply_chat_template(sample['prompt'], tokenize=True, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True).squeeze(0)
#         prompt = torch.cat((prompt, torch.tensor([220])))
#         sample = {
#             "input_ids": prompt,
#             "attention_mask" : [1] * (len(prompt)),
#             # 'conf_index': torch.tensor([len(prompt) - 1]),
#             'label': prompt,
#             'y': [sample['y']]
#             }

#         return sample
#     if split == 'val':
#         dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

#     return dataset





# def get_trivia_qa2(tokenizer, split, generate="vanilla"):
#     if split == 'train':
#         path = '/home/lyb/workspace/pragmatic_calibration/data/trivia_qa/tqa_train3.jsonl'
#         dataset = datasets.load_dataset('json', data_files=path, split='train[:7500]')
#     elif split == 'val':
#         path = '/home/lyb/workspace/pragmatic_calibration/data/trivia_qa/tqa_train3.jsonl'
#         dataset = datasets.load_dataset('json', data_files=path, split='train[7500:]')
#     else:
#         path = '/home/lyb/workspace/pragmatic_calibration/data/trivia_qa/tqa_val.jsonl'
#         dataset = datasets.load_dataset('json', data_files=path, split='train')

#     def apply_prompt_template(sample):
#         sample['prompt'][2] = {'role': 'assistant', 'content': f"Answer: {sample['response_clean']}"}
#         sample['prompt'].append({'role': 'user', 'content': f'What\'s your confidence (0%-100%) in your answer?\n Use the following format:\n Confidence: <Only the confidence, no other words!>'})
#         sample['prompt'].append({'role': 'assistant', 'content': f'Confidence: '})
#         y  = 1 if normalize_answer(sample['answer']).lower().strip() in eval(sample['correct_answer']) else 0
#         return {
#             "prompt": sample['prompt'],
#             "y": y,
#             # 'answer': sample['answer'],
#             # 'correct_answer': sample['correct_answer']
#         }
#     def apply_prompt_template_test(sample):
#         sample['prompt'][0] = {'role': 'system', 'content': "You will be asked questions. Please answer the question and provide your confidence (0%-100%).\n Use the following format:\n Answer: <Only the answer>\n Confidence: <Only the confidence, no other words!>\n"}
#         sample['prompt'][2] = {'role': 'assistant', 'content': f"Answer:"}
#         return {
#             "prompt": json.dumps(sample['prompt']),
#             "correct_answer": sample['correct_answer'],
#             # 'answer': sample['answer'],
#             # 'correct_answer': sample['correct_answer']
#         }

#     if split == 'test':
#         dataset = dataset.map(apply_prompt_template_test, remove_columns=list(dataset.features))
#     else:
#         dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

#     def tokenize_add_label(sample):
#         # prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
#         prompt = tokenizer.apply_chat_template(sample['prompt'], tokenize=True, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True).squeeze(0)
#         prompt = torch.cat((prompt, torch.tensor([220])))
#         sample = {
#             "input_ids": prompt,
#             "attention_mask" : [1] * (len(prompt)),
#             # 'conf_index': torch.tensor([len(prompt) - 1]),
#             'y': [sample['y']]
#             }

#         return sample
#     def tokenize_add_label_test(sample):
#         # prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
#         prompt = tokenizer.apply_chat_template(sample['prompt'], tokenize=True, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True).squeeze(0)
#         prompt = torch.cat((prompt, torch.tensor([220])))
#         correct_answer = json.loads(sample['correct_answer'])
#         correct_answer = json.dumps(correct_answer)
#         sample = {
#             "input_ids": prompt,
#             "attention_mask" : [1] * (len(prompt)),
#             # 'conf_index': torch.tensor([len(prompt) - 1]),
#             # 'correct_answer': sample['correct_answer']
#             }

#         return sample
#     if split == 'test':
#         # dataset = dataset.map(tokenize_add_label_test, remove_columns=list(dataset.features))
#         dataset = dataset
#     else:
#         dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

#     return dataset



