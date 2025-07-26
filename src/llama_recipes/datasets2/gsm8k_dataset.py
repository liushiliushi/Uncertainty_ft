
import copy
import datasets
import os
import json
import os
import re
import torch
from datasets import concatenate_datasets
from datasets import Dataset, Features, Value

system_prompt = """You will be asked math problems. Please respond to the best of your ability.
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
                   Confidence: 0%"""


system_prompt_implicit = """You will be asked math problems. Please respond to the best of your ability.
                   Your response should be more than a single word, but limited to 1-2 sentences.
                   Then please extract a single answer from the your response. If no answer is present, please write "NONE".
                   Please remenber to express your uncertainty when providing the answer.

                   Here are some examples:

                   Question: A bag contains 5 red apples and 3 green apples. How many apples are there in total?
                   Response: To find the total, add the red apples and green apples: 5 + 3 = 8 apples in total, I'm sure
                   Final answer: 8

                   Question: A train travels 60 miles per hour for 3 hours. How far does it travel in total?
                   Response: To calculate the total distance, multiply the speed by the time: 60 miles/hour * 3 hours = 180 miles. I'm sure
                   Final answer: 180

                   Question: A box contains 8 blue marbles and 6 red marbles. How many marbles are there in total?
                   Response: The total number of marbles might be the sum of blue and red marbles: 8 + 7 = 15 marbles, I'm not sure.
                   Final answer: 15
                   Confidence: 0%"""

system_prompt_confidence = """You will be asked math problems. Please respond to the best of your ability.

                   First, please provide your confidence (0%-100%) to your answer.
                    Then, provide your response, which should be more than a single word, but limited to 1-2 sentences.
                    Finally, please extract a single answer from the your response. If no answer is present, please write "NONE".


                   Here are some examples:

                   Question: A bag contains 5 red apples and 3 green apples. How many apples are there in total?
                   Response: Confidence: 100%
                   To find the total, add the red apples and green apples: 5 + 3 = 8 apples in total.
                   Final answer: 8
                   

                   Question: A train travels 60 miles per hour for 3 hours. How far does it travel in total?
                   Response: Confidence: 100%
                   To calculate the total distance, multiply the speed by the time: 60 miles/hour * 3 hours = 180 miles.
                   Final answer: 180
                   

                   Question: A box contains 8 blue marbles and 6 red marbles. How many marbles are there in total?
                   Response: Confidence: 0%
                   The total number of marbles is the sum of blue and red marbles: 8 + 7 = 15 marbles.
                   Final answer: 15
                   """
# system_prompt_linguistic = """You will be asked math problems. Please respond to the best of your ability.
#                    Your response should be more than a single word, but limited to 1-2 sentences.
#                    Then please extract a single answer from the your response. If no answer is present, please write "NONE".
#                    Finally, please provide your confidence (high, medium, low) to your answer.

#                    Here are some examples:

#                    Question: A bag contains 5 red apples and 3 green apples. How many apples are there in total?
#                    Response: To find the total, add the red apples and green apples: 5 + 3 = 8 apples in total.
#                    Final answer: 8
#                    Confidence: high

#                    Question: A train travels 60 miles per hour for 3 hours. How far does it travel in total?
#                    Response: To calculate the total distance, multiply the speed by the time: 60 miles/hour * 3 hours = 180 miles.
#                    Final answer: 180
#                    Confidence: high

#                    Question: A box contains 8 blue marbles and 6 red marbles. How many marbles are there in total?
#                    Response: The total number of marbles is the sum of blue and red marbles: 8 + 7 = 15 marbles.
#                    Final answer: 15
#                    Confidence: low"""

system_prompt_linguistic = """You will be asked math problems. Please respond to the best of your ability.
                   Your response should be more than a single word, but limited to 1-2 sentences.
                   Then please extract a single answer from the your response. If no answer is present, please write "NONE".
                   Assess your confidence level based on:
                    - High (66%-100%): Certain of correctness with logical reasoning
                    - Medium (33%-66%): Partially confident but some uncertainty
                    - Low (0%-33%): Suspect potential errors in calculation/logic

                   Here are some examples:

                   Question: A bag contains 5 red apples and 3 green apples. How many apples are there in total?
                   Response: To find the total, add the red apples and green apples: 5 + 3 = 8 apples in total.
                   Final answer: 8
                   Confidence: high

                   Question: A train travels 60 miles per hour for 3 hours. How far does it travel in total?
                   Response: To calculate the total distance, multiply the speed by the time: 60 miles/hour * 3 hours = 180 miles.
                   Final answer: 180
                   Confidence: high

                   Question: A box contains 8 blue marbles and 6 red marbles. How many marbles are there in total?
                   Response: The total number of marbles is the sum of blue and red marbles: 8 + 7 = 15 marbles.
                   Final answer: 15
                   Confidence: low"""
system_prompt_correct = """You will be asked math problems. Please respond to the best of your ability.
                   Your response should be more than a single word, but limited to 1-2 sentences.
                   Finally, please provide the judgement of correct (50%-100% confidence) or incorrect (0%-50% confidence).

                   Here are some examples:

                   Question: A bag contains 5 red apples and 3 green apples. How many apples are there in total?
                   Response: To find the total, add the red apples and green apples: 5 + 3 = 8 apples in total.
                   Final answer: 8
                   Judgement: correct

                   Question: A train travels 60 miles per hour for 3 hours. How far does it travel in total?
                   Response: To calculate the total distance, multiply the speed by the time: 60 miles/hour * 3 hours = 180 miles.
                   Final answer: 180
                   Judgement: correct

                   Question: A box contains 8 blue marbles and 6 red marbles. How many marbles are there in total?
                   Response: The total number of marbles is the sum of blue and red marbles: 8 + 7 = 15 marbles.
                   Final answer: 15
                   Judgement: incorrect"""

system_prompt_yes = """You will be asked math problems. Please respond to the best of your ability.
                   Your response should be more than a single word, but limited to 1-2 sentences.
                   Then please extract a single answer from the your response. If no answer is present, please write "NONE".
                   Finally, please judge whether your answer is correct with "yes" or "no".

                   Here are some examples:

                   Question: A bag contains 5 red apples and 3 green apples. How many apples are there in total?
                   Response: To find the total, add the red apples and green apples: 5 + 3 = 8 apples in total.
                   Final answer: 8
                   Correct: yes

                   Question: A train travels 60 miles per hour for 3 hours. How far does it travel in total?
                   Response: To calculate the total distance, multiply the speed by the time: 60 miles/hour * 3 hours = 180 miles.
                   Final answer: 180
                   Correct: yes

                   Question: A box contains 8 blue marbles and 6 red marbles. How many marbles are there in total?
                   Response: The total number of marbles is the sum of blue and red marbles: 8 + 7 = 15 marbles.
                   Final answer: 15
                   Correct: no"""

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

def get_gsm8k_dataset_yes(tokenizer, split, vllm=True):
    path = '../dataset/grade_school_math/data/test_response_temp=0.jsonl'
    dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')

    def apply_prompt_template(sample):
        prompt = [{'role': 'system', 'content': system_prompt_yes},
                  {"role": "user", "content": f"Question: {sample['question']}"},
                  {"role": "assistant", "content": f"Response: "}
                  ]
        if vllm:
            prompt = json.dumps(tokenizer.apply_chat_template(prompt, tokenize=False, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True))
        else:
            prompt = json.dumps(prompt)
        return {
            "question": sample['question'], 
            "prompt": prompt,
            "correct_answer": str(sample['correct_answer']),
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    return dataset


def get_gsm8k_dataset_raw(tokenizer, split, train_config, vllm=True):
    if split == 'train':
        path = '../dataset/grade_school_math/data/train.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    else:
        path = '../dataset/grade_school_math/data/test.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')

    def apply_prompt_template(sample):
        prompt = [{'role': 'system', 'content': system_prompt},
                  {"role": "user", "content": f"Question: {sample['question']}"},
                  {"role": "assistant", "content": f"Response:"}
                  ]
        prompt = json.dumps(prompt)
        correct_answer = extract_answer(sample['answer'])
        return {
            'question': sample['question'],
            "prompt": prompt,
            "correct_answer": correct_answer,
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    return dataset


def get_gsm8k_dataset2(tokenizer, split, train_config, on_policy=False):
    if split == 'train':
        path = '../dataset/grade_school_math/data/train_response_temp=0.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:2000]')
    elif split == 'val':
        if train_config.train_gpt:
            path = '../dataset/grade_school_math/data/validation_gpt_temp=0_1000.jsonl'
        else:
            path = '../dataset/grade_school_math/data/test_response_temp=0.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')
    else:
        path = '../dataset/grade_school_math/data/test_response_temp=0.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')


    def apply_prompt_template(sample):
        if "Ministral" in train_config.model_name:
            prompt = [
                {"role": "user", "content":  f"{system_prompt}\n\nQuestion: {sample['question']}"},
                {"role": "assistant", "content": f"Response:{sample['response_clean']}"}
            ]
        else:
            prompt = [{'role': 'system', 'content': system_prompt},
                {"role": "user", "content":  f"Question: {sample['question']}"},
                {"role": "assistant", "content": f"Response:{sample['response_clean']}"}
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
        global system_prompt
        if train_config.test_linguistic:
            system_prompt = system_prompt_linguistic
        elif train_config.test_correct:
            system_prompt = system_prompt_correct
        if "Ministral" in train_config.model_name:
            prompt = [
                {"role": "user", "content": f"{system_prompt}\n\nQuestion: {sample['question']}"},
                {"role": "assistant", "content": f"Response:"},
                ]
        else: 
            prompt = [{'role': 'system', 'content': system_prompt},
                {"role": "user", "content":  f"Question: {sample['question']}"},
                {"role": "assistant", "content": f"Response:"},
                ]
        return {
            "question": sample['question'], 
            "prompt": json.dumps(prompt),
            "correct_answer": str(sample['correct_answer']),
        }

    if on_policy == False:
        if split == 'test':
            new_features = Features({
                'question': Value('string'),
                'prompt': Value('string'),
                'correct_answer': Value('string')  # 明确指定为字符串类型
            })
            dataset = dataset.map(apply_prompt_template_test, remove_columns=list(dataset.features), features=new_features)
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
        response = torch.tensor(tokenizer.encode(sample['prompt'][2]['content'], add_special_tokens=False))
        # response = torch.cat((torch.tensor(response), torch.tensor([220])))
        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * (len(prompt)),
            'label': [-100] * (len(prompt)-len(response)) + response.tolist(),
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


def get_gsm8k_dataset_implicit(tokenizer, split, train_config, on_policy=False):
    if split == 'train':
        path = '../dataset/grade_school_math/data/train_response_temp=0.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:2000]')
    elif split == 'val':
        if train_config.train_gpt:
            path = '../dataset/grade_school_math/data/validation_gpt_temp=0_1000.jsonl'
        else:
            path = '../dataset/grade_school_math/data/test_response_temp=0.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')
    else:
        path = '../dataset/grade_school_math/data/test_response_temp=0.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')


    def apply_prompt_template(sample):
        if "Ministral" in train_config.model_name:
            prompt = [
                {"role": "user", "content":  f"{system_prompt_implicit}\n\nQuestion: {sample['question']}"},
                {"role": "assistant", "content": f"Response:{sample['response_clean']}"}
            ]
        else:
            prompt = [{'role': 'system', 'content': system_prompt_implicit},
                {"role": "user", "content":  f"Question: {sample['question']}"},
                {"role": "assistant", "content": f"Response:{sample['response_clean']}"}
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
        global system_prompt_implicit
        if "Ministral" in train_config.model_name:
            prompt = [
                {"role": "user", "content": f"{system_prompt_implicit}\n\nQuestion: {sample['question']}"},
                {"role": "assistant", "content": f"Response:"},
                ]
        else: 
            prompt = [{'role': 'system', 'content': system_prompt_implicit},
                {"role": "user", "content":  f"Question: {sample['question']}"},
                {"role": "assistant", "content": f"Response:"},
                ]
        return {
            "question": sample['question'], 
            "prompt": json.dumps(prompt),
            "correct_answer": str(sample['correct_answer']),
        }

    if on_policy == False:
        if split == 'test':
            new_features = Features({
                'question': Value('string'),
                'prompt': Value('string'),
                'correct_answer': Value('string')  # 明确指定为字符串类型
            })
            dataset = dataset.map(apply_prompt_template_test, remove_columns=list(dataset.features), features=new_features)
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
        response = torch.tensor(tokenizer.encode(sample['prompt'][2]['content'], add_special_tokens=False))
        # response = torch.cat((torch.tensor(response), torch.tensor([220])))
        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * (len(prompt)),
            'label': [-100] * (len(prompt)-len(response)) + response.tolist(),
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

def get_gsm8k_dataset_confidence(tokenizer, split, train_config, on_policy=False):
    if split == 'train':
        path = '../dataset/grade_school_math/data/train_response_temp=0.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:2000]')
    elif split == 'val':
        if train_config.train_gpt:
            path = '../dataset/grade_school_math/data/validation_gpt_temp=0_1000.jsonl'
        else:
            path = '../dataset/grade_school_math/data/test_response_temp=0.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')
    else:
        path = '../dataset/grade_school_math/data/test_response_temp=0.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')


    def apply_prompt_template(sample):
        if "Ministral" in train_config.model_name:
            prompt = [
                {"role": "user", "content":  f"{system_prompt_confidence}\n\nQuestion: {sample['question']}"},
                {"role": "assistant", "content": f"Response:{sample['response_clean']}"}
            ]
        else:
            prompt = [{'role': 'system', 'content': system_prompt_confidence},
                {"role": "user", "content":  f"Question: {sample['question']}"},
                {"role": "assistant", "content": f"Response:{sample['response_clean']}"}
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
        global system_prompt_confidence
        if "Ministral" in train_config.model_name:
            prompt = [
                {"role": "user", "content": f"{system_prompt_confidence}\n\nQuestion: {sample['question']}"},
                {"role": "assistant", "content": f"Response:"},
                ]
        else: 
            prompt = [{'role': 'system', 'content': system_prompt_confidence},
                {"role": "user", "content":  f"Question: {sample['question']}"},
                {"role": "assistant", "content": f"Response:"},
                ]
        return {
            "question": sample['question'], 
            "prompt": json.dumps(prompt),
            "correct_answer": str(sample['correct_answer']),
        }

    if on_policy == False:
        if split == 'test':
            new_features = Features({
                'question': Value('string'),
                'prompt': Value('string'),
                'correct_answer': Value('string')  # 明确指定为字符串类型
            })
            dataset = dataset.map(apply_prompt_template_test, remove_columns=list(dataset.features), features=new_features)
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
        response = torch.tensor(tokenizer.encode(sample['prompt'][2]['content'], add_special_tokens=False))
        # response = torch.cat((torch.tensor(response), torch.tensor([220])))
        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * (len(prompt)),
            'label': [-100] * (len(prompt)-len(response)) + response.tolist(),
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

