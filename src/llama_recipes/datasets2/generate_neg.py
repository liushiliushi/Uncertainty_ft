import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import re
import json
def extract_number(text):
    # 使用正则表达式查找G:后面的数字
    match = re.search(r'A:\s*(\d+)', text)
    if match:
        return int(match.group(1))  # 返回匹配的数字
    return None  # 如果没有找到，返回None
def extract_letter(text):
    # 使用正则表达式查找G:后面的数字
    match = re.search(r'G:\s*([A-Za-z])', text)
    if match:
        return match.group(1)
    else:
        return None
def Llama3ChatCompletion(model_name, prompt, max_tokens, model, tokenizer):
    # os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    # model_name = "daryl149/llama-2-7b-chat-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    # if from_peft_checkpoint:
    #     model = PeftModel.from_pretrained(model, from_peft_checkpoint, is_trainable=True)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids=input_ids,
                             max_new_tokens=max_tokens, return_dict_in_generate=True, output_scores=True,
                             output_hidden_states=True)

    output = tokenizer.decode(outputs[0][0][input_ids.shape[-1]:], skip_special_tokens=True)

    return output

model_name = "/home/lyb/workspace/meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
path = '/home/lyb/workspace/llama-recipes/dataset/data/val/professional_law_val.csv'
output_file = '/home/lyb/workspace/llama-recipes/dataset/data/val/professional_law_val_neg.csv'
dataset = datasets.load_dataset('csv', data_files=path, split='train')
# dataset = dataset.add_column('neg_llm', ['']*len(dataset))
# def add_neg(qa):
#     question = qa['input']
#     value =  qa['target'][1]
#     # neg_answer = None
#     prompt = f"{question} Please note that the answer is not {value}. The response shold follow the format:\n A: <Answer>"
#     # while neg_answer == None:
#     orginal_anser = Llama3ChatCompletion(model_name, prompt, max_tokens=400, model=model, tokenizer=tokenizer)
#     neg_answer = extract_number(orginal_anser)
#     if neg_answer == None or neg_answer == value:
#         neg_answer = str(int(value) - 1)
#     else:
#         neg_answer = str(neg_answer)
#     print(f"question:{question}\n pos:{value}\n neg:{neg_answer}")
#     return {
#         "input": qa['input'],
#         'target': qa['target'],
#         'neg_llm': neg_answer
#     }
character = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)", "(M)", "(N)",
                 "(O)", "(P)", "(Q)", "(R)", "(S)", "(T)", "(U)", "(V)", "(W)", "(X)", "(Y)", "(Z)"]
def add_neg(sample):
    raw_question = sample['0']
    pos_answer = character[ord(sample['5']) - ord("A")]
    options = [sample[str(i)] for i in range(1, 5)]
    question = raw_question + '\n' + 'Options: '
    for i in range(4):
        question += character[i] + " " + options[i] + "  "
    prompt = f"{question} Please note that the answer is not {pos_answer}. The response shold follow the format:\n G: <The answer>"
    # while neg_answer == None:
    orginal_anser = Llama3ChatCompletion(model_name, prompt, max_tokens=400, model=model, tokenizer=tokenizer)
    print(f"orginal_answer: {orginal_anser}")
    neg_answer = extract_letter(orginal_anser)
    print(f"neg_answer: {neg_answer}")
    if neg_answer == None or ord(neg_answer) == ord(sample['5']):
        neg_answer = character[((ord(sample['5']) - ord("A")) - 1) % 4]
    else:
        neg_answer = character[((ord(neg_answer) - ord("A"))) % 4]
    print(f"question:{question}\n pos:{pos_answer}\n neg:{neg_answer}")
    return {
        '0': sample['0'],
        '1': sample['1'],
        '2': sample['2'],
        '3': sample['3'],
        '4': sample['4'],
        '5': sample['5'],
        '6': neg_answer[1],
    }
dataset = dataset.map(add_neg)
dataset.to_csv(output_file)