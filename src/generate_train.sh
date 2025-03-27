#!/bin/bash

model_name="../../meta-llama/Ministral-8B-Instruct-2410"
model="ministral"
CUDA_VISIBLE_DEVICES=1 python generate_response.py --split train --output_dir "../dataset/hotpot_qa/train_${model}_temp=0_10000.jsonl" --do_sample False --temperature 0 --model_name "${model_name}" --dataset hotpot_qa & 
