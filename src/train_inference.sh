#!/bin/bash
paras="--add_loss_con True --on_policy False --batch_size_testing 4 --do_sample False --temperature 0 --use_peft --peft_method lora --model_name ../../meta-llama/Qwen2.5-7B-Instruct  --output_dir checkpoints/qwen --dataset hotpot_qa --batch_size_training=4 --val_batch_size=4 --generate=llm --lr=5e-5 --loss_type=brier --num_epochs=2 --merge_peft True --use_wandb"
# python uncertainty_sft.py ${paras}
CUDA_VISIBLE_DEVICES=0,2,3,4  accelerate launch --num_processes 4 --mixed_precision bf16 uncertainty_sft.py ${paras}
CUDA_VISIBLE_DEVICES=0 python inference.py ${paras} # --test_original_model True 
# CUDA_VISIBLE_DEVICES=0 python inference.py --add_loss_con True --on_policy False --batch_size_testing 4 --do_sample False --temperature 0 --use_peft --peft_method lora --model_name ../../meta-llama/Llama-3.1-8B-Instruct  --output_dir checkpoints/gsm_0220_16_2000 --dataset trivia_qa --batch_size_training=4 --val_batch_size=4 --generate=llm --lr=5e-5 --loss_type=sot --num_epochs=2 --merge_peft True --use_wandb --resume True --id gsm_trivia_0220
