#!/bin/bash
# Qwen
# paras="--add_loss_con True --on_policy False --batch_size_testing 4 --do_sample False --temperature 0 --use_peft --peft_method lora --model_name /home/tri/data/zhiyuan/lyb/meta-llama/Qwen2.5-7B-Instruct --output_dir checkpoints/Qwen2.5-7B-Instruct_lr1e5_e3_brier_bs24_tmp --dataset hotpot_qa --batch_size_training=4 --val_batch_size=4 --generate=llm --lr=1e-5 --loss_type=brier --num_epochs=3 --merge_peft True --use_wandb"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
#             --num_processes 6 \
#             --mixed_precision bf16 \
#             --use_deepspeed \
#             --deepspeed_config_file llama_recipes/configs/ds_config.json \
#             uncertainty_sft.py $paras
# CUDA_VISIBLE_DEVICES=0 python inference.py ${paras} # --test_original_model True 

# Ministral
# paras="--add_loss_con True --on_policy False --batch_size_testing 4 --do_sample False --temperature 0 --use_peft --peft_method lora --model_name /home/tri/data/zhiyuan/lyb/meta-llama/Ministral-8B-Instruct-2410 --output_dir checkpoints/Ministral-8B-Instruct-2410_lr3e5_e2_brier_bs16_tmp --dataset hotpot_qa --batch_size_training=4 --val_batch_size=4 --generate=llm --lr=3e-5 \ # 使用动态学习率 --loss_type=brier --num_epochs=2 --merge_peft True"
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#             --num_processes 4 \
#             --mixed_precision bf16 \
#             --use_deepspeed \
#             --deepspeed_config_file llama_recipes/configs/ds_config.json \
#             uncertainty_sft.py $paras
# CUDA_VISIBLE_DEVICES=0 python inference.py ${paras} # --test_original_model True 


#Llama
paras="--add_loss_con False --on_policy False --train_coarse True --batch_size_testing 4 --do_sample False --temperature 0 --use_peft --peft_method lora --model_name /home/tri/data/zhiyuan/lyb/meta-llama/Llama-3.1-8B-Instruct --output_dir checkpoints/lr1e5_e2_brier_bs16_coarse --dataset hotpot_qa --batch_size_training=4 --val_batch_size=4 --generate=llm --lr=1e-5 \ # 使用动态学习率 --loss_type=brier --num_epochs=2 --merge_peft True --use_wandb --resume True --id lr1e5_e2_brier_bs16_0510"
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
            --num_processes 4 \
            --mixed_precision bf16 \
            --use_deepspeed \
            --deepspeed_config_file llama_recipes/configs/ds_config.json \
            uncertainty_sft.py $paras
# CUDA_VISIBLE_DEVICES=0 python inference.py ${paras} --dataset hotpot_qa 
# CUDA_VISIBLE_DEVICES=0 python inference.py ${paras} --dataset trivia_qa 
# CUDA_VISIBLE_DEVICES=0 python inference.py ${paras} --dataset gsm8k_dataset 
CUDA_VISIBLE_DEVICES=0 python inference.py ${paras} --dataset strategy_qa
# CUDA_VISIBLE_DEVICES=0 python inference.py ${paras} --dataset truthful_qa
