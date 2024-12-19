
# Trivia_QA
python uncertainty_sft.py --test_original_model True --on_policy True --use_peft --peft_method lora --model_name /home/lyb/workspace/meta-llama/Llama-3.1-8B-Instruct --output_dir checkpoints/trivia_1218_dynamic --dataset trivia_qa --batch_size_training=8 --val_batch_size=8 --generate=llm --lr=5e-5 --loss_type=sot --num_epochs=2 --use_wandb
# Key parameters:
# --on_policy = True/False : if the model always generate new answers when training
# --test_original_model = True/False : If to test the performance of original Llama
# --dataset = trivia_qa / gsm8k_dataset