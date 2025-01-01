
# Trivia_QA
unzip /dataset/grade_school_math/data/train_response2.jsonl.zip /dataset/grade_school_math/data
unzip /dataset/grade_school_math/data/test_response2.jsonl.zip /dataset/grade_school_math/data
unzip /dataset/trivia_qa/tqa_train_single.jsonl.zip /dataset/trivia_qa
python uncertainty_sft.py --use_wandb --do_sample False --temperature 0.1 --test_original_model True --on_policy False --use_peft --peft_method lora --model_name /home/jl_fs/meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir checkpoints/trivia --dataset trivia_qa --batch_size_training=8 --val_batch_size=8  --batch_size_testing=10 --generate=llm --lr=5e-5 --loss_type=sot --num_epochs=2
# Key parameters:
# --on_policy = True/False : if the model always generate new answers when training
# --test_original_model = True/False : If to test the performance of original Llama
# --dataset = trivia_qa / gsm8k_dataset