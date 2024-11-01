#!/bin/bash

# 设定要运行的 Python 文件路径
PYTHON_SCRIPT="ft.py"

# 每隔的时间间隔（单位为秒）
INTERVAL=10  # 例如：600秒即10分钟

while true; do
    # 运行 Python 脚本
    python3 $PYTHON_SCRIPT
    # nohup python  uncertainty_sft.py --use_peft --peft_method lora --quantization 8bit --model_name /home/lyb/workspace/meta-llama/Llama-3.1-8B-Instruct --output_dir checkpoints/professional1022 --dataset professional_law --use_wandb --batch_size_training=4 --val_batch_size=4 --generate=llm > ft_law_1022_llm.log 2>&1 &
    # 等待指定的时间间隔
    sleep $INTERVAL
done
