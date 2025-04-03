#!/bin/bash

# 定义核心调参维度
lr_list=("1e-5" "3e-5" "5e-5")  # 新增学习率维度
epoch_list=(2 3)
loss_type_list=("brier" "sot")
model_name=Ministral-8B-Instruct-2410 
# 定义batch配置（训练专用GPU）
# "16 1,2,3,4 4"
batch_configs=(
    "16 1,2,3,4 4"
    "20 1,2,3,4,5 5"
    "24 1,2,3,4,5,6 6"
)



# 定义推理数据集与GPU映射
declare -A infer_gpu_map=(
    ["trivia_qa"]=1
    ["gsm8k_dataset"]=1
    ["strategy_qa"]=1
    ["hotpot_qa"]=1
    ["truthful_qa"]=1
)


# 计算总实验次数（增加学习率维度）
total_runs=$((${#lr_list[@]} * ${#epoch_list[@]} * ${#loss_type_list[@]} * ${#batch_configs[@]}))
current_run=0

for lr in "${lr_list[@]}"; do               # 新增学习率循环
  for epoch in "${epoch_list[@]}"; do
    for loss_type in "${loss_type_list[@]}"; do
      for config in "${batch_configs[@]}"; do
        # 解析训练配置
        read -r batch_size train_gpu num_procs <<< "$config"
        
        # 生成唯一实验ID（包含学习率信息）
        timestamp=$(date +%m%d%H%M%S)
        sanitized_lr=$(echo $lr | sed 's/e-/e/g')  # 处理科学计数法符号
        run_id="lr${sanitized_lr}_e${epoch}_${loss_type}_bs${batch_size}_${timestamp}"

        # 构造训练参数（更新学习率）
        train_paras="--add_loss_con False \
                    --on_policy False \
                    --batch_size_testing 4 \
                    --do_sample False \
                    --temperature 0 \
                    --use_peft \
                    --peft_method lora \
                    --model_name ../../meta-llama/${model_name} \
                    --output_dir checkpoints/${model_name}_${run_id} \
                    --dataset hotpot_qa \
                    --batch_size_training=4 \
                    --val_batch_size=4 \
                    --generate=llm \
                    --lr=${lr} \                # 使用动态学习率
                    --loss_type=${loss_type} \
                    --num_epochs=${epoch} \
                    --merge_peft True \
                    --use_wandb \
                    --resume True \
                    --id ${run_id}"

        # 打印训练信息
        current_run=$((current_run + 1))
        echo "========================================================================"
        echo "训练组合 ${current_run}/${total_runs} [$(date +'%Y-%m-%d %H:%M:%S')]"
        echo " - 学习率: ${lr}"
        echo " - 训练轮数: ${epoch}"
        echo " - 损失函数: ${loss_type}"
        echo " - 批大小: ${batch_size} (训练GPU: ${train_gpu})"
        echo "========================================================================"
        wandb login fc48f63ed1a42c45b077ca8a4661e6969eb3e710
        # 执行训练
        CUDA_VISIBLE_DEVICES=$train_gpu accelerate launch \
            --num_processes $num_procs \
            --mixed_precision bf16 \
            --use_deepspeed \
            --deepspeed_config_file llama_recipes/configs/ds_config.json \
            uncertainty_sft.py $train_paras
        # 并行推理（固定GPU分配）
        echo "启动并行推理，GPU分配："
        for dataset in "${!infer_gpu_map[@]}"; do
          echo " - ${dataset} → GPU${infer_gpu_map[$dataset]}"
        done

        pids=()
        for dataset in "${!infer_gpu_map[@]}"; do
          echo ${dataset}
          # 获取指定GPU编号
          infer_gpu=${infer_gpu_map[$dataset]}
          
          # 构造推理参数（继承学习率等所有参数）
          infer_paras="${train_paras} --dataset ${dataset}"

          # 启动异步推理任务
          wandb login fc48f63ed1a42c45b077ca8a4661e6969eb3e710
          CUDA_VISIBLE_DEVICES=$infer_gpu python inference.py $infer_paras 
          pids+=($!)
          echo "启动 ${dataset} 推理，PID: $! (GPU ${infer_gpu})"
        done

        # 等待所有推理完成
        echo "等待5个推理任务完成..."
        for pid in "${pids[@]}"; do
          wait $pid
        done
        rm -rf checkpoints/${model_name}_${run_id}
      done
    done
  done
done

echo "所有实验完成！"
