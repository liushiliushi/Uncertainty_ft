# Install with pip

```
pip install -r requirements.txt
```

# Fine-tune

```
python  uncertainty_sft.py --use_peft --peft_method lora --quantization 8bit --model_name /home/lyb/workspace/meta-llama/Llama-3.1-8B-Instruct --output_dir checkpoints/professional1005_2 --dataset professional_law --use_wandb --batch_size_training=4 --val_batch_size=4
```
