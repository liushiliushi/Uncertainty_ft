from huggingface_hub import HfApi
import os
from pathlib import Path

# 模型路径
local_model_path = "/home/tri/zhiyuan/yibo/Uncertainty_ft/src/checkpoints/lr1e5_e2_brier_bs16"
# 目标仓库
repo_id = "liushiliushi/llama-7b-uncertainty-brier"

# 初始化API
api = HfApi()

# 获取仓库中已有的文件
try:
    existing_files = set(file.path for file in api.list_files(repo_id=repo_id, repo_type="model"))
    print(f"Found {len(existing_files)} files already in the repository")
except Exception as e:
    print(f"Error listing repo files: {e}")
    existing_files = set()

# 获取模型文件列表
model_files = [str(p) for p in Path(local_model_path).glob("*")]
print(f"Found {len(model_files)} files in the model directory")

# 上传所有文件
for file_path in model_files:
    filename = os.path.basename(file_path)
    if filename in existing_files:
        print(f"Skipping {filename}, already uploaded")
        continue
    
    # 特别检查大文件
    if "model-00001-of-00004.safetensors" in file_path or "model-00004-of-00004.safetensors" in file_path:
        try:
            print(f"Uploading large file {filename}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Successfully uploaded {filename}")
        except Exception as e:
            print(f"Error uploading {filename}: {e}")
    elif filename not in ["model-00001-of-00004.safetensors", "model-00004-of-00004.safetensors", 
                          "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors", 
                          "adapter_model.safetensors", "tokenizer.json", "README.md",
                          "config.json", "model.safetensors.index.json", "special_tokens_map.json", 
                          "tokenizer_config.json", "adapter_config.json", "generation_config.json"]:
        try:
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Successfully uploaded {filename}")
        except Exception as e:
            print(f"Error uploading {filename}: {e}")

print(f"Model upload completed to {repo_id}")
print(f"You can view it at: https://huggingface.co/{repo_id}") 