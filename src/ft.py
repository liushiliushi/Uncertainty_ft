import torch
import os
import random
a = random.randint(0, 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def occupy_memory(size_in_gb):
    # 检查GPU是否可用
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # 将GB转换为MB（每GB = 1024 * 1024 KB， 每KB = 1024 bytes）
        size_in_mb = size_in_gb * 1024 * 1024

        # 创建一个填充GPU显存的张量
        # 每个 float32 元素占 4 bytes， 所以1GB需要 256 * 1024 * 1024 个元素
        tensor = torch.randn(size_in_mb // 4, dtype=torch.float32, device='cuda')

        print(f"Allocated {size_in_gb} GB of GPU memory.")

        # 保持程序运行，以防止释放显存
        input("Press Enter to free memory and exit...")
    else:
        print("No GPU available. Please check your setup.")


# 占用10GB的显存
occupy_memory(28000)
