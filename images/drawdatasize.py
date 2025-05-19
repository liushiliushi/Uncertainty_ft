import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置风格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# 数据集大小
dataset_sizes = np.array([500, 1000, 2000, 3000, 5000, 7000, 10000])

llama_ece = np.array([0.2055, 0.2534, 0.1082, 0.0974, 0.0891, 0.1088, 0.1191])
qwen_ece = np.array([0.3241, 0.3053, 0.2872, 0.2867, 0.2791, 0.2852, 0.2923])
mistral_ece = np.array([0.3520, 0.2651, 0.1884, 0.2072, 0.1697, 0.1895, 0.1906])
llama_auroc = np.array([0.6216, 0.6534, 0.6740, 0.6824, 0.6750, 0.6830, 0.6705])
qwen_auroc = np.array([0.6332, 0.6642, 0.6861, 0.6890, 0.6905, 0.6962, 0.6920])
mistral_auroc = np.array([0.5688, 0.6583, 0.6810, 0.6735, 0.6821, 0.6830, 0.6795])

# 创建两个子图
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

axes[0].plot(dataset_sizes, llama_ece, 'o-', label='LlaMa', linewidth=2)
axes[0].plot(dataset_sizes, qwen_ece, 's-', label='Qwen', linewidth=2)
axes[0].plot(dataset_sizes, mistral_ece, '^-', label='Mistral', linewidth=2)
axes[0].set_xlabel('Dataset Size')
axes[0].set_ylabel('ECE')
axes[0].grid(True)

axes[1].plot(dataset_sizes, llama_auroc, 'o-', label='LlaMa', linewidth=2)
axes[1].plot(dataset_sizes, qwen_auroc, 's-', label='Qwen', linewidth=2)
axes[1].plot(dataset_sizes, mistral_auroc, '^-', label='Ministral', linewidth=2)
axes[1].set_xlabel('Dataset Size')
axes[1].set_ylabel('AUROC')
axes[1].legend()
axes[1].grid(True)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('model_performance_vs_datasize.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()

# 输出表格数据
print("\nData Table: ECE (Expected Calibration Error) - Lower is better")
print("Dataset Size | LlaMa | Qwen | Mistral")
print("-------------|-------|------|--------")
for i, size in enumerate(dataset_sizes):
    print(f"{size:12d} | {llama_ece[i]:.4f} | {qwen_ece[i]:.4f} | {mistral_ece[i]:.4f}")

print("\nData Table: AUROC - Higher is better")
print("Dataset Size | LlaMa | Qwen | Mistral")
print("-------------|-------|------|--------")
for i, size in enumerate(dataset_sizes):
    print(f"{size:12d} | {llama_auroc[i]:.4f} | {qwen_auroc[i]:.4f} | {mistral_auroc[i]:.4f}")
