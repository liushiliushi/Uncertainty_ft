from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# 设置模型ID
model_id = "liushiliushi/llama-7b-uncertainty-brier"

print(f"开始加载模型: {model_id}")

# 加载PEFT配置
print("加载PEFT配置...")
peft_config = PeftConfig.from_pretrained(model_id)
print(f"基础模型路径: {peft_config.base_model_name_or_path}")

# 加载tokenizer
print("加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("tokenizer加载成功")

# 加载基础模型
print("加载基础模型...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float16,
        load_in_8bit=False,  # 避免使用device_map="auto"
        device_map=None,  # 避免使用device_map
        trust_remote_code=True,
    )
    print("基础模型加载成功")
except Exception as e:
    print(f"基础模型加载失败: {e}")
    print("尝试直接加载适配器模型...")
    # 如果无法访问基础模型，直接加载 PEFT 模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=None,  # 避免使用device_map
            trust_remote_code=True,
        )
        print("PEFT模型直接加载成功!")
        
        # 简单测试生成
        prompt = "What is the capital of France? Express your uncertainty."
        print(f"\n测试生成，输入：'{prompt}'")

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            model = model.to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n生成结果:")
        print(response)
        exit(0)
    except Exception as e:
        print(f"PEFT模型直接加载失败: {e}")
        exit(1)

# 如果基础模型加载成功，继续加载PEFT模型
print("加载PEFT适配器...")
try:
    model = PeftModel.from_pretrained(base_model, model_id)
    print("PEFT模型加载成功!")
except Exception as e:
    print(f"PEFT模型加载失败: {e}")
    exit(1)

# 简单测试生成
prompt = "What is the capital of France? Express your uncertainty."
print(f"\n测试生成，输入：'{prompt}'")

inputs = tokenizer(prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = inputs.to("cuda")
    model = model.to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n生成结果:")
print(response) 