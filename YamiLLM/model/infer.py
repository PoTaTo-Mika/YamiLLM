import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import json
from model import SimpleLanguageModel

with open('D:/PythonProject/YamiLLM/checkpoints/vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

tokenizer = PreTrainedTokenizerFast.from_pretrained("D:/PythonProject/YamiLLM/checkpoints")

# 设置模型参数
vocab_size = len(vocab)
d_model = 256
nhead = 4
num_layers = 2

# 初始化模型
model = SimpleLanguageModel(vocab_size, d_model, nhead, num_layers)

# 加载训练好的模型权重
model.load_state_dict(torch.load('D:/PythonProject/YamiLLM/checkpoints/model.pth'))
model.eval()  # 设置为评估模式

# 将模型移到可用设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 文本生成函数
def generate_text(model, tokenizer, start_text, max_length=256):
    model.eval()
    input_ids = tokenizer.encode(start_text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# 使用模型生成文本
start_text = "显然，狐人这种"
generated_text = generate_text(model, tokenizer, start_text)
print(f"输入: {start_text}")
print(f"生成的文本: {generated_text}")