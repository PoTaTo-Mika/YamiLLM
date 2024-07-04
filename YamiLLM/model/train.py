import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from transformers import PreTrainedTokenizerFast
import json

# 加载tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("D:/PythonProject/YamiLLM/checkpoints")

# 使用tokenizer进行编码
def encode(text):
    return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)

# 在Dataset类中使用
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        encoding = self.tokenizer(line, max_length=self.max_length, truncation=True, return_tensors='pt')
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)

def collate_batch(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks

# 创建dataset实例
max_length = 256  # 或其他适合你数据的长度
dataset = TextDataset('D:/PythonProject/YamiLLM/data/data.txt',tokenizer, max_length)

class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(SimpleLanguageModel, self).__init__()
        
        # Embedding层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码 (这里使用简单的可学习位置编码)
        self.pos_encoder = nn.Embedding(1000, d_model)  # 假设最大序列长度为1000
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 线性层
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, src_key_padding_mask=None):
        # x的形状: (batch_size, seq_len)
        
        # 应用词嵌入
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        positions = torch.arange(x.size(1)).unsqueeze(0).to(x.device)
        x = x + self.pos_encoder(positions)
        
        # 应用Transformer
        x = x.permute(1, 0, 2)  # Transformer期望的输入形状: (seq_len, batch_size, d_model)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = x.permute(1, 0, 2)  # 变回 (batch_size, seq_len, d_model)
        
        # 应用线性层
        x = self.fc_out(x)
        
        return x
    
with open('D:/PythonProject/YamiLLM/checkpoints/vocab.json', 'r',encoding ='utf-8') as f:
    vocab = json.load(f)

vocab_size = len(vocab)

d_model = 256  # 嵌入维度
nhead = 4      # 注意力头数
num_layers = 2 # Transformer层数

model = SimpleLanguageModel(vocab_size, d_model, nhead, num_layers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 准备数据
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

# 训练循环
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_ids, attention_mask in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # 准备输入和目标
        input_seq = input_ids[:, :-1]
        target_seq = input_ids[:, 1:]
        mask = attention_mask[:, :-1]
        
        # 前向传播
        output = model(input_seq, src_key_padding_mask=~mask.bool())
        
        # 计算损失
        output = output.contiguous().view(-1, vocab_size)
        target_seq = target_seq.contiguous().view(-1)
        
        loss = criterion(output, target_seq)
        
        # 应用掩码
        mask = mask.contiguous().view(-1).float()
        loss = (loss * mask).sum() / mask.sum()
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 打印每个epoch的平均损失
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# 保存模型
path = 'checkpoints'    
torch.save(model.state_dict(), f'{path}/model.pth')
print(f"Model successfully saved to {path}")
