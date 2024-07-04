import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import json

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