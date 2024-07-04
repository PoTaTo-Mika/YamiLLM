import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import json

class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(SimpleLanguageModel, self).__init__()
        
        # Embedding��
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # λ�ñ��� (����ʹ�ü򵥵Ŀ�ѧϰλ�ñ���)
        self.pos_encoder = nn.Embedding(1000, d_model)  # ����������г���Ϊ1000
        
        # Transformer��
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ���Բ�
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, src_key_padding_mask=None):
        # x����״: (batch_size, seq_len)
        
        # Ӧ�ô�Ƕ��
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        
        # ���λ�ñ���
        positions = torch.arange(x.size(1)).unsqueeze(0).to(x.device)
        x = x + self.pos_encoder(positions)
        
        # Ӧ��Transformer
        x = x.permute(1, 0, 2)  # Transformer������������״: (seq_len, batch_size, d_model)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = x.permute(1, 0, 2)  # ��� (batch_size, seq_len, d_model)
        
        # Ӧ�����Բ�
        x = self.fc_out(x)
        
        return x