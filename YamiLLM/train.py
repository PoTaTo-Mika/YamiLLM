import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model.llama_non_parallel import Transformer, ModelArgs
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import logging
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# 使用已有的tokenizer
model_name = "PoTaTo721/YamiLLM"
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir="checkpoints")

class MultiFileTextDataset(Dataset):
    def __init__(self, directory, tokenizer, max_length):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._load_data_from_directory(directory)

    def _load_data_from_directory(self, directory):
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.txt'):
                self._load_data_from_file(file_path)

    def _load_data_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(line.strip())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'start_pos': int(0) 
        }

# 训练函数
def train(model, train_loader, optimizer, scheduler, device, gradient_accumulation_steps):
    model.train()
    total_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        input_ids = batch['input_ids'].to(device)
        start_pos = batch['start_pos'].to(device)
        
        # 移除 @torch.inference_mode() 装饰器的影响
        with torch.set_grad_enabled(True):
            outputs = model(input_ids, start_pos=start_pos)
        
        # 计算损失（假设使用交叉熵损失）
        loss = nn.CrossEntropyLoss()(outputs.view(-1, model.vocab_size), input_ids.view(-1))
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


# 评估函数
def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            start_pos = batch['start_pos'][0].item()
            
            outputs = model(input_ids, start_pos=start_pos)
            
            labels = input_ids[:, 1:].contiguous()
            logits = outputs[:, :-1, :].contiguous()
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(eval_loader)


def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 设置超参数
    batch_size = 2
    gradient_accumulation_steps = 2
    num_epochs = 10
    learning_rate = 5e-5
    warmup_steps = 10
    max_length = 512
    save_steps = 2

    # 初始化模型
    model_args = ModelArgs(
        dim=4096,
        n_layers=2,
        n_heads=4,
        vocab_size=50000, 
        multiple_of=256,
        max_seq_len=max_length,
    )
    model = Transformer(model_args)
    
    # 加载数据
    train_dataset = MultiFileTextDataset(r'data/train', tokenizer, max_length)
    eval_dataset = MultiFileTextDataset(r'data/eval', tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=4)
    
    # 设置优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_loader) * num_epochs)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 初始化 GradScaler
    scaler = GradScaler()
    
    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            start_pos = batch['start_pos'][0].item()
        
            with autocast():
                outputs = model(input_ids, start_pos=start_pos)
                labels = input_ids[:, 1:].contiguous()
                logits = outputs[:, :-1, :].contiguous()
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            scaler.scale(loss).backward(retain_graph=True)
        
            if (global_step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        
            global_step += 1
            total_loss += loss.item()
        
            # 保存检查点
            if global_step % save_steps == 0:
                checkpoint = {
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': loss.item(),
                }
                torch.save(checkpoint, f'checkpoint_step_{global_step}.pt')
                logger.info(f'Saved checkpoint at step {global_step}')
    
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        
        # 评估模型
        eval_loss = evaluate(model, eval_loader, device)
    
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {avg_loss:.4f}')
        logger.info(f'Eval Loss: {eval_loss:.4f}')

        # 保存最佳模型
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            logger.info(f'New best model saved with eval loss: {best_eval_loss:.4f}')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)