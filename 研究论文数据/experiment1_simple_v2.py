"""
实验1: WikiText-103 语言建模 (简化稳定版)
使用本地已下载的数据
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
import math

# 设置
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 配置
CONFIG = {
    'vocab_size': 50257,
    'd_model': 128,      # 进一步减小
    'n_layers': 4,       # 减少层数
    'n_heads': 4,
    'seq_len': 128,
    'batch_size': 8,
    'max_steps': 300,    # 快速测试
    'peak_lr': 1e-3,
    'warmup_steps': 30,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'dropout': 0.1,
    'eval_every': 50
}

class SimpleLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        self.pos_emb = nn.Embedding(CONFIG['seq_len'], CONFIG['d_model'])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG['d_model'],
            nhead=CONFIG['n_heads'],
            dim_feedforward=4*CONFIG['d_model'],
            dropout=CONFIG['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=CONFIG['n_layers'])
        self.lm_head = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'], bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.dropout = nn.Dropout(CONFIG['dropout'])
        
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1))
        x = self.dropout(tok_emb + pos_emb)
        
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, CONFIG['vocab_size']), shift_labels.view(-1))
        
        return {'logits': logits, 'loss': loss}

def load_data_from_file():
    """从本地文件加载数据"""
    # 检查是否有wikitext文件
    wikitext_path = r'C:\Users\MR\Desktop\初代激素场网络\文本\hormonic_v3 - 副本\wikitext103_raw.txt'
    
    if Path(wikitext_path).exists():
        print(f"[INFO] Loading from {wikitext_path}")
        with open(wikitext_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read(1000000)  # 只读前1MB
        
        print(f"[INFO] Loaded {len(text)} characters")
        
        # 简单的字符编码
        char_to_id = {c: i+100 for i, c in enumerate(set(text))}
        tokens = [char_to_id.get(c, 0) for c in text[:50000]]
        
        # 限制vocab大小
        tokens = [min(t, CONFIG['vocab_size']-1) for t in tokens]
        
        print(f"[INFO] Tokenized to {len(tokens)} tokens")
    else:
        print("[INFO] File not found, using synthetic data")
        tokens = np.random.randint(0, 1000, 50000).tolist()
    
    # 分割
    n = len(tokens)
    train = tokens[:int(0.8*n)]
    valid = tokens[int(0.8*n):int(0.9*n)]
    test = tokens[int(0.9*n):]
    
    return train, valid, test

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return max(1, len(self.tokens) // self.seq_len - 1)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))
        return {
            'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
            'labels': torch.tensor(chunk[1:], dtype=torch.long)
        }

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch['input_ids'].to(device), batch['labels'].to(device))
            if out['loss'] is not None:
                total_loss += out['loss'].item()
                n += 1
    avg_loss = total_loss / n if n > 0 else 0
    ppl = math.exp(min(avg_loss, 10))
    return avg_loss, ppl

def train():
    print("="*60)
    print("实验1: WikiText 语言建模 (简化版)")
    print("="*60)
    print(f"Config: {CONFIG}")
    print()
    
    # 数据
    train_tok, valid_tok, test_tok = load_data_from_file()
    print(f"Train: {len(train_tok)}, Valid: {len(valid_tok)}, Test: {len(test_tok)}")
    
    train_ds = TextDataset(train_tok, CONFIG['seq_len'])
    valid_ds = TextDataset(valid_tok, CONFIG['seq_len'])
    test_ds = TextDataset(test_tok, CONFIG['seq_len'])
    
    train_loader = DataLoader(train_ds, CONFIG['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_ds, CONFIG['batch_size'])
    test_loader = DataLoader(test_ds, CONFIG['batch_size'])
    
    print(f"Train batches: {len(train_loader)}, Valid: {len(valid_loader)}")
    print()
    
    # 模型
    model = SimpleLM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M")
    print()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['peak_lr'], 
                           weight_decay=CONFIG['weight_decay'])
    
    def lr_lambda(step):
        if step < CONFIG['warmup_steps']:
            return step / CONFIG['warmup_steps']
        progress = (step - CONFIG['warmup_steps']) / (CONFIG['max_steps'] - CONFIG['warmup_steps'])
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练
    results = {'steps': [], 'train_loss': [], 'valid_ppl': [], 'best_ppl': float('inf'), 'best_step': 0}
    start_time = time.time()
    global_step = 0
    
    print("开始训练...")
    print()
    model.train()
    
    for epoch in range(100):
        for batch in train_loader:
            if global_step >= CONFIG['max_steps']:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            out = model(input_ids, labels)
            loss = out['loss']
            
            if loss is None:
                continue
            
            loss.backward()
            
            if CONFIG['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            # 评估
            if global_step % CONFIG['eval_every'] == 0:
                _, valid_ppl = evaluate(model, valid_loader)
                results['steps'].append(global_step)
                results['train_loss'].append(loss.item())
                results['valid_ppl'].append(valid_ppl)
                
                if valid_ppl < results['best_ppl']:
                    results['best_ppl'] = valid_ppl
                    results['best_step'] = global_step
                    torch.save(model.state_dict(), 'experiment1_best_v2.pt')
                
                elapsed = (time.time() - start_time) / 60
                print(f"Step {global_step:4d}/{CONFIG['max_steps']} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Valid PPL: {valid_ppl:.2f} | "
                      f"Best: {results['best_ppl']:.2f} | "
                      f"Time: {elapsed:.1f}m")
                model.train()
        
        if global_step >= CONFIG['max_steps']:
            break
    
    # 最终测试
    if Path('experiment1_best_v2.pt').exists():
        model.load_state_dict(torch.load('experiment1_best_v2.pt'))
    _, test_ppl = evaluate(model, test_loader)
    
    total_time = (time.time() - start_time) / 3600
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"Best Valid PPL: {results['best_ppl']:.2f} (Step {results['best_step']})")
    print(f"Test PPL: {test_ppl:.2f}")
    print(f"Total Time: {total_time:.2f}h")
    print("="*60)
    
    # 保存
    results['test_ppl'] = test_ppl
    results['total_time'] = total_time
    results['config'] = CONFIG
    
    with open('experiment1_results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到 experiment1_results_v2.json")

if __name__ == '__main__':
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    train()
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
