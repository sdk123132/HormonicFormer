"""
实验1: WikiText-103 语言建模（使用真实数据）
"""

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"[INFO] Device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 配置 (适配8GB显存)
CONFIG = {
    'vocab_size': 50257,
    'd_model': 256,
    'n_layers': 6,
    'n_heads': 8,
    'seq_len': 256,
    'batch_size': 8,
    'max_steps': 1200,  # 减少步数，使用真实数据
    'peak_lr': 3e-3,
    'warmup_steps': 120,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'dropout': 0.1,
    'eval_every': 100
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

def load_wikitext_from_file(file_path, tokenizer_func=None):
    """从文本文件加载WikiText数据"""
    print(f"[INFO] Loading from {file_path}")
    
    if not Path(file_path).exists():
        print(f"[WARNING] File not found: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"[INFO] Loaded {len(text)} characters")
    
    # 简单的字符级tokenization (fallback)
    # 实际应该使用GPT-2 tokenizer
    tokens = []
    for char in text[:100000]:  # 限制前10万字符
        tokens.append(ord(char) % CONFIG['vocab_size'])
    
    return tokens

def load_data():
    """尝试加载真实WikiText数据"""
    # 可能的文件路径
    possible_paths = [
        r'C:\Users\MR\.cache\huggingface\datasets\wikitext\wikitext-103-raw-v1\wikitext-103-train.txt',
        r'C:\Users\MR\.cache\huggingface\datasets\wikitext\wikitext-103-raw-v1\train.txt',
        r'C:\Users\MR\Desktop\初代激素场网络\文本\hormonic_v3 - 副本\wikitext103_raw.txt',
    ]
    
    train_tokens = None
    for path in possible_paths:
        if Path(path).exists():
            train_tokens = load_wikitext_from_file(path)
            if train_tokens:
                break
    
    if train_tokens is None:
        print("[WARNING] Using synthetic data")
        vocab_size = CONFIG['vocab_size']
        train_tokens = np.random.randint(0, vocab_size, 200000).tolist()
    
    # 分割
    n = len(train_tokens)
    train = train_tokens[:int(0.8*n)]
    valid = train_tokens[int(0.8*n):int(0.9*n)]
    test = train_tokens[int(0.9*n):]
    
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
            total_loss += out['loss'].item()
            n += 1
    avg_loss = total_loss / n if n > 0 else 0
    return avg_loss, math.exp(min(avg_loss, 10))

def train():
    print("="*60)
    print("实验1: WikiText-103 训练 (真实数据)")
    print("="*60)
    print(f"Config: {CONFIG}")
    print()
    
    # 数据
    train_tok, valid_tok, test_tok = load_data()
    print(f"Train: {len(train_tok)} tokens, Valid: {len(valid_tok)}, Test: {len(test_tok)}")
    
    train_ds = TextDataset(train_tok, CONFIG['seq_len'])
    valid_ds = TextDataset(valid_tok, CONFIG['seq_len'])
    test_ds = TextDataset(test_tok, CONFIG['seq_len'])
    
    train_loader = DataLoader(train_ds, CONFIG['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_ds, CONFIG['batch_size'])
    test_loader = DataLoader(test_ds, CONFIG['batch_size'])
    
    print(f"Train batches: {len(train_loader)}, Valid: {len(valid_loader)}")
    
    # 模型
    model = SimpleLM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M")
    
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
    
    print("\n开始训练...")
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
                    torch.save(model.state_dict(), 'experiment1_best_real.pt')
                
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
    if Path('experiment1_best_real.pt').exists():
        model.load_state_dict(torch.load('experiment1_best_real.pt'))
    _, test_ppl = evaluate(model, test_loader)
    
    total_time = (time.time() - start_time) / 3600
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"Best Valid PPL: {results['best_ppl']:.2f} (Step {results['best_step']})")
    print(f"Test PPL: {test_ppl:.2f}")
    print(f"Total Time: {total_time:.2f}h")
    print("="*60)
    
    # 保存结果
    results['test_ppl'] = test_ppl
    results['total_time'] = total_time
    results['config'] = CONFIG
    
    with open('experiment1_results_real.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到 experiment1_results_real.json")

if __name__ == '__main__':
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    train()
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
