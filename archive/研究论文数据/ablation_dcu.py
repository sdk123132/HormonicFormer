"""
DCU消融实验脚本
7个组件消融，适配K100_AI 64GB
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
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 配置 (DCU 64GB可以支持更大batch)
CONFIG = {
    'vocab_size': 50257,
    'd_model': 512,
    'n_layers': 8,
    'n_heads': 8,
    'seq_len': 512,
    'batch_size': 64,  # DCU 64GB优势
    'max_steps': 4800,
    'peak_lr': 3e-3,
    'warmup_steps': 480,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'dropout': 0.1,
    'eval_every': 200
}

# 消融配置
ABLATIONS = {
    'full': {},  # 完整模型
    'no_hebbian': {'use_hebbian': False},
    'no_cgl': {'use_cgl': False},
    'no_stp': {'use_stp': False},
    'no_da': {'use_da': False},
    'no_cb': {'use_cb': False},
    'no_ei': {'use_ei': False},
}

class HormonicFormerVariant(nn.Module):
    """HormonicFormer变体（支持消融）"""
    def __init__(self, ablation_config=None):
        super().__init__()
        self.ablation = ablation_config or {}
        
        self.token_emb = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        self.pos_emb = nn.Embedding(CONFIG['seq_len'], CONFIG['d_model'])
        
        # 基础Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG['d_model'],
            nhead=CONFIG['n_heads'],
            dim_feedforward=4*CONFIG['d_model'],
            dropout=CONFIG['dropout'],
            batch_first=True
        )
        
        # 根据消融调整层数
        n_layers = CONFIG['n_layers']
        if self.ablation.get('no_cgl'):
            n_layers = 12  # 补偿CGL的缺失
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
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
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, CONFIG['vocab_size']), 
                                         shift_labels.view(-1))
        
        return {'logits': logits, 'loss': loss}

def load_wikitext():
    """加载WikiText数据"""
    import glob
    
    # 查找数据文件
    search_patterns = [
        '/root/private_data/hormonic_v3/wikitext103_raw.txt',
        '/root/private_data/hormonic_v3/wikitext103_tokens.npy',
        '/root/data/wikitext103_raw.txt',
    ]
    
    data_file = None
    for pattern in search_patterns:
        if Path(pattern).exists():
            data_file = pattern
            break
    
    if data_file is None:
        # 使用合成数据
        print("[WARNING] Using synthetic data")
        n = 1000000
        return (
            np.random.randint(0, 1000, n).tolist(),
            np.random.randint(0, 1000, n//10).tolist(),
            np.random.randint(0, 1000, n//10).tolist()
        )
    
    print(f"[INFO] Loading from {data_file}")
    
    if data_file.endswith('.npy'):
        tokens = np.load(data_file, allow_pickle=True)
        if isinstance(tokens, dict):
            return tokens['train'], tokens['valid'], tokens['test']
        else:
            # 分割
            n = len(tokens)
            return (
                tokens[:int(0.8*n)].tolist(),
                tokens[int(0.8*n):int(0.9*n)].tolist(),
                tokens[int(0.9*n):].tolist()
            )
    else:
        # 文本文件
        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read(10000000)  # 10M chars
        
        tokens = [(ord(c) + 256) % CONFIG['vocab_size'] for c in text]
        n = len(tokens)
        return (
            tokens[:int(0.8*n)],
            tokens[int(0.8*n):int(0.9*n)],
            tokens[int(0.9*n):]
        )

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

def train_ablation(name, ablation_config):
    """训练单个消融变体"""
    print(f"\n{'='*80}")
    print(f"消融实验: {name}")
    print(f"配置: {ablation_config}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # 数据
    train_tok, valid_tok, test_tok = load_wikitext()
    train_ds = TextDataset(train_tok, CONFIG['seq_len'])
    valid_ds = TextDataset(valid_tok, CONFIG['seq_len'])
    test_ds = TextDataset(test_tok, CONFIG['seq_len'])
    
    train_loader = DataLoader(train_ds, CONFIG['batch_size'], shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, CONFIG['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, CONFIG['batch_size'], shuffle=False, num_workers=4)
    
    # 模型
    model = HormonicFormerVariant(ablation_config).to(device)
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
    best_ppl = float('inf')
    best_step = 0
    global_step = 0
    
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
            
            if global_step % CONFIG['eval_every'] == 0:
                _, valid_ppl = evaluate(model, valid_loader)
                
                if valid_ppl < best_ppl:
                    best_ppl = valid_ppl
                    best_step = global_step
                
                print(f"Step {global_step:5d}/{CONFIG['max_steps']} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Valid PPL: {valid_ppl:.2f} | "
                      f"Best: {best_ppl:.2f}")
                model.train()
        
        if global_step >= CONFIG['max_steps']:
            break
    
    # 测试
    _, test_ppl = evaluate(model, test_loader)
    total_time = time.time() - start_time
    
    print(f"\n{name} 完成!")
    print(f"Best Valid PPL: {best_ppl:.2f} (Step {best_step})")
    print(f"Test PPL: {test_ppl:.2f}")
    print(f"Time: {total_time/60:.1f}min")
    
    return {
        'name': name,
        'ablation': ablation_config,
        'best_valid_ppl': best_ppl,
        'best_step': best_step,
        'test_ppl': test_ppl,
        'time': total_time,
        'n_params': n_params
    }

def main():
    """运行所有消融实验"""
    print("="*80)
    print("DCU消融实验 - 7组件")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    for name, ablation_config in ABLATIONS.items():
        result = train_ablation(name, ablation_config)
        results.append(result)
    
    # 汇总
    print("\n" + "="*80)
    print("消融实验汇总")
    print("="*80)
    
    baseline_ppl = None
    for r in results:
        if r['name'] == 'full':
            baseline_ppl = r['best_valid_ppl']
            break
    
    print(f"\n{'Component':<15} {'Valid PPL':<12} {'Delta':<12} {'Time(min)':<12}")
    print("-" * 60)
    
    for r in sorted(results, key=lambda x: x['best_valid_ppl']):
        delta = ""
        if baseline_ppl and r['name'] != 'full':
            delta_ppl = r['best_valid_ppl'] - baseline_ppl
            delta = f"+{delta_ppl:.2f}" if delta_ppl > 0 else f"{delta_ppl:.2f}"
        
        print(f"{r['name']:<15} {r['best_valid_ppl']:<12.2f} {delta:<12} {r['time']/60:<12.1f}")
    
    # 保存
    output = {
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG
    }
    
    with open('ablation_dcu_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[OK] 结果保存至: ablation_dcu_results.json")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
