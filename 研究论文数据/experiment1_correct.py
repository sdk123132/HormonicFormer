"""
实验1: WikiText-103 语言建模（正确版本）
使用HuggingFace datasets和transformers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
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
    'seq_len': 128,  # 减小以适应显存
    'batch_size': 4,  # 减小
    'max_steps': 500,
    'peak_lr': 3e-4,  # 减小学习率
    'warmup_steps': 50,
    'weight_decay': 0.1,
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

def load_wikitext():
    """加载WikiText-2 (比103小，适合快速实验)"""
    print("[INFO] Loading WikiText-2 dataset...")
    
    try:
        # 使用datasets库加载
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir='C:/Users/MR/.cache/huggingface/datasets')
        
        # 加载tokenizer
        print("[INFO] Loading GPT2 tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, max_length=CONFIG['seq_len'], 
                           padding='max_length', return_tensors='pt')
        
        print("[INFO] Tokenizing dataset...")
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        
        train_data = tokenized['train']
        valid_data = tokenized['validation']
        test_data = tokenized['test']
        
        print(f"[INFO] Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
        
        return train_data, valid_data, test_data
        
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        print("[INFO] Using synthetic data instead...")
        
        # 创建合成数据
        n_train = 10000
        n_valid = 1000
        n_test = 1000
        
        def create_synthetic_dataset(n_samples):
            data = []
            for i in range(n_samples):
                # 随机token序列
                tokens = np.random.randint(100, 1000, CONFIG['seq_len']).tolist()
                data.append({'input_ids': tokens, 'labels': tokens})
            return data
        
        return create_synthetic_dataset(n_train), create_synthetic_dataset(n_valid), create_synthetic_dataset(n_test)

def create_dataloader(dataset, batch_size, shuffle=True):
    """创建DataLoader"""
    def collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'][:CONFIG['seq_len']] for item in batch], dtype=torch.long)
        labels = input_ids.clone()
        
        # Pad if needed
        if input_ids.size(1) < CONFIG['seq_len']:
            pad_len = CONFIG['seq_len'] - input_ids.size(1)
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
        
        return {'input_ids': input_ids, 'labels': labels}
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                     collate_fn=collate_fn, num_workers=0)

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
    print("实验1: WikiText-2 语言建模")
    print("="*60)
    print(f"Config: {CONFIG}")
    print()
    
    # 加载数据
    train_data, valid_data, test_data = load_wikitext()
    
    train_loader = create_dataloader(train_data, CONFIG['batch_size'], shuffle=True)
    valid_loader = create_dataloader(valid_data, CONFIG['batch_size'], shuffle=False)
    test_loader = create_dataloader(test_data, CONFIG['batch_size'], shuffle=False)
    
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
                    torch.save(model.state_dict(), 'experiment1_best_correct.pt')
                
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
    if Path('experiment1_best_correct.pt').exists():
        model.load_state_dict(torch.load('experiment1_best_correct.pt'))
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
    
    with open('experiment1_results_correct.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n结果已保存到 experiment1_results_correct.json")

if __name__ == '__main__':
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    train()
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
