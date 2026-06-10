"""
实验1: WikiText-103 语言建模训练脚本
适配RTX 5070 8GB显存
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
import sys

# 配置
SEED = 42
VOCAB_SIZE = 50257
D_MODEL = 512
N_LAYERS = 8
N_HEADS = 8
SEQ_LEN = 512
BATCH_SIZE = 4  # 8GB显存限制
MAX_STEPS = 4800
PEAK_LR = 3e-3
WARMUP_STEPS = 480
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
DROPOUT = 0.1
EVAL_EVERY = 200

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class HormonicFormerLM(nn.Module):
    """简化版HormonicFormer"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Tie weights
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        token_emb = self.token_emb(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_emb(pos_ids)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.vocab_size), 
                                         shift_labels.view(-1))
        
        return {'logits': logits, 'loss': loss}

def load_data():
    """加载WikiText-103数据"""
    print("[INFO] Loading WikiText-103...")
    
    # 使用本地缓存
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', cache_dir='C:/Users/MR/.cache/huggingface/datasets')
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=SEQ_LEN)
    
    print("[INFO] Tokenizing...")
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    return tokenized['train'], tokenized['validation'], tokenized['test'], tokenizer

def create_dataloader(dataset, batch_size, shuffle=True):
    """创建DataLoader"""
    def collate_fn(batch):
        input_ids = torch.tensor([item['input_ids'][:SEQ_LEN] for item in batch], dtype=torch.long)
        # Pad if needed
        if input_ids.size(1) < SEQ_LEN:
            pad_len = SEQ_LEN - input_ids.size(1)
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
        return {'input_ids': input_ids, 'labels': input_ids.clone()}
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      collate_fn=collate_fn, num_workers=0)

def train():
    """主训练函数"""
    print("="*80)
    print("实验1: WikiText-103 训练")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载数据
    train_data, val_data, test_data, tokenizer = load_data()
    
    train_loader = create_dataloader(train_data, BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_data, BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(test_data, BATCH_SIZE, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()
    
    # 创建模型
    model = HormonicFormerLM(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params/1e6:.2f}M)")
    print()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练
    results = {
        'config': {
            'vocab_size': VOCAB_SIZE,
            'd_model': D_MODEL,
            'n_layers': N_LAYERS,
            'n_heads': N_HEADS,
            'seq_len': SEQ_LEN,
            'batch_size': BATCH_SIZE,
            'max_steps': MAX_STEPS,
            'peak_lr': PEAK_LR,
            'warmup_steps': WARMUP_STEPS,
            'weight_decay': WEIGHT_DECAY,
            'grad_clip': GRAD_CLIP,
            'dropout': DROPOUT
        },
        'train_losses': [],
        'valid_ppls': [],
        'steps': [],
        'best_valid_ppl': float('inf'),
        'best_step': 0
    }
    
    global_step = 0
    start_time = time.time()
    
    print("Starting training...")
    print()
    
    model.train()
    for epoch in range(100):  # 足够多的epochs
        for batch in train_loader:
            if global_step >= MAX_STEPS:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']
            
            loss.backward()
            
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            # 评估
            if global_step % EVAL_EVERY == 0 or global_step == 1:
                model.eval()
                val_loss = 0
                n_val = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input = val_batch['input_ids'].to(device)
                        val_label = val_batch['labels'].to(device)
                        val_out = model(val_input, labels=val_label)
                        val_loss += val_out['loss'].item()
                        n_val += 1
                
                avg_val_loss = val_loss / n_val if n_val > 0 else 0
                val_ppl = np.exp(avg_val_loss) if avg_val_loss < 10 else float('inf')
                
                results['steps'].append(global_step)
                results['train_losses'].append(loss.item())
                results['valid_ppls'].append(val_ppl)
                
                if val_ppl < results['best_valid_ppl']:
                    results['best_valid_ppl'] = val_ppl
                    results['best_step'] =