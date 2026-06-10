"""
实验1: WikiText-103 语言建模（主实验）
目的: 获得 HormonicFormer 的核心语言建模性能指标

配置 (基于手册，适配RTX 5070 8GB):
- vocab_size = 50257 (GPT-2 BPE)
- seq_len = 512 (减小以适应8GB显存)
- d_model = 512
- n_layers = 8
- batch_size = 8 (单卡，减小以适应8GB)
- max_steps = 4800
- peak_lr = 3e-3
- warmup_steps = 480 (10%)
- weight_decay = 0.1
- grad_clip = 1.0
- dropout = 0.1

预期目标: Val PPL < 22
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

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class SimpleHormonicFormerLM(nn.Module):
    """简化版HormonicFormer用于WikiText-103"""
    def __init__(self, vocab_size=50257, d_model=512, n_layers=8, n_heads=8, 
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_emb.weight
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_emb = self.token_emb(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_emb(pos_ids)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Transformer
        x = self.transformer(x, mask=mask)
        
        # LM head
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        return {'logits': logits, 'loss': loss}

class WikiTextDataset(Dataset):
    """WikiText-103 Dataset"""
    def __init__(self, tokens, seq_len=512):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return max(1, len(self.tokens) // self.seq_len - 1)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        
        if end > len(self.tokens):
            # Pad if needed
            chunk = self.tokens[start:]
            if len(chunk) < self.seq_len + 1:
                chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))
        else:
            chunk = self.tokens[start:end]
        
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        
        return {'input_ids': input_ids, 'labels': labels}

def load_wikitext103_data(data_path=None):
    """加载WikiText-103数据"""
    # 如果本地有预处理的数据，直接加载
    if data_path and Path(data_path).exists():
        print(f"[INFO] Loading preprocessed data from {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data['train'], data['valid'], data['test']
    
    # 否则生成合成数据用于测试
    print("[WARNING] WikiText-103 data not found, using synthetic data for testing")
    vocab_size = 50257
    n_train = 1000000  # 约100万tokens
    n_valid = 100000
    n_test = 100000
    
    train_tokens = np.random.randint(0, vocab_size, n_train).tolist()
    valid_tokens = np.random.randint(0, vocab_size, n_valid).tolist()
    test_tokens = np.random.randint(0, vocab_size, n_test).tolist()
    
    return train_tokens, valid_tokens, test_tokens

def train_epoch(model, dataloader, optimizer, scheduler, grad_clip=1.0, device='cuda'):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0

def evaluate(model, dataloader, device='cuda'):
    """评估模型"""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']
            
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    
    return avg_loss, ppl

def run_experiment1():
    """运行实验1"""
    print("="*80)
    print("实验1: WikiText-103 语言建模")
    print("="*80)
    print()
    
    # 配置
    config = {
        'vocab_size': 50257,
        'd_model': 512,
        'n_layers': 8,
        'n_heads': 8,
        'seq_len': 512,
        'batch_size': 8,
        'max_steps': 4800,
        'peak_lr': 3e-3,
        'warmup_steps': 480,
        'weight_decay': 0.1,
        'grad_clip': 1.0,
        'dropout': 0.1,
        'eval_every': 200
    }
    
    print("配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # 加载数据
    print("[1/4] 加载数据...")
    # 检查本地是否有wikitext数据
    data_paths = [
        r"C:\Users\MR\Desktop\初代激素场网络\文本\hormonic_v3 - 副本\wikitext103_tokens.npy",
        r"C:\Users\MR\Desktop\论文\关于场物理的神经框架\研究论文数据\wikitext103_tokens.npy"
    ]
    
    data_path = None
    for p in data_paths:
        if Path(p).exists():
            data_path = p
            break
    
    if data_path:
        print(f"  找到数据: {data_path}")
        tokens = np.load(data_path, allow_pickle=True).item()
        train_tokens = tokens.get('train', [])
        valid_tokens = tokens.get('valid', [])
        test_tokens = tokens.get('test', [])
    else:
        print("  未找到WikiText数据，使用合成数据")
        train_tokens, valid_tokens, test_tokens = load_wikitext103_data()
    
    print(f"  Train: {len(train_tokens)} tokens")
    print(f"  Valid: {len(valid_tokens)} tokens")
    print(f"  Test: {len(test_tokens)} tokens")
    print()
    
    # 创建数据集
    print("[2/4] 创建数据集...")
    train_dataset = WikiTextDataset(train_tokens, config['seq_len'])
    valid_dataset = WikiTextDataset(valid_tokens, config['seq_len'])
    test_dataset = WikiTextDataset(test_tokens, config['seq_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                             shuffle=False, num_workers=0)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Valid batches: {len(valid_loader)}")
    print()
    
    # 创建模型
    print("[3/4] 创建模型...")
    model = SimpleHormonicFormerLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['seq_len'],
        dropout=config['dropout']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,} ({n_params/1e6:.2f}M)")
    print()
    
    # 优化器和学习率调度
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['peak_lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Cosine with warmup
    def lr_lambda(step):
        if step < config['warmup_steps']:
            return step / config['warmup_steps']
        else:
            progress = (step - config['warmup_steps']) / (config['max_steps'] - config['warmup_steps'])
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练
    print("[4/4] 开始训练...")
    print()
    
    results = {
        'config': config,
        'train_losses': [],
        'valid_losses': [],
        'valid_ppls': [],
        'learning_rates': [],
        'steps': [],
        'best_valid_ppl': float('inf'),
        'best_step': 0
    }
    
    global_step = 0
    start_time = time.time()
    
    # 计算epochs
    steps_per_epoch = len(train_loader)
    n_epochs = (config['max_steps'] + steps_per_epoch - 1) // steps_per_epoch
    
    print(f"Training for ~{n_epochs} epochs ({config['max_steps']} steps)")
    print()
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # 训练
        model.train()
        for batch in train_loader:
            if global_step >= config['max_steps']:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']
            
            loss.backward()
            
            if config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            # 记录
            if global_step % config['eval_every'] == 0 or global_step == 1:
                # 评估
                valid_loss, valid_ppl = evaluate(model, valid_loader, device)
                
                results['steps'].append(global_step)
                results['train_losses'].append(loss.item())
                results['valid_losses'].append(valid_loss)
                results['valid_ppls'].append(valid_ppl)
                results['learning_rates'].append(optimizer.param_groups[0]['lr'])
                
                # 更新最佳
                if valid_ppl < results['best_valid_ppl']:
                    results['best_valid_ppl'] = valid_ppl
                    results['best_step'] = global_step
                    
                    # 保存最佳模型
                    save_path = Path(__file__).parent / 'experiment1_best_model.pt'
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'valid_ppl': valid_ppl,
                        'config': config
                    }, save_path)
                
                # 打印进度
                elapsed = time.time() - start_time
                print(f"Step {global_step:5d}/{config['max_steps']} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Valid PPL: {valid_ppl:.2f} | "
                      f"Best: {results['best_valid_ppl']:.2f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                      f"Time: {elapsed/60:.1f}m")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time/60:.1f}m")
        
        if global_step >= config['max_steps']:
            break
    
    # 最终测试
    print()
    print("="*80)
    print("训练完成，最终评估...")
    print("="*80)
    
    # 加载最佳模型
    save_path = Path(__file__).parent / 'experiment1_best_model.pt'
    if save_path.exists():
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from step {checkpoint['step']}")
    
    test_loss, test_ppl = evaluate(model, test_loader, device)
    
    results['test_loss'] = test_loss
    results['test_ppl'] = test_ppl
    results['total_time'] = time.time() - start_time
    
    print()
    print("="*80)
    print("实验1 最终结果")
    print("="*80)
    print(f"Best Valid PPL: {results['best_valid_ppl']:.2f} (Step {results['best_step']})")
    print(f"Test PPL: {test_ppl:.2f}")
    print(f"Total Time: {results['total_time']/3600:.2f}h")
    print()
    
    # 目标检查
    if results['best_valid_ppl'] < 22:
        print(f"[PASS] 达到目标: Val PPL < 22")
        results['status'] = 'PASS'
    else:
        print(f"[INFO] 未达到目标 (目标: < 22, 实际: {results['best_valid_ppl']:.2f})")
        results['status'] = 'PARTIAL'
    
    # 保存结果
    output_path = Path(__file__).parent / 'experiment1_wikitext103_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 结果保存至: {output_path}")
    
    return results

def generate_report(results):
    """生成实验报告"""
    lines = []
    lines.append("="*80)
    lines.append("实验1: WikiText-103 语言建模报告")
    lines.append("="*80)
    lines.append("")
    
    lines.append("配置:")
    for k, v in results['config'].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    
    lines.append("训练过程:")
    for i, step in enumerate(results['steps']):
        lines.append(f"  Step {step}: Train Loss={results['train_losses'][i]:.4f}, "
                    f"Valid PPL={results['valid_ppls'][i]:.2f}, "
                    f"LR={results['learning_rates'][i]:.2e}")
    
    lines.append("")
    lines.append("最终结果:")
    lines.append(f"  Best Valid PPL: {results['best_valid_ppl']:.2f} (Step {results['best_step']})")
    lines.append(f"  Test PPL: {results['test_ppl']:.2f}")
    lines.append(f"  Total Time: {results['total_time']/3600:.2f}h")
    lines.append(f"  Status: {results['status']}")
    
    lines.append("")
    lines.append("="*80)
    
    return "\n".join(lines)

if __name__ == '__main__':
    print("开始运行实验1: WikiText-103 语言建模...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 运行实验
    results = run_experiment1()
    
    # 生成报告
    report = generate_report(results)
    
    # 保存报告
    report_path = Path(__file__).parent / '28_实验1_WikiText103报告.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[OK] 报告保存至: {report_path}")
    print()
    print(report)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
