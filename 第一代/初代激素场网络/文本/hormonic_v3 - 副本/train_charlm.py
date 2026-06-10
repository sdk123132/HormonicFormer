"""
HormonicCharLM - Character-level Language Model
基于 Copy Task Wrapper 模式
"""
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hormonicformer_v3 import HormonicFormer
import urllib.request
import os


class CharDataset(Dataset):
    """字符级数据集"""
    def __init__(self, text, seq_len, train=True, train_split=0.9):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        n = int(train_split * len(data))
        self.data = data[:n] if train else data[n:]
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


class HormonicCharLM(nn.Module):
    """
    字符级语言模型 Wrapper
    序列 → 嵌入 → 伪二维图 → HormonicFormer → 序列预测
    """
    def __init__(self, config, vocab_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        d_model = config['model']['d_model']
        
        # 字符嵌入
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 调整配置：使用 1x1 patch 保持序列长度
        config['model']['seq_len'] = seq_len
        config['model']['patch_size'] = 1
        
        # HormonicFormer
        self.model = HormonicFormer(config)
        
        # 替换分类头
        self.model.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, vocab_size)
        )
    
    def forward(self, input_ids, targets=None):
        """
        input_ids: [B, S]
        targets: [B, S] (optional)
        返回: logits [B, S, vocab_size], loss (optional)
        """
        B, S = input_ids.shape
        
        # 嵌入
        x = self.embed(input_ids)  # [B, S, d_model]
        
        # 构造伪二维输入 [B, 1, 1, S]（和 Copy Task 相同）
        # 将嵌入维度作为通道，序列作为宽度
        img = x.transpose(1, 2).unsqueeze(2)  # [B, d_model, 1, S]
        
        # 通过 patch_embed（kernel=1, stride=1 保持长度）
        x = self.model.patch_embed(img)  # [B, d_model, 1, S]
        x = x.flatten(2).transpose(1, 2)  # [B, S, d_model]
        
        # 通过 blocks
        x_embed = x.detach().clone()
        aux_loss = 0
        for i, block in enumerate(self.model.blocks):
            x = block(x, x_embed=x_embed if block.use_feedback else None)
        
        # 分类
        logits = self.model.classifier(x)  # [B, S, vocab_size]
        
        if targets is not None:
            # 移位预测：用 [0:S-1] 预测 [1:S]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_targets.view(-1)
            )
            return logits, loss
        
        return logits


def download_tiny_shakespeare(path='./data/tinyshakespeare.txt'):
    """下载 Tiny Shakespeare"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f'Downloading Tiny Shakespeare...')
    urllib.request.urlretrieve(url, path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    correct = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        logits, loss = model(x, y)
        
        if torch.isnan(loss):
            print('[WARN] NaN loss, skipping batch')
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * (x.size(1) - 1)  # 减去1因为移位
        total_tokens += (x.size(0) * (x.size(1) - 1))
        
        # 计算准确率（next token prediction）
        shift_logits = logits[:, :-1, :]
        shift_targets = y[:, 1:]
        pred = shift_logits.argmax(dim=-1)
        correct += (pred == shift_targets).sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    acc = correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, perplexity, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        
        total_loss += loss.item() * (x.size(1) - 1)
        total_tokens += (x.size(0) * (x.size(1) - 1))
        
        shift_logits = logits[:, :-1, :]
        shift_targets = y[:, 1:]
        pred = shift_logits.argmax(dim=-1)
        correct += (pred == shift_targets).sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    acc = correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, perplexity, acc


def generate_text(model, dataset, device, prompt="To be, or not to be", length=200):
    """生成文本"""
    model.eval()
    
    # 编码 prompt
    prompt_ids = [dataset.stoi.get(c, 0) for c in prompt]
    if len(prompt_ids) > dataset.seq_len:
        prompt_ids = prompt_ids[-dataset.seq_len:]
    
    x = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    generated = list(prompt)
    
    with torch.no_grad():
        for _ in range(length):
            # 如果序列太长，截断
            if x.size(1) > dataset.seq_len:
                x = x[:, -dataset.seq_len:]
            
            logits = model(x)
            next_token_logits = logits[0, -1, :]  # 最后一个位置
            
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(dataset.itos[next_token])
            x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
    
    return ''.join(generated)


def main():
    print('=== HormonicCharLM - Tiny Shakespeare ===')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # 配置
    seq_len = 128
    vocab_size = None  # 从数据确定
    
    config = {
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'seq_len': seq_len,
            'n_steps': 3,
            'n_classes': 65,  # 会被覆盖
            'patch_size': 1,
            'D0_amp': 0.002,
            'D0_phase': 0.002,
            'dt': 0.02,
            'noise_scale': 0.0,
            'dropout': 0.1,
            'ei_balance': {'enabled': True, 'tau_e': 2.0, 'tau_i': 1.0, 'gamma_e': 1.0, 'gamma_i': 0.8, 'w_inh': 0.3, 'inh_radius': 3},
            'sensory_feedback': {'enabled': True, 'feedback_strength': 0.3, 'feedback_freq': 1},
            'hebbian': {'enabled': True, 'eta_hebb': 0.001, 'eta_anti': 0.0005, 'sync_threshold': 0.5, 'tau_hebb': 10.0, 'decay': 0.999},
            'cross_freq_coupling': {'enabled': False},
            'energy_constraint': {'enabled': False}
        },
        'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
        'bwo': {'use_bwo': True, 'evolve_interval': 5, 'flip_ratio': 0.3},
        'pc': {'use_pc': False, 'pred_hidden_mult': 4, 'aux_weight': 0.01}
    }
    
    # 下载数据
    text = download_tiny_shakespeare()
    print(f'\nDataset: Tiny Shakespeare')
    print(f'  Total chars: {len(text)}')
    
    # 创建数据集
    train_ds = CharDataset(text, seq_len, train=True)
    val_ds = CharDataset(text, seq_len, train=False)
    vocab_size = train_ds.vocab_size
    
    print(f'  Vocab size: {vocab_size}')
    print(f'  Train samples: {len(train_ds)}')
    print(f'  Val samples: {len(val_ds)}')
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 更新配置
    config['model']['n_classes'] = vocab_size
    
    # 创建模型
    model = HormonicCharLM(config, vocab_size, seq_len).to(device)
    print(f'\nModel: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    
    # 训练
    print('\nTraining 10 epochs...')
    best_ppl = float('inf')
    
    for epoch in range(10):
        train_loss, train_ppl, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_ppl, val_acc = evaluate(model, val_loader, device)
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train: Loss={train_loss:.3f}, PPL={train_ppl:.1f}, Acc={train_acc*100:.1f}%')
        print(f'  Val:   Loss={val_loss:.3f}, PPL={val_ppl:.1f}, Acc={val_acc*100:.1f}%')
        
        # 生成样本
        if epoch % 2 == 0:
            generated = generate_text(model, train_ds, device, 
                                   prompt="To be, or not to be", length=100)
            print(f'  Sample: {generated[:100]}...')
        
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'vocab': {'stoi': train_ds.stoi, 'itos': train_ds.itos},
                'config': config
            }, 'best_charlm.pt')
    
    print(f'\nBest Val PPL: {best_ppl:.1f}')
    print('Done!')


if __name__ == '__main__':
    main()
