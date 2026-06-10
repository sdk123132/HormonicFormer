"""
修复的 Copy Task 实验
简化版本：直接使用 1D 序列输入
"""
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'models'))
sys.path.insert(0, str(PROJECT_ROOT / 'field'))

from hormonicformer_v3 import HormonicFormer


class CopyTaskDataset(Dataset):
    """序列复制任务数据集"""
    def __init__(self, vocab_size, seq_len, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq, seq.clone()


class HormonicWrapper(nn.Module):
    """包装 HormonicFormer 用于序列任务"""
    def __init__(self, config, vocab_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.config = config
        
        # 修改配置：使用 1x1 patch（每个 token 作为一个像素）
        config['model']['patch_size'] = 1
        config['model']['seq_len'] = seq_len  # 关键：确保 seq_len 匹配
        
        self.model = HormonicFormer(config)
        
        # 替换分类头
        d_model = config['model']['d_model']
        self.model.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, vocab_size)
        )

    def forward(self, seq):
        """seq: [B, S] -> logits: [B, S, V]"""
        B, S = seq.shape
        
        # 将序列转换为单通道图像 [B, 1, 1, S]（高度为1的图像）
        # 这样 patch_embed (kernel=1, stride=1) 会保持长度
        vals = (seq.float() / (self.vocab_size - 1)) * 2 - 1  # 归一化到 [-1, 1]
        img = vals.view(B, 1, 1, S)  # [B, 1, 1, S] - 1行，S列的图像
        
        # 通过 HormonicFormer
        x = self.model.patch_embed(img)  # [B, d_model, 1, S]
        x = x.flatten(2).transpose(1, 2)  # [B, S, d_model]
        
        # 通过 blocks
        x_embed = x.detach().clone()
        for block in self.model.blocks:
            x = block(x, x_embed=x_embed if block.use_feedback else None)
        
        # 分类
        logits = self.model.classifier(x)  # [B, S, vocab_size]
        
        return logits


def train_epoch(model, loader, optimizer, device, vocab_size):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)  # [B, S, V]
        
        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * S, V),
            y.reshape(B * S)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += B * S

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device, vocab_size):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * S, V),
            y.reshape(B * S)
        )

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += B * S

    return total_loss / len(loader), correct / total


def run_experiment(seq_len, epochs, batch_size, device):
    vocab_size = 100
    
    print(f'\n{"="*60}')
    print(f'Sequence Length: {seq_len}')
    print(f'{"="*60}')
    
    # 创建配置
    config = {
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'seq_len': seq_len,  # 关键：匹配输入长度
            'n_steps': 3,
            'n_classes': vocab_size,
            'patch_size': 1,  # 1x1 patch，保持长度
            'D0_amp': 0.002,
            'D0_phase': 0.002,
            'dt': 0.02,
            'noise_scale': 0.01,
            'dropout': 0.1,
            'ei_balance': {
                'enabled': True,
                'tau_e': 2.0,
                'tau_i': 1.0,
                'gamma_e': 1.0,
                'gamma_i': 0.8,
                'w_inh': 0.3,
                'inh_radius': 3
            },
            'sensory_feedback': {
                'enabled': True,
                'feedback_strength': 0.3,
                'feedback_freq': 1
            },
            'hebbian': {
                'enabled': True,
                'eta_hebb': 0.001,
                'eta_anti': 0.0005,
                'sync_threshold': 0.5,
                'tau_hebb': 10.0,
                'decay': 0.999
            },
            'cross_freq_coupling': {'enabled': False},
            'energy_constraint': {'enabled': False}
        },
        'neuromod': {
            'da_init': 0.5,
            'da_min': 0.1,
            'da_max': 0.9,
            'use_cb': False
        },
        'bwo': {
            'use_bwo': True,
            'evolve_interval': 5,
            'flip_ratio': 0.3
        },
        'pc': {
            'use_pc': True,
            'pred_hidden_mult': 4,
            'aux_weight': 0.01
        }
    }
    
    # 创建模型
    model = HormonicWrapper(config, vocab_size, seq_len).to(device)
    
    print(f'  Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
    
    # 数据集
    train_ds = CopyTaskDataset(vocab_size, seq_len, num_samples=10000)
    val_ds = CopyTaskDataset(vocab_size, seq_len, num_samples=1000)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.001
    )
    
    # 训练
    best_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, vocab_size)
        val_loss, val_acc = evaluate(model, val_loader, device, vocab_size)
        
        print(f'  Epoch {epoch+1}/{epochs}: Train Acc={train_acc*100:.1f}%, Val Acc={val_acc*100:.1f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
    
    print(f'\n  Best Val Acc: {best_acc*100:.1f}%')
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[16, 64, 128])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    
    results = {}
    for seq_len in args.seq_lengths:
        acc = run_experiment(seq_len, args.epochs, args.batch_size, device)
        results[seq_len] = acc
    
    print(f'\n{"="*60}')
    print('Summary:')
    for seq_len, acc in sorted(results.items()):
        print(f'  S={seq_len}: {acc*100:.1f}%')


if __name__ == '__main__':
    main()
