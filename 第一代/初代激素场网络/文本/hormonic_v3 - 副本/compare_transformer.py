"""
Transformer Comparison
Compare HormonicFormer v3 with Transformer (same params)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import time
from datetime import datetime
import math

from hormonicformer_v3 import HormonicFormer


class CopyTaskDataset(Dataset):
    """Copy Task 数据集"""
    def __init__(self, seq_len, num_samples=10000):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.vocab_size = 10
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = x.clone()
        return x, y


class PositionalEncoding(nn.Module):
    """Transformer 位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """微型 Transformer，参数量与 HormonicFormer 匹配"""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, seq_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src):
        # src: [B, S]
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # 因果掩码
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        output = self.transformer_encoder(src, mask=mask, is_causal=True)
        output = self.fc_out(output)
        return output


class HormonicWrapper(nn.Module):
    """HormonicFormer v3 包装器"""
    def __init__(self, config, vocab_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        config['model']['seq_len'] = seq_len
        config['model']['patch_size'] = 1
        config['model']['n_classes'] = vocab_size
        
        self.model = HormonicFormer(config)
        d_model = config['model']['d_model']
        self.model.classifier = nn.Linear(d_model, vocab_size)
    
    def forward(self, seq):
        B, S = seq.shape
        vals = (seq.float() / (self.vocab_size - 1)) * 2 - 1
        img = vals.view(B, 1, 1, S)
        
        x = self.model.patch_embed(img)
        x = x.flatten(2).transpose(1, 2)
        
        for block in self.model.blocks:
            x = block(x)
        
        return self.model.classifier(x)


def get_hormonic_config():
    """HormonicFormer 配置（与 Transformer 参数量匹配）"""
    return {
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'seq_len': 128,
            'n_steps': 4,
            'n_classes': 10,
            'patch_size': 1,
            'D0_amp': 0.002,
            'D0_phase': 0.002,
            'dt': 0.02,
            'noise_scale': 0.0,
            'dropout': 0.1,
            'ei_balance': {'enabled': True, 'target_ratio': 4.0},
            'sensory_feedback': {'enabled': True, 'top_k': 8},
            'hebbian': {'enabled': True, 'lr': 0.001, 'decay': 0.99},
            'cross_freq_coupling': {'enabled': True},
            'energy_constraint': {'enabled': True}
        },
        'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': True},
        'bwo': {'use_bwo': False},
        'pc': {'use_pc': False}
    }


def run_comparison(model_name, model, train_loader, val_loader, device, epochs=10):
    """运行对比实验"""
    print(f'\n{"="*60}')
    print(f'模型: {model_name}')
    print(f'参数量: {sum(p.numel() for p in model.parameters())/1e6:.3f}M')
    print(f'{"="*60}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    results = []
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练
        model.train()
        train_correct, train_total = 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            pred = logits.argmax(dim=-1)
            train_correct += (pred == y).sum().item()
            train_total += y.numel()
        
        train_acc = train_correct / train_total
        
        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=-1)
                val_correct += (pred == y).sum().item()
                val_total += y.numel()
        
        val_acc = val_correct / val_total
        epoch_time = time.time() - start_time
        
        results.append({
            'epoch': epoch + 1,
            'train_acc': round(train_acc * 100, 2),
            'val_acc': round(val_acc * 100, 2),
            'time_sec': round(epoch_time, 2)
        })
        
        print(f'Epoch {epoch+1}: Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%, Time={epoch_time:.1f}s')
    
    return results


def main():
    """主对比实验"""
    print('='*60)
    print('HormonicFormer v3 vs Transformer 对比实验')
    print(f'开始时间: {datetime.now()}')
    print('='*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 128
    batch_size = 32
    
    # 数据集
    train_ds = CopyTaskDataset(seq_len, 8000)
    val_ds = CopyTaskDataset(seq_len, 2000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f'\n任务: Copy Task S={seq_len}')
    print(f'训练集: {len(train_ds)}, 验证集: {len(val_ds)}')
    print(f'Batch size: {batch_size}')
    
    all_results = {}
    
    # 1. HormonicFormer v3
    print('\n' + '='*60)
    print('测试 HormonicFormer v3')
    config = get_hormonic_config()
    model_h = HormonicWrapper(config, 10, seq_len).to(device)
    results_h = run_comparison('HormonicFormer v3', model_h, train_loader, val_loader, device, epochs=10)
    all_results['hormonicformer'] = {
        'name': 'HormonicFormer v3',
        'params_M': sum(p.numel() for p in model_h.parameters())/1e6,
        'results': results_h
    }
    
    # 2. Transformer
    print('\n' + '='*60)
    print('测试 Transformer')
    model_t = TransformerModel(vocab_size=10, d_model=128, n_heads=4, n_layers=2, seq_len=seq_len).to(device)
    results_t = run_comparison('Transformer', model_t, train_loader, val_loader, device, epochs=10)
    all_results['transformer'] = {
        'name': 'Transformer',
        'params_M': sum(p.numel() for p in model_t.parameters())/1e6,
        'results': results_t
    }
    
    # 保存结果
    output = {
        'timestamp': str(datetime.now()),
        'task': f'Copy Task S={seq_len}',
        'comparison': all_results
    }
    
    with open('comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # 打印对比表
    print('\n' + '='*60)
    print('对比结果汇总')
    print('='*60)
    print(f'\n{"Epoch":<8}{"HormonicFormer":<20}{"Transformer":<20}{"Diff":<15}')
    print('-'*60)
    
    for i in range(10):
        h_acc = all_results['hormonicformer']['results'][i]['val_acc']
        t_acc = all_results['transformer']['results'][i]['val_acc']
        diff = h_acc - t_acc
        print(f'{i+1:<8}{h_acc:<20.1f}{t_acc:<20.1f}{diff:+.1f}%')
    
    print('\n' + '='*60)
    print('对比实验完成！')
    print(f'结果保存到: comparison_results.json')
    print(f'结束时间: {datetime.now()}')
    print('='*60)


if __name__ == '__main__':
    main()
