"""
Long Sequence Battle: HormonicFormer vs Transformer S=512
测试长程泛化能力
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
import math
import time

from hormonicformer_v3 import HormonicFormer


class CopyDataset(Dataset):
    def __init__(self, seq_len, n_samples=200):
        self.seq_len = seq_len
        self.n_samples = n_samples
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        x = torch.randint(0, 10, (self.seq_len,))
        return x, x.clone()


class PositionalEncoding(nn.Module):
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
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=1, seq_len=512, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.fc(x)


class HormonicWrapper(nn.Module):
    def __init__(self, cfg, vocab, seq_len):
        super().__init__()
        cfg['model']['seq_len'] = seq_len
        cfg['model']['patch_size'] = 1
        cfg['model']['n_classes'] = vocab
        self.model = HormonicFormer(cfg)
        self.model.classifier = nn.Linear(cfg['model']['d_model'], vocab)
    def forward(self, seq):
        B, S = seq.shape
        img = ((seq.float() / 9) * 2 - 1).view(B, 1, 1, S)
        x = self.model.patch_embed(img).flatten(2).transpose(1, 2)
        for blk in self.model.blocks:
            x = blk(x)
        return self.model.classifier(x)


def get_cfg(seq_len):
    return {
        'model': {
            'd_model': 64, 'n_heads': 2, 'n_layers': 1, 'seq_len': seq_len,
            'n_steps': 4, 'n_classes': 10, 'patch_size': 1,
            'D0_amp': 0.002, 'D0_phase': 0.002, 'dt': 0.02,
            'noise_scale': 0.0, 'dropout': 0.0,
            'ei_balance': {'enabled': True, 'target_ratio': 4.0},
            'sensory_feedback': {'enabled': True, 'top_k': 4},
            'hebbian': {'enabled': False},
            'cross_freq_coupling': {'enabled': True},
            'energy_constraint': {'enabled': True}
        },
        'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
        'bwo': {'use_bwo': False}, 'pc': {'use_pc': False}
    }


def train(name, model, train_loader, val_loader, dev, epochs=5):
    print(f'\n>>> {name} ({sum(p.numel() for p in model.parameters())/1e3:.0f}K params)')
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    results = []
    
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        tc, tt = 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=-1)
            tc += (pred == y).sum().item()
            tt += y.numel()
        train_acc = tc / tt
        
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                pred = logits.argmax(dim=-1)
                vc += (pred == y).sum().item()
                vt += y.numel()
        val_acc = vc / vt
        elapsed = time.time() - t0
        results.append(val_acc)
        print(f'  E{epoch+1}: Val={val_acc*100:.1f}% Time={elapsed:.1f}s')
    return results


print('='*60)
print('Long Sequence Battle: S=512')
print('='*60)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {dev}')

# Small dataset for speed
train_ds = CopyDataset(512, 200)
val_ds = CopyDataset(512, 50)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# HormonicFormer
print('\n--- HormonicFormer ---')
model_h = HormonicWrapper(get_cfg(512), 10, 512).to(dev)
res_h = train('HormonicFormer S=512', model_h, train_loader, val_loader, dev, epochs=5)

# Transformer
print('\n--- Transformer ---')
model_t = TransformerModel(vocab_size=10, d_model=64, n_heads=2, n_layers=1, seq_len=512).to(dev)
res_t = train('Transformer S=512', model_t, train_loader, val_loader, dev, epochs=5)

# Summary
print('\n' + '='*60)
print('S=512 Battle Results')
print('='*60)
print(f'\n{"Epoch":<8}{"Hormonic":<15}{"Transformer":<15}{"Diff":<10}')
print('-'*60)
for i in range(5):
    h = res_h[i] * 100
    t = res_t[i] * 100
    print(f'{i+1:<8}{h:<15.1f}{t:<15.1f}{h-t:+.1f}%')
print('='*60)
