"""
Transformer vs HormonicFormer - Small Model
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
    def __init__(self, seq_len, n_samples=1000):
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
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=1, seq_len=64, dropout=0.0):
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


def get_hormonic_cfg():
    return {
        'model': {
            'd_model': 64, 'n_heads': 2, 'n_layers': 1, 'seq_len': 64,
            'n_steps': 2, 'n_classes': 10, 'patch_size': 1,
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
    print(f'\n>>> {name}')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_params/1e6:.3f}M ({n_params:,})')
    
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    results = []
    
    for epoch in range(epochs):
        t0 = time.time()
        
        model.train()
        train_correct, train_total = 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=-1)
            train_correct += (pred == y).sum().item()
            train_total += y.numel()
        train_acc = train_correct / train_total
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                pred = logits.argmax(dim=-1)
                val_correct += (pred == y).sum().item()
                val_total += y.numel()
        val_acc = val_correct / val_total
        
        t_elapsed = time.time() - t0
        results.append((train_acc, val_acc, t_elapsed))
        print(f'  E{epoch+1}: Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%, Time={t_elapsed:.1f}s')
    
    return results


print('='*60)
print('Transformer vs HormonicFormer')
print('Copy Task S=64, d_model=64, n_layers=1')
print('='*60)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {dev}')

# Data
train_ds = CopyDataset(64, 800)
val_ds = CopyDataset(64, 200)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# HormonicFormer
cfg = get_hormonic_cfg()
model_h = HormonicWrapper(cfg, 10, 64).to(dev)
res_h = train('HormonicFormer', model_h, train_loader, val_loader, dev, epochs=5)

# Transformer
model_t = TransformerModel(vocab_size=10, d_model=64, n_heads=2, n_layers=1, seq_len=64, dropout=0.0).to(dev)
res_t = train('Transformer', model_t, train_loader, val_loader, dev, epochs=5)

# Summary
print('\n' + '='*60)
print('Comparison Summary')
print('='*60)
print(f'\n{"Epoch":<8}{"Hormonic":<15}{"Transformer":<15}{"Diff":<10}{"Time H":<10}{"Time T":<10}')
print('-'*60)
for i in range(5):
    h_acc, h_time = res_h[i][1] * 100, res_h[i][2]
    t_acc, t_time = res_t[i][1] * 100, res_t[i][2]
    diff = h_acc - t_acc
    print(f'{i+1:<8}{h_acc:<15.1f}{t_acc:<15.1f}{diff:+.1f}%     {h_time:<10.1f}{t_time:<10.1f}')

print('\n' + '='*60)
