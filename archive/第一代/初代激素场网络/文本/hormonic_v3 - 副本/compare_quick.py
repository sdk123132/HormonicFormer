"""
Transformer vs HormonicFormer Comparison
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

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
    def __init__(self, seq_len, n_samples=2000):
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
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, seq_len=128, dropout=0.1):
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
            'd_model': 128, 'n_heads': 4, 'n_layers': 2, 'seq_len': 128,
            'n_steps': 4, 'n_classes': 10, 'patch_size': 1,
            'D0_amp': 0.002, 'D0_phase': 0.002, 'dt': 0.02,
            'noise_scale': 0.0, 'dropout': 0.1,
            'ei_balance': {'enabled': True, 'target_ratio': 4.0},
            'sensory_feedback': {'enabled': True, 'top_k': 8},
            'hebbian': {'enabled': False},
            'cross_freq_coupling': {'enabled': True},
            'energy_constraint': {'enabled': True}
        },
        'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
        'bwo': {'use_bwo': False}, 'pc': {'use_pc': False}
    }


def train_model(name, model, train_loader, val_loader, dev, epochs=10):
    print(f'\n>>> {name}')
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M')
    
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    results = []
    
    for epoch in range(epochs):
        t0 = time.time()
        
        # Train
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
        
        # Val
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
print('Copy Task S=128, 10 epochs')
print('='*60)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {dev}')

# Data
train_ds = CopyDataset(128, 4000)
val_ds = CopyDataset(128, 1000)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# HormonicFormer
cfg = get_hormonic_cfg()
model_h = HormonicWrapper(cfg, 10, 128).to(dev)
res_h = train_model('HormonicFormer', model_h, train_loader, val_loader, dev, epochs=10)

# Transformer
model_t = TransformerModel(vocab_size=10, d_model=128, n_heads=4, n_layers=2, seq_len=128).to(dev)
res_t = train_model('Transformer', model_t, train_loader, val_loader, dev, epochs=10)

# Summary
print('\n' + '='*60)
print('Comparison Summary')
print('='*60)
print(f'\n{"Epoch":<8}{"Hormonic":<15}{"Transformer":<15}{"Diff":<10}')
print('-'*60)
for i in range(10):
    h_acc = res_h[i][1] * 100
    t_acc = res_t[i][1] * 100
    diff = h_acc - t_acc
    print(f'{i+1:<8}{h_acc:<15.1f}{t_acc:<15.1f}{diff:+.1f}%')

print('\n' + '='*60)
