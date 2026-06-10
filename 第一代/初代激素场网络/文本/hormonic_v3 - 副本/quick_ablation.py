"""
Quick Ablation Test - 3 experiments only
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

print('Importing HormonicFormer...')
from hormonicformer_v3 import HormonicFormer
print('Import OK')

class CopyDataset(Dataset):
    def __init__(self, seq_len, n_samples=1000):
        self.seq_len = seq_len
        self.n_samples = n_samples
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        x = torch.randint(0, 10, (self.seq_len,))
        return x, x.clone()

class Wrapper(nn.Module):
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

def get_cfg():
    return {
        'model': {
            'd_model': 64, 'n_heads': 2, 'n_layers': 1, 'seq_len': 64,
            'n_steps': 2, 'n_classes': 10, 'patch_size': 1,
            'D0_amp': 0.002, 'D0_phase': 0.002, 'dt': 0.02,
            'noise_scale': 0.0, 'dropout': 0.0,
            'ei_balance': {'enabled': True, 'target_ratio': 4.0},
            'sensory_feedback': {'enabled': True, 'top_k': 4},
            'hebbian': {'enabled': True, 'lr': 0.001, 'decay': 0.99},
            'cross_freq_coupling': {'enabled': True},
            'energy_constraint': {'enabled': True}
        },
        'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
        'bwo': {'use_bwo': False}, 'pc': {'use_pc': False}
    }

def test(name, cfg, epochs=3):
    print(f'\n>>> {name}')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}')
    
    ds = CopyDataset(64, 500)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    
    model = Wrapper(cfg, 10, 64).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
        acc = correct / total
        print(f'  Epoch {epoch+1}: Acc={acc*100:.1f}%')
    return acc

print('='*50)
print('Quick Ablation Test')
print('='*50)

# 1. Full
cfg = get_cfg()
acc_full = test('Full Model', cfg)

# 2. No Diffusion
cfg = get_cfg()
cfg['model']['D0_amp'] = 0.0
cfg['model']['D0_phase'] = 0.0
acc_no_diff = test('No Diffusion', cfg)

# 3. No Hebbian
cfg = get_cfg()
cfg['model']['hebbian']['enabled'] = False
acc_no_heb = test('No Hebbian', cfg)

print('\n' + '='*50)
print('Summary:')
print(f'  Full Model:     {acc_full*100:.1f}%')
print(f'  No Diffusion:   {acc_no_diff*100:.1f}%')
print(f'  No Hebbian:     {acc_no_heb*100:.1f}%')
print('='*50)
