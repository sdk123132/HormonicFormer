"""
Extended Ablation Test
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
            'hebbian': {'enabled': False, 'lr': 0.001, 'decay': 0.99},
            'cross_freq_coupling': {'enabled': True},
            'energy_constraint': {'enabled': True}
        },
        'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
        'bwo': {'use_bwo': False}, 'pc': {'use_pc': False}
    }

def test(name, cfg, epochs=3):
    print(f'\n>>> {name}')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        print(f'  E{epoch+1}: {acc*100:.1f}%')
    return acc

print('='*50)
print('Extended Ablation (No Hebbian Base)')
print('='*50)

results = []

# Base: No Hebbian
cfg = get_cfg()
acc = test('Base (No Hebbian)', cfg)
results.append(('Base (No Hebbian)', acc))

# No Diffusion
cfg = get_cfg()
cfg['model']['D0_amp'] = 0.0
cfg['model']['D0_phase'] = 0.0
acc = test('No Diffusion', cfg)
results.append(('No Diffusion', acc))

# No Reaction (dt=0)
cfg = get_cfg()
cfg['model']['dt'] = 0.0
acc = test('No Reaction (dt=0)', cfg)
results.append(('No Reaction', acc))

# No Sensory Feedback
cfg = get_cfg()
cfg['model']['sensory_feedback']['enabled'] = False
acc = test('No Feedback', cfg)
results.append(('No Feedback', acc))

# No E/I Balance
cfg = get_cfg()
cfg['model']['ei_balance']['enabled'] = False
acc = test('No E/I Balance', cfg)
results.append(('No E/I', acc))

# No Cross-Freq
cfg = get_cfg()
cfg['model']['cross_freq_coupling']['enabled'] = False
acc = test('No Cross-Freq', cfg)
results.append(('No Cross-Freq', acc))

# No Energy
cfg = get_cfg()
cfg['model']['energy_constraint']['enabled'] = False
acc = test('No Energy', cfg)
results.append(('No Energy', acc))

print('\n' + '='*50)
print('Summary (Epoch 3 Acc):')
for name, acc in results:
    print(f'  {name:<20}: {acc*100:.1f}%')
print('='*50)
