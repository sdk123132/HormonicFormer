"""
Ablation Study - Simple Version
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

from hormonicformer_v3 import HormonicFormer


class CopyTaskDataset(Dataset):
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


class CopyTaskWrapper(nn.Module):
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


def get_base_config():
    return {
        'model': {
            'd_model': 128, 'n_heads': 4, 'n_layers': 2, 'seq_len': 128,
            'n_steps': 4, 'n_classes': 10, 'patch_size': 1,
            'D0_amp': 0.002, 'D0_phase': 0.002, 'dt': 0.02,
            'noise_scale': 0.0, 'dropout': 0.1,
            'ei_balance': {'enabled': True, 'target_ratio': 4.0},
            'sensory_feedback': {'enabled': True, 'top_k': 8},
            'hebbian': {'enabled': True, 'lr': 0.001, 'decay': 0.99},
            'cross_freq_coupling': {'enabled': True},
            'energy_constraint': {'enabled': True}
        },
        'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': True},
        'bwo': {'use_bwo': False}, 'pc': {'use_pc': False}
    }


def run_exp(name, config, seq_len=128, epochs=5):
    print(f'\n=== {name} ===')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_ds = CopyTaskDataset(seq_len, 8000)
    val_ds = CopyTaskDataset(seq_len, 2000)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = CopyTaskWrapper(config, 10, seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
            loss.backward()
            optimizer.step()
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
        train_acc = correct / total
        
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
        best_acc = max(best_acc, val_acc)
        
        print(f'  E{epoch+1}: Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%')
    
    return {'name': name, 'best_val_acc': round(best_acc * 100, 2)}


def main():
    print('Ablation Study - Copy Task S=128')
    print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []
    
    # 1. Full
    cfg = get_base_config()
    results.append(run_exp('Full', cfg))
    
    # 2. No Diffusion
    cfg = get_base_config()
    cfg['model']['D0_amp'] = 0.0
    cfg['model']['D0_phase'] = 0.0
    results.append(run_exp('No Diffusion', cfg))
    
    # 3. No Reaction (dt=0)
    cfg = get_base_config()
    cfg['model']['dt'] = 0.0
    results.append(run_exp('No Reaction', cfg))
    
    # 4. No Hebbian
    cfg = get_base_config()
    cfg['model']['hebbian']['enabled'] = False
    results.append(run_exp('No Hebbian', cfg))
    
    # 5. No Sensory Feedback
    cfg = get_base_config()
    cfg['model']['sensory_feedback']['enabled'] = False
    results.append(run_exp('No Feedback', cfg))
    
    # Save
    with open('ablation_simple.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n=== Summary ===')
    for r in results:
        print(f"{r['name']:<20}: {r['best_val_acc']:.1f}%")


if __name__ == '__main__':
    main()
