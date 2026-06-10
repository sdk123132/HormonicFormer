"""
Minimal Copy Task test
"""
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from hormonicformer_v3 import HormonicFormer


class CopyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        s = self.data[idx]
        return s, s.clone()


class Wrapper(nn.Module):
    def __init__(self, config, vocab_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        config['model']['seq_len'] = seq_len
        config['model']['patch_size'] = 1
        
        self.model = HormonicFormer(config)
        d_model = config['model']['d_model']
        self.model.classifier = nn.Linear(d_model, vocab_size)
    
    def forward(self, seq):
        B, S = seq.shape
        vals = (seq.float() / (self.vocab_size - 1)) * 2 - 1
        img = vals.view(B, 1, 1, S)
        
        x = self.model.patch_embed(img)
        x = x.flatten(2).transpose(1, 2)
        
        x_embed = x.detach().clone()
        for block in self.model.blocks:
            x = block(x, x_embed=x_embed if block.use_feedback else None)
        
        return self.model.classifier(x)


# Config
config = {
    'model': {
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'seq_len': 16,
        'n_steps': 3,
        'n_classes': 10,
        'patch_size': 1,
        'D0_amp': 0.002,
        'D0_phase': 0.002,
        'dt': 0.02,
        'noise_scale': 0.01,
        'dropout': 0.1,
        'ei_balance': {'enabled': True, 'tau_e': 2.0, 'tau_i': 1.0, 'gamma_e': 1.0, 'gamma_i': 0.8, 'w_inh': 0.3, 'inh_radius': 3},
        'sensory_feedback': {'enabled': True, 'feedback_strength': 0.3, 'feedback_freq': 1},
        'hebbian': {'enabled': True, 'eta_hebb': 0.001, 'eta_anti': 0.0005, 'sync_threshold': 0.5, 'tau_hebb': 10.0, 'decay': 0.999},
        'cross_freq_coupling': {'enabled': False},
        'energy_constraint': {'enabled': False}
    },
    'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
    'bwo': {'use_bwo': True, 'evolve_interval': 5, 'flip_ratio': 0.3},
    'pc': {'use_pc': True, 'pred_hidden_mult': 4, 'aux_weight': 0.01}
}

print('Setup...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

vocab_size, seq_len = 10, 16
model = Wrapper(config, vocab_size, seq_len).to(device)
print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

train_ds = CopyDataset(vocab_size, seq_len, 1000)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
print(f'Dataset: {len(train_ds)} samples')

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

print('\nTraining...')
for epoch in range(3):
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    
    acc = correct / total
    print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc*100:.1f}%')

print('\nDone!')
