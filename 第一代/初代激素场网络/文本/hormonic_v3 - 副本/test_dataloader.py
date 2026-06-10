"""
Test DataLoader with batch
"""
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
from torch.utils.data import Dataset, DataLoader
from hormonicformer_v3 import HormonicFormer
import torch.nn as nn

class CopyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        s = self.data[idx]
        return s, s.clone()

print('Creating dataset...')
ds = CopyDataset(10, 128, 100)
print(f'Dataset: {len(ds)}')

print('Creating DataLoader...')
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
print(f'Loader: {len(loader)} batches')

print('Getting one batch...')
x, y = next(iter(loader))
print(f'Batch: x={x.shape}, y={y.shape}')

print('Creating model...')
config = {
    'model': {
        'd_model': 64, 'n_heads': 4, 'n_layers': 2, 'seq_len': 128, 'n_steps': 2,
        'n_classes': 10, 'patch_size': 1, 'D0_amp': 0.002, 'D0_phase': 0.002,
        'dt': 0.02, 'noise_scale': 0.0, 'dropout': 0.1,
        'ei_balance': {'enabled': False}, 'sensory_feedback': {'enabled': False},
        'hebbian': {'enabled': False}, 'cross_freq_coupling': {'enabled': False},
        'energy_constraint': {'enabled': False}
    },
    'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
    'bwo': {'use_bwo': False}, 'pc': {'use_pc': False}
}

class Wrapper(nn.Module):
    def __init__(self, config, vocab_size, seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        config['model']['seq_len'] = seq_len
        config['model']['patch_size'] = 1
        self.model = HormonicFormer(config)
        self.model.classifier = nn.Linear(config['model']['d_model'], vocab_size)
    
    def forward(self, seq):
        B, S = seq.shape
        vals = (seq.float() / (self.vocab_size - 1)) * 2 - 1
        img = vals.view(B, 1, 1, S)
        x = self.model.patch_embed(img)
        x = x.flatten(2).transpose(1, 2)
        for block in self.model.blocks:
            x = block(x)
        return self.model.classifier(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = Wrapper(config, 10, 128).to(device)
print(f'Model created!')

print('Testing forward...')
x = x.to(device)
logits = model(x)
print(f'Logits: {logits.shape}')

print('Testing backward...')
import torch.nn.functional as F
y = y.to(device)
loss = F.cross_entropy(logits.view(-1, 10), y.view(-1))
print(f'Loss: {loss.item():.4f}')
loss.backward()
print('Backward done!')

print('\nAll tests passed!')
