import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
print('Step 1: Import modules...')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hormonicformer_v3 import HormonicFormer
print('Step 2: Modules imported')

class CopyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        s = self.data[idx]
        return s, s.clone()
print('Step 3: Dataset class defined')

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
print('Step 4: Wrapper class defined')

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

print('Step 5: Creating dataset...')
train_ds = CopyDataset(10, 128, 100)
print(f'Step 6: Dataset created: {len(train_ds)}')

print('Step 7: Creating DataLoader...')
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
print(f'Step 8: DataLoader created: {len(train_loader)} batches')

print('Step 9: Creating model...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Wrapper(config, 10, 128).to(device)
print(f'Step 10: Model created on {device}')

print('Step 11: Getting one batch...')
x, y = next(iter(train_loader))
print(f'Step 12: Batch: x={x.shape}, y={y.shape}')

print('Step 13: Forward pass...')
x = x.to(device)
logits = model(x)
print(f'Step 14: Logits: {logits.shape}')

print('Step 15: Loss...')
y = y.to(device)
loss = F.cross_entropy(logits.view(-1, 10), y.view(-1))
print(f'Step 16: Loss: {loss.item():.4f}')

print('Step 17: Backward...')
loss.backward()
print('Step 18: Backward done!')

print('Step 19: Optimizer...')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.step()
print('Step 20: Optimizer step done!')

print('\nAll steps passed! Training loop should work.')
