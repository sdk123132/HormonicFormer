"""
Simple Copy Task S=128 training
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

print('=== Copy Task S=128 ===')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

vocab_size, seq_len = 10, 128
model = Wrapper(config, vocab_size, seq_len).to(device)
print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.3f}M params')

train_ds = CopyDataset(vocab_size, seq_len, 2000)
val_ds = CopyDataset(vocab_size, seq_len, 200)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('\nTraining 10 epochs...')
for epoch in range(10):
    # Train
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    train_acc = correct / total
    
    # Val
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
    
    print(f'Epoch {epoch+1}: Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%')

print('\nDone!')
