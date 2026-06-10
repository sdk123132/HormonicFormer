"""
Copy Task S=128 - Debug version
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


# Minimal config
config = {
    'model': {
        'd_model': 64,  # 减小模型
        'n_heads': 4,
        'n_layers': 2,
        'seq_len': 128,
        'n_steps': 2,  # 减少步数
        'n_classes': 10,
        'patch_size': 1,
        'D0_amp': 0.002,
        'D0_phase': 0.002,
        'dt': 0.02,
        'noise_scale': 0.0,
        'dropout': 0.1,
        'ei_balance': {'enabled': False},  # 禁用 EI
        'sensory_feedback': {'enabled': False},  # 禁用反馈
        'hebbian': {'enabled': False},  # 禁用 Hebbian
        'cross_freq_coupling': {'enabled': False},
        'energy_constraint': {'enabled': False}
    },
    'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
    'bwo': {'use_bwo': False},
    'pc': {'use_pc': False}
}

print('=== Copy Task S=128 (Debug) ===')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

vocab_size, seq_len = 10, 128
print('Creating model...')
model = Wrapper(config, vocab_size, seq_len).to(device)
print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.3f}M params')

print('Creating dataset...')
train_ds = CopyDataset(vocab_size, seq_len, 1000)  # 减小数据集
val_ds = CopyDataset(vocab_size, seq_len, 200)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

print('Testing one batch...')
x, y = next(iter(train_loader))
x, y = x.to(device), y.to(device)
print(f'Batch: x={x.shape}, y={y.shape}')

print('Forward pass...')
logits = model(x)
print(f'Logits: {logits.shape}')

print('Loss...')
loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
print(f'Loss: {loss.item():.4f}')

print('Backward...')
loss.backward()
print('Backward done')

print('\n=== Training 5 epochs ===')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for i, (x, y) in enumerate(train_loader):
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
        
        if i % 10 == 0:
            print(f'  batch {i}/{len(train_loader)}: loss={loss.item():.4f}')
    
    train_acc = correct / total
    print(f'Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, acc={train_acc*100:.1f}%')

print('\nDone!')
