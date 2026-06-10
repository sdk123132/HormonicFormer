"""
Minimal Char LM test
"""
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hormonicformer_v3 import HormonicFormer

class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len - 1
    
    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y

class HormonicCharLM(nn.Module):
    def __init__(self, config, vocab_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        d_model = config['model']['d_model']
        
        self.embed = nn.Embedding(vocab_size, d_model)
        config['model']['seq_len'] = seq_len
        config['model']['patch_size'] = 1
        self.model = HormonicFormer(config)
        self.model.classifier = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, targets=None):
        B, S = input_ids.shape
        x = self.embed(input_ids)  # [B, S, d_model]
        vals = x.mean(dim=-1)  # [B, S]
        img = vals.view(B, 1, 1, S)  # [B, 1, 1, S]
        
        x = self.model.patch_embed(img)  # [B, d_model, 1, S]
        x = x.flatten(2).transpose(1, 2)  # [B, S, d_model]
        
        for block in self.model.blocks:
            x = block(x)
        
        logits = self.model.classifier(x)  # [B, S, vocab_size]
        
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_targets.view(-1)
            )
            return logits, loss
        return logits

# Test
text = "To be, or not to be, that is the question." * 50
print(f'Text length: {len(text)}')

ds = CharDataset(text, seq_len=32)
print(f'Dataset: {len(ds)} samples, vocab={ds.vocab_size}')

loader = DataLoader(ds, batch_size=8, shuffle=True)

config = {
    'model': {
        'd_model': 64, 'n_heads': 4, 'n_layers': 2, 'seq_len': 32, 'n_steps': 2,
        'n_classes': 10, 'patch_size': 1, 'D0_amp': 0.002, 'D0_phase': 0.002,
        'dt': 0.02, 'noise_scale': 0.0, 'dropout': 0.1,
        'ei_balance': {'enabled': False}, 'sensory_feedback': {'enabled': False},
        'hebbian': {'enabled': False}, 'cross_freq_coupling': {'enabled': False},
        'energy_constraint': {'enabled': False}
    },
    'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
    'bwo': {'use_bwo': False}, 'pc': {'use_pc': False}
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = HormonicCharLM(config, ds.vocab_size, 32).to(device)
print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.3f}M params')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('\nTraining 3 epochs...')
for epoch in range(3):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}: Loss={total_loss/len(loader):.3f}')

print('\nDone!')
