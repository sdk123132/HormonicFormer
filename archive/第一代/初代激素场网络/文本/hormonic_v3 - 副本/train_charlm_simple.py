"""
Char LM - 基于 Copy Task 成功模式
核心：保持 seq_to_patches，只改损失为 next token prediction
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
        y = self.data[idx + 1: idx + self.seq_len + 1]  # 移位目标
        return x, y

class CharLMWrapper(nn.Module):
    """和 Copy Task 完全相同的结构，只改分类头"""
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
        """和 Copy Task 完全相同"""
        B, S = seq.shape
        vals = (seq.float() / (self.vocab_size - 1)) * 2 - 1
        img = vals.view(B, 1, 1, S)
        
        x = self.model.patch_embed(img)
        x = x.flatten(2).transpose(1, 2)
        
        for block in self.model.blocks:
            x = block(x)
        
        return self.model.classifier(x)

# 使用 Hamlet 文本
text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life;""" * 20

print('=== Char LM (S=64) ===')
print(f'Text length: {len(text)}')

seq_len = 64
ds = CharDataset(text, seq_len)
print(f'Dataset: {len(ds)} samples, vocab={ds.vocab_size}')

# 数据加载器
train_size = int(0.9 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# 配置（和 Copy Task 成功配置相同）
config = {
    'model': {
        'd_model': 64, 'n_heads': 4, 'n_layers': 2, 'seq_len': seq_len, 'n_steps': 2,
        'n_classes': ds.vocab_size, 'patch_size': 1, 'D0_amp': 0.002, 'D0_phase': 0.002,
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

model = CharLMWrapper(config, ds.vocab_size, seq_len).to(device)
print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.3f}M params')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('\nTraining 10 epochs...')
for epoch in range(10):
    # Train
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        logits = model(x)  # [B, S, vocab_size]
        
        # Next token prediction: 用 [0:S-1] 预测 [1:S]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = y[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, ds.vocab_size),
            shift_targets.view(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = shift_logits.argmax(dim=-1)
        correct += (pred == shift_targets).sum().item()
        total += shift_targets.numel()
    
    train_acc = correct / total
    
    # Val
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            shift_logits = logits[:, :-1, :]
            shift_targets = y[:, 1:]
            pred = shift_logits.argmax(dim=-1)
            val_correct += (pred == shift_targets).sum().item()
            val_total += shift_targets.numel()
    
    val_acc = val_correct / val_total
    avg_loss = total_loss / len(train_loader)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f'Epoch {epoch+1}: Loss={avg_loss:.3f}, PPL={ppl:.1f}, Train Acc={train_acc*100:.1f}%, Val Acc={val_acc*100:.1f}%')

print('\nDone!')
