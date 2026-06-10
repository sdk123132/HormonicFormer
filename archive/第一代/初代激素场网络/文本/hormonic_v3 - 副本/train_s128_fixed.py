"""
Copy Task S=128 - Fixed version
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


# Config - 禁用 BWO 避免数值问题，降低学习率
config = {
    'model': {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'seq_len': 128,
        'n_steps': 3,
        'n_classes': 10,
        'patch_size': 1,
        'D0_amp': 0.002,
        'D0_phase': 0.002,
        'dt': 0.02,
        'noise_scale': 0.0,  # 禁用噪声
        'dropout': 0.1,
        'ei_balance': {'enabled': True, 'tau_e': 2.0, 'tau_i': 1.0, 'gamma_e': 1.0, 'gamma_i': 0.8, 'w_inh': 0.3, 'inh_radius': 3},
        'sensory_feedback': {'enabled': True, 'feedback_strength': 0.3, 'feedback_freq': 1},
        'hebbian': {'enabled': True, 'eta_hebb': 0.0005, 'eta_anti': 0.0002, 'sync_threshold': 0.5, 'tau_hebb': 10.0, 'decay': 0.999},  # 降低 Hebbian 学习率
        'cross_freq_coupling': {'enabled': False},
        'energy_constraint': {'enabled': False}
    },
    'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
    'bwo': {'use_bwo': False, 'evolve_interval': 5, 'flip_ratio': 0.3},  # 禁用 BWO
    'pc': {'use_pc': False, 'pred_hidden_mult': 4, 'aux_weight': 0.01}  # 禁用 PC
}

import sys
sys.stdout.reconfigure(line_buffering=True)
print('=== Copy Task S=128 (Fixed) ===', flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

vocab_size, seq_len = 10, 128
model = Wrapper(config, vocab_size, seq_len).to(device)
print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params', flush=True)

# 减小 batch size 避免显存问题
train_ds = CopyDataset(vocab_size, seq_len, 5000)
val_ds = CopyDataset(vocab_size, seq_len, 500)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)  # BS=16
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, BS=16', flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)  # 降低学习率
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

print('\nTraining...', flush=True)
best_acc = 0
for epoch in range(15):  # 更多 epoch
    # Train
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        try:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            # 检查 NaN
            if torch.isnan(loss):
                print(f'  [WARN] NaN loss at batch {batch_idx}, skipping', flush=True)
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
        except Exception as e:
            print(f'  [ERROR] Batch {batch_idx}: {e}', flush=True)
            continue
    
    train_acc = correct / total if total > 0 else 0
    
    # Val
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            try:
                logits = model(x)
                pred = logits.argmax(dim=-1)
                val_correct += (pred == y).sum().item()
                val_total += y.numel()
            except:
                continue
    
    val_acc = val_correct / val_total if val_total > 0 else 0
    print(f'Epoch {epoch+1}/15: Train Acc={train_acc*100:.1f}%, Val Acc={val_acc*100:.1f}%', flush=True)
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'config': config,
            'val_acc': val_acc
        }, 'best_s128.pt')
        print(f'  [Saved] Best model: {val_acc*100:.1f}%', flush=True)
    
    # 显存清理
    if device.type == 'cuda':
        torch.cuda.empty_cache()

print(f'\nDone! Best Val Acc: {best_acc*100:.1f}%', flush=True)
