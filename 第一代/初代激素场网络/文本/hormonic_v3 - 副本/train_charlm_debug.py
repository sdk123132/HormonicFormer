"""
Char LM with error handling and memory monitoring
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

class CharLMWrapper(nn.Module):
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

print('=== Char LM with Monitoring (S=64) ===')
print(f'Text length: {len(text)}')

seq_len = 64
ds = CharDataset(text, seq_len)
print(f'Dataset: {len(ds)} samples, vocab={ds.vocab_size}')

# 数据加载器
train_size = int(0.9 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # BS=16 更安全
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# 配置（禁用 BWO 和 Hebbian 避免数值问题）
config = {
    'model': {
        'd_model': 64, 'n_heads': 4, 'n_layers': 2, 'seq_len': seq_len, 'n_steps': 2,
        'n_classes': ds.vocab_size, 'patch_size': 1, 'D0_amp': 0.002, 'D0_phase': 0.002,
        'dt': 0.02, 'noise_scale': 0.0, 'dropout': 0.1,
        'ei_balance': {'enabled': False}, 'sensory_feedback': {'enabled': False},
        'hebbian': {'enabled': False},  # 禁用 Hebbian
        'cross_freq_coupling': {'enabled': False},
        'energy_constraint': {'enabled': False}
    },
    'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
    'bwo': {'use_bwo': False},  # 禁用 BWO
    'pc': {'use_pc': False}
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')

model = CharLMWrapper(config, ds.vocab_size, seq_len).to(device)
print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.3f}M params')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 降低学习率

print('\nTraining 10 epochs with monitoring...')
print('='*60)

for epoch in range(10):
    # Train
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # 确保数据有效
        if x is None or y is None:
            print(f'[WARN] Epoch {epoch+1}, Batch {batch_idx}: None data')
            continue
        if x.size(0) == 0 or y.size(0) == 0:
            print(f'[WARN] Epoch {epoch+1}, Batch {batch_idx}: Empty batch')
            continue
        
        try:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            
            # Next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = y[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, ds.vocab_size),
                shift_targets.view(-1)
            )
            
            # NaN 检查
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'[WARN] Epoch {epoch+1}, Batch {batch_idx}: NaN/Inf loss, skipping')
                continue
            
            loss.backward()
            
            # 梯度检查
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > 10:
                print(f'[WARN] Epoch {epoch+1}, Batch {batch_idx}: Large grad norm {total_norm:.2f}')
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = shift_logits.argmax(dim=-1)
            correct += (pred == shift_targets).sum().item()
            total += shift_targets.numel()
            
            # 显存监控
            if batch_idx % 50 == 0 and device.type == 'cuda':
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                print(f'  Batch {batch_idx}: Loss={loss.item():.3f}, GPU={allocated:.0f}MB/{reserved:.0f}MB')
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f'[ERROR] OOM at Epoch {epoch+1}, Batch {batch_idx}')
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
                
        except Exception as e:
            print(f'[ERROR] Epoch {epoch+1}, Batch {batch_idx}: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    train_acc = correct / total if total > 0 else 0
    
    # Val
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            try:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                shift_logits = logits[:, :-1, :]
                shift_targets = y[:, 1:]
                pred = shift_logits.argmax(dim=-1)
                val_correct += (pred == shift_targets).sum().item()
                val_total += shift_targets.numel()
            except Exception as e:
                print(f'[ERROR] Val batch: {e}')
                continue
    
    val_acc = val_correct / val_total if val_total > 0 else 0
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
    ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 10 else float('inf')
    
    print(f'Epoch {epoch+1}: Loss={avg_loss:.3f}, PPL={ppl:.1f}, Train Acc={train_acc*100:.1f}%, Val Acc={val_acc*100:.1f}%')
    print('-'*60)
    
    # 显存清理
    if device.type == 'cuda':
        torch.cuda.empty_cache()

print('\n' + '='*60)
print('Training completed!')
