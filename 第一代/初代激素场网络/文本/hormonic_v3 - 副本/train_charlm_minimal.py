"""
Char LM - 最小化配置，完整异常捕获
目标：跑通 10 个 epoch，观察 PPL 趋势
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hormonicformer_v3 import HormonicFormer
import traceback


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
        return max(0, len(self.data) - self.seq_len - 1)
    
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

print('=== Char LM (Minimal Config) ===')
print(f'Text length: {len(text)}')

# 最小配置
seq_len = 32  # 减小序列长度
batch_size = 8  # 减小 batch size

ds = CharDataset(text, seq_len)
print(f'Dataset: {len(ds)} samples, vocab={ds.vocab_size}')

# 数据加载器
train_size = int(0.9 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, BS={batch_size}')

# 最小模型配置
config = {
    'model': {
        'd_model': 32,  # 减小模型
        'n_heads': 2,   # 减小头数
        'n_layers': 1,  # 单层
        'seq_len': seq_len,
        'n_steps': 2,
        'n_classes': ds.vocab_size,
        'patch_size': 1,
        'D0_amp': 0.002,
        'D0_phase': 0.002,
        'dt': 0.02,
        'noise_scale': 0.0,
        'dropout': 0.0,  # 禁用 dropout
        'ei_balance': {'enabled': False},
        'sensory_feedback': {'enabled': False},
        'hebbian': {'enabled': False},
        'cross_freq_coupling': {'enabled': False},
        'energy_constraint': {'enabled': False}
    },
    'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
    'bwo': {'use_bwo': False},
    'pc': {'use_pc': False}
}

# 强制使用 CPU
device = torch.device('cpu')
print(f'Device: {device}')
print('Using CPU for training')

model = CharLMWrapper(config, ds.vocab_size, seq_len).to(device)
print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.3f}M params')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('\nTraining 10 epochs...')
print('='*60)

best_ppl = float('inf')

for epoch in range(10):
    print(f'\nEpoch {epoch+1} start')
    
    # CPU training, no cache to clear
    
    # 手动迭代 DataLoader
    train_iter = iter(train_loader)
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    batch_idx = 0
    while True:
        # 获取数据
        try:
            x, y = next(train_iter)
        except StopIteration:
            break
        except Exception as e:
            print(f'[ERROR] DataLoader at epoch {epoch+1}, batch {batch_idx}: {e}')
            traceback.print_exc()
            break
        
        # 训练
        try:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            
            # Next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = y[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.reshape(-1, ds.vocab_size),
                shift_targets.reshape(-1)
            )
            
            # NaN/Inf 检查
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'[WARN] Epoch {epoch+1}, Batch {batch_idx}: NaN/Inf loss, skipping')
                batch_idx += 1
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = shift_logits.argmax(dim=-1)
            correct += (pred == shift_targets).sum().item()
            total += shift_targets.numel()
            
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx}: Loss={loss.item():.3f}')
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f'[ERROR] OOM at epoch {epoch+1}, batch {batch_idx}')
                batch_idx += 1
                continue
            else:
                print(f'[ERROR] RuntimeError at epoch {epoch+1}, batch {batch_idx}: {e}')
                traceback.print_exc()
                raise
        except Exception as e:
            print(f'[ERROR] Other error at epoch {epoch+1}, batch {batch_idx}: {e}')
            traceback.print_exc()
            raise
        
        batch_idx += 1
    
    train_acc = correct / total if total > 0 else 0
    avg_loss = total_loss / batch_idx if batch_idx > 0 else float('inf')
    
    # Validation
    print(f'  Running validation...')
    model.eval()
    val_correct, val_total = 0, 0
    val_loss_total = 0
    val_batches = 0
    
    with torch.no_grad():
        val_iter = iter(val_loader)
        while True:
            try:
                x, y = next(val_iter)
            except StopIteration:
                break
            
            try:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                
                shift_logits = logits[:, :-1, :]
                shift_targets = y[:, 1:]
                
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, ds.vocab_size),
                    shift_targets.reshape(-1)
                )
                
                val_loss_total += loss.item()
                pred = shift_logits.argmax(dim=-1)
                val_correct += (pred == shift_targets).sum().item()
                val_total += shift_targets.numel()
                val_batches += 1
            except Exception as e:
                print(f'[ERROR] Val batch: {e}')
                continue
    
    val_acc = val_correct / val_total if val_total > 0 else 0
    val_avg_loss = val_loss_total / val_batches if val_batches > 0 else float('inf')
    
    # 计算 PPL
    try:
        train_ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 10 else float('inf')
        val_ppl = torch.exp(torch.tensor(val_avg_loss)).item() if val_avg_loss < 10 else float('inf')
    except:
        train_ppl = float('inf')
        val_ppl = float('inf')
    
    print(f'Epoch {epoch+1} Summary:')
    print(f'  Train: Loss={avg_loss:.3f}, PPL={train_ppl:.1f}, Acc={train_acc*100:.1f}%')
    print(f'  Val:   Loss={val_avg_loss:.3f}, PPL={val_ppl:.1f}, Acc={val_acc*100:.1f}%')
    
    if val_ppl < best_ppl and val_ppl < 1000:
        best_ppl = val_ppl
        print(f'  [Best PPL: {best_ppl:.1f}]')
    
    print('-'*60)

print('\n' + '='*60)
print(f'Training completed! Best Val PPL: {best_ppl:.1f}')
print('='*60)
