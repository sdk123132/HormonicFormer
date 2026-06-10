"""
HormonicCharLM - 使用本地数据或国内镜像
"""
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hormonicformer_v3 import HormonicFormer
import os


class CharDataset(Dataset):
    def __init__(self, text, seq_len, train=True, train_split=0.9):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        n = int(train_split * len(data))
        self.data = data[:n] if train else data[n:]
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)
    
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
        self.model.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, vocab_size)
        )
    
    def forward(self, input_ids, targets=None):
        B, S = input_ids.shape
        
        x = self.embed(input_ids)  # [B, S, d_model]
        # 构造伪图像 [B, 1, 1, S] - 使用 d_model 作为"高度"，但需要 1 通道
        # 方案：将嵌入投影到标量值，类似 Copy Task
        vals = x.mean(dim=-1, keepdim=True)  # [B, S, 1] - 平均池化到标量
        vals = vals.squeeze(-1)  # [B, S]
        img = vals.view(B, 1, 1, S)  # [B, 1, 1, S] - 单通道图像
        
        x = self.model.patch_embed(img)  # [B, d_model, 1, S]
        x = x.flatten(2).transpose(1, 2)  # [B, S, d_model]
        
        x_embed = x.detach().clone()
        for block in self.model.blocks:
            x = block(x, x_embed=x_embed if block.use_feedback else None)
        
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


def get_sample_text():
    """使用示例文本（如果下载失败）"""
    sample = """To be, or not to be, that is the question:
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
That makes calamity of so long life;
For who would bear the whips and scorns of time,
The oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office and the spurns
That patient merit of the unworthy takes,
When he himself might his quietus make
With a bare bodkin? who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all;
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry,
And lose the name of action."""
    return sample * 10  # 重复10次增加数据量


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    correct = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        logits, loss = model(x, y)
        
        if torch.isnan(loss):
            print('[WARN] NaN loss, skipping')
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * (x.size(1) - 1)
        total_tokens += (x.size(0) * (x.size(1) - 1))
        
        shift_logits = logits[:, :-1, :]
        shift_targets = y[:, 1:]
        pred = shift_logits.argmax(dim=-1)
        correct += (pred == shift_targets).sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 10 else float('inf')
    acc = correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, ppl, acc


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        
        total_loss += loss.item() * (x.size(1) - 1)
        total_tokens += (x.size(0) * (x.size(1) - 1))
        
        shift_logits = logits[:, :-1, :]
        shift_targets = y[:, 1:]
        pred = shift_logits.argmax(dim=-1)
        correct += (pred == shift_targets).sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 10 else float('inf')
    acc = correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, ppl, acc


def main():
    print('=== HormonicCharLM ===')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    seq_len = 128
    
    config = {
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'seq_len': seq_len,
            'n_steps': 3,
            'n_classes': 65,
            'patch_size': 1,
            'D0_amp': 0.002,
            'D0_phase': 0.002,
            'dt': 0.02,
            'noise_scale': 0.0,
            'dropout': 0.1,
            'ei_balance': {'enabled': True, 'tau_e': 2.0, 'tau_i': 1.0, 'gamma_e': 1.0, 'gamma_i': 0.8, 'w_inh': 0.3, 'inh_radius': 3},
            'sensory_feedback': {'enabled': True, 'feedback_strength': 0.3, 'feedback_freq': 1},
            'hebbian': {'enabled': True, 'eta_hebb': 0.001, 'eta_anti': 0.0005, 'sync_threshold': 0.5, 'tau_hebb': 10.0, 'decay': 0.999},
            'cross_freq_coupling': {'enabled': False},
            'energy_constraint': {'enabled': False}
        },
        'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
        'bwo': {'use_bwo': True, 'evolve_interval': 5, 'flip_ratio': 0.3},
        'pc': {'use_pc': False, 'pred_hidden_mult': 4, 'aux_weight': 0.01}
    }
    
    # 使用本地示例文本
    text = get_sample_text()
    print(f'\nDataset: Sample text (Hamlet)')
    print(f'  Total chars: {len(text)}')
    
    train_ds = CharDataset(text, seq_len, train=True)
    val_ds = CharDataset(text, seq_len, train=False)
    vocab_size = train_ds.vocab_size
    
    print(f'  Vocab size: {vocab_size}')
    print(f'  Train samples: {len(train_ds)}')
    print(f'  Val samples: {len(val_ds)}')
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    
    config['model']['n_classes'] = vocab_size
    
    model = HormonicCharLM(config, vocab_size, seq_len).to(device)
    print(f'\nModel: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    
    print('\nTraining 10 epochs...')
    best_ppl = float('inf')
    
    for epoch in range(10):
        train_loss, train_ppl, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_ppl, val_acc = evaluate(model, val_loader, device)
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train: Loss={train_loss:.3f}, PPL={train_ppl:.1f}, Acc={train_acc*100:.1f}%')
        print(f'  Val:   Loss={val_loss:.3f}, PPL={val_ppl:.1f}, Acc={val_acc*100:.1f}%')
        
        if val_ppl < best_ppl and val_ppl < 1000:
            best_ppl = val_ppl
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'vocab': {'stoi': train_ds.stoi, 'itos': train_ds.itos},
                'config': config
            }, 'best_charlm.pt')
    
    print(f'\nBest Val PPL: {best_ppl:.1f}')


if __name__ == '__main__':
    main()
