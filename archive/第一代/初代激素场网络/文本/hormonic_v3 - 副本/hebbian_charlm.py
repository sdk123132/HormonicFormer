"""
Hebbian on Char LM - Quick Test
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print("Step 1: Importing...")
from hormonicformer_v3 import HormonicFormer
print("Import OK")

class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        self.seq_len = seq_len
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        self.n_samples = len(self.data) - seq_len - 1
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


class CharLM(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, cfg):
        super().__init__()
        self.vocab_size = vocab_size
        cfg['model']['seq_len'] = seq_len
        cfg['model']['patch_size'] = 1
        cfg['model']['n_classes'] = vocab_size
        self.model = HormonicFormer(cfg)
        self.classifier = nn.Linear(d_model, vocab_size)
    
    def forward(self, seq):
        B, S = seq.shape
        img = ((seq.float() / (self.vocab_size - 1)) * 2 - 1).view(B, 1, 1, S)
        x = self.model.patch_embed(img).flatten(2).transpose(1, 2)
        for blk in self.model.blocks:
            x = blk(x)
        return self.classifier(x)


# Text
TEXT = ("In the beginning God created the heaven and the earth. And the earth was without form. " * 50).replace('\n', ' ')
print(f"Text: {len(TEXT)} chars")

# Configs
def make_cfg(hebbian):
    return {
        'model': {
            'd_model': 64, 'n_heads': 2, 'n_layers': 2, 'seq_len': 64,
            'n_steps': 2, 'n_classes': 10, 'patch_size': 1,
            'D0_amp': 0.002, 'D0_phase': 0.002, 'dt': 0.02,
            'noise_scale': 0.0, 'dropout': 0.0,
            'ei_balance': {'enabled': True, 'target_ratio': 4.0},
            'sensory_feedback': {'enabled': True, 'top_k': 4},
            'hebbian': {'enabled': hebbian, 'lr': 0.005, 'decay': 0.99},
            'cross_freq_coupling': {'enabled': True},
            'energy_constraint': {'enabled': True}
        },
        'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
        'bwo': {'use_bwo': False}, 'pc': {'use_pc': False}
    }

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {dev}")

# Dataset
ds = CharDataset(TEXT, 64)
n_train = int(0.9 * ds.n_samples)
train_ds = torch.utils.data.Subset(ds, range(n_train))
val_ds = torch.utils.data.Subset(ds, range(n_train, ds.n_samples))
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
print(f"Vocab: {ds.vocab_size}, Train: {len(train_ds)}, Val: {len(val_ds)}")

def train(name, hebbian):
    print(f"\n{'='*50}")
    print(f">>> {name} (Hebbian={hebbian})")
    print(f"{'='*50}")
    
    model = CharLM(ds.vocab_size, 64, 64, make_cfg(hebbian)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e3:.1f}K")
    
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    
    for epoch in range(10):
        # === Hebbian Warmup Schedule (only if blocks have hebbian) ===
        has_hebb = hasattr(model.model.blocks[0], 'hebbian')
        if has_hebb:
            if epoch <= 3:
                h_lr = 0.001
            elif epoch <= 7:
                h_lr = 0.001 * (0.5 ** (epoch - 3))
            else:
                h_lr = 0.0
            for blk in model.model.blocks:
                if hasattr(blk, 'hebbian'):
                    blk.hebbian.eta_hebb = h_lr
                    blk.hebbian.eta_anti = h_lr * 0.5
            if epoch == 0 or h_lr == 0.0 or epoch == 3:
                print(f"  [Heb LR: {h_lr:.5f}]")
        
        # Train
        model.train()
        tl, tc, tt = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, ds.vocab_size), y[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            pred = logits[:, :-1, :].argmax(-1)
            tc += (pred == y[:, 1:]).sum().item()
            tt += y[:, 1:].numel()
        
        train_ppl = torch.exp(torch.tensor(tl / max(1, len(train_loader)))).item()
        train_acc = tc / max(1, tt)
        
        # Val
        model.eval()
        vl, vc, vt = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, ds.vocab_size), y[:, 1:].reshape(-1))
                vl += loss.item()
                pred = logits[:, :-1, :].argmax(-1)
                vc += (pred == y[:, 1:]).sum().item()
                vt += y[:, 1:].numel()
        
        val_ppl = torch.exp(torch.tensor(vl / max(1, len(val_loader)))).item()
        val_acc = vc / max(1, vt)
        
        history.append([train_ppl, val_ppl, train_acc, val_acc])
        print(f"  E{epoch+1}: T-PPL={train_ppl:.2f} T-Acc={train_acc*100:.1f}% | "
              f"V-PPL={val_ppl:.2f} V-Acc={val_acc*100:.1f}%")
    
    return history

res0 = train("No Hebbian", False)
res1 = train("With Hebbian", True)

# Final comparison
print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"\n{'Epoch':<6}{'NoHeb PPL':<12}{'Heb PPL':<12}{'PPL Diff':<12}{'NoHeb Acc':<12}{'Heb Acc':<12}")
print("-"*50)
for i in range(10):
    nh_ppl = res0[i][1]
    h_ppl = res1[i][1]
    diff = nh_ppl - h_ppl
    nh_acc = res0[i][3] * 100
    h_acc = res1[i][3] * 100
    marker = " <-" if h_ppl == min(res1[j][1] for j in range(10)) else ""
    print(f"{i+1:<6}{nh_ppl:<12.2f}{h_ppl:<12.2f}{diff:+.2f}       {nh_acc:<12.1f}{h_acc:<12.1f}{marker}")
print("="*50)
