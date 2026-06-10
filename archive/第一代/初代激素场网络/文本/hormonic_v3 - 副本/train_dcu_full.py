"""
HormonicForCausalLM - DCU Full Scale Training
K100_AI 64GB, full WikiText-103, BS=64, 100K steps
"""
import os, sys, time, gc, json
import numpy as np
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
sys.path.insert(0, '/root/private_data/hormonic_v3/models')
sys.path.insert(0, '/root/private_data/hormonic_v3/field')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from hormonicformer_v3 import HormonicForCausalLM

print("="*60)
print("HormonicForCausalLM - DCU Full Scale")
print("="*60)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {dev}")
if dev.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# Tokenizer (local, no network needed)
print("\nLoading GPT2Tokenizer (local)...")
tokenizer = GPT2Tokenizer.from_pretrained('/root/private_data/hormonic_v3/gpt2_tokenizer')
tokenizer.pad_token = tokenizer.eos_token
print(f"Vocab: {tokenizer.vocab_size}")

# Load pre-tokenized data (no network needed)
print("\nLoading pre-tokenized WikiText-103...")
all_tokens = np.load('/root/private_data/hormonic_v3/wikitext103_tokens.npy')
all_tokens = all_tokens.astype(np.int32)
print(f"Total tokens: {len(all_tokens):,} ({len(all_tokens)/1e6:.1f}M)")

# Split train/val (last 500K for val)
val_size = 500_000
train_tok = all_tokens[:-val_size].tolist()
val_tok = all_tokens[-val_size:].tolist()
print(f"Train: {len(train_tok):,}, Val: {len(val_tok):,}")


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n = len(tokens) - seq_len - 1
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# Config - leverage 64GB VRAM
S = 1024        # longer context
BS = 32         # large batch
EPOCHS = 3
MAX_STEPS = 100000
EVAL_EVERY = 500
SAVE_EVERY = 5000
LR = 3e-4

train_ds = TokenDataset(train_tok, S)
val_ds = TokenDataset(val_tok, S)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=2, drop_last=True)
print(f"\nTrain batches: {len(train_loader):,}, Val batches: {len(val_loader):,}")
print(f"Config: seq_len={S}, batch_size={BS}, max_steps={MAX_STEPS}")

# Model - full size
print("\nCreating model (d_model=768, n_layers=12, n_heads=12)...")
model = HormonicForCausalLM.from_config(
    vocab_size=tokenizer.vocab_size,
    d_model=768, n_layers=12, n_heads=12,
    seq_len=S, n_steps=2, dt=0.02, dropout=0.1,
    use_hebbian=True, hebbian_lr=0.001,
    use_gradient_checkpointing=True
)
n_params = model.param_count()
print(f"Parameters: {n_params/1e6:.1f}M ({n_params:,})")
model = model.to(dev)

if dev.type == 'cuda':
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Optimizer + cosine scheduler
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
warmup_steps = 1000

def get_lr(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, MAX_STEPS - warmup_steps)
    return max(0.1, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item())

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)


def evaluate(max_batches=100):
    model.eval()
    total_loss, n = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(dev), y.to(dev)
            loss, _ = model(x, labels=y)
            total_loss += loss.item()
            n += 1
            if n >= max_batches:
                break
    model.train()
    avg = total_loss / max(1, n)
    return avg, torch.exp(torch.tensor(avg)).item()


# Training
print(f"\n{'='*60}")
print(f"Training {MAX_STEPS} steps")
print(f"Hebbian Warmup: E0-2=full, E3-5=decay, E6+=off")
print(f"{'='*60}")

model.train()
log = []
global_step = 0
best_val_ppl = float('inf')
t0 = time.time()

for epoch in range(EPOCHS):
    h_lr = model.set_hebbian_warmup(epoch)
    print(f"\n=== Epoch {epoch} === Hebbian LR: {h_lr:.4f}")
    
    for batch_idx, (x, y) in enumerate(train_loader):
        if global_step >= MAX_STEPS:
            break
        
        x, y = x.to(dev), y.to(dev)
        
        opt.zero_grad()
        try:
            loss, _ = model(x, labels=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  OOM at step {global_step}! Clearing cache...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise
        
        train_ppl = torch.exp(loss.detach()).item()
        
        if global_step % 100 == 0:
            elapsed = time.time() - t0
            lr_now = opt.param_groups[0]['lr']
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else 0
            print(f"  Step {global_step:6d}: Loss={loss.item():.3f} PPL={train_ppl:.1f} "
                  f"LR={lr_now:.2e} H-LR={h_lr:.4f} {elapsed:.0f}s VRAM={mem:.1f}GB")
        
        if global_step > 0 and global_step % EVAL_EVERY == 0:
            val_loss, val_ppl = evaluate()
            tag = '*BEST*' if val_ppl < best_val_ppl else ''
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
            print(f"  >>> Val Loss={val_loss:.3f} Val PPL={val_ppl:.1f} {tag}")
            log.append({
                'step': global_step, 'epoch': epoch,
                'train_loss': loss.item(), 'train_ppl': train_ppl,
                'val_loss': val_loss, 'val_ppl': val_ppl,
                'h_lr': h_lr, 'lr': lr_now
            })
        
        if global_step > 0 and global_step % SAVE_EVERY == 0:
            ckpt_path = f'/root/private_data/hormonic_v3/ckpt_step{global_step}.pt'
            torch.save({
                'step': global_step, 'epoch': epoch,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'best_val_ppl': best_val_ppl,
                'log': log
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")
            # Also save log
            with open('/root/private_data/hormonic_v3/train_log.json', 'w') as f:
                json.dump(log, f, indent=2)
        
        global_step += 1
    
    if global_step >= MAX_STEPS:
        break

# Final
elapsed = time.time() - t0
val_loss, val_ppl = evaluate()
print(f"\n{'='*60}")
print(f"Done in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
print(f"Final Val PPL: {val_ppl:.1f}")
print(f"Best Val PPL:  {best_val_ppl:.1f}")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

# Save final
torch.save({
    'step': global_step, 'model': model.state_dict(),
    'best_val_ppl': best_val_ppl, 'log': log
}, '/root/private_data/hormonic_v3/ckpt_final.pt')
with open('/root/private_data/hormonic_v3/train_log.json', 'w') as f:
    json.dump(log, f, indent=2)
print("Saved final checkpoint and log.")
