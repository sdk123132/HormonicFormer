"""
HormonicForCausalLM - WikiText-103 Training
82.9M params, d=768, L=12, H=12
GPT2 tokenizer, Hebbian Warmup, gradient checkpointing.
"""
import os, sys, time, gc, json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from hormonicformer_v3 import HormonicForCausalLM

print("="*60)
print("HormonicForCausalLM - WikiText-103")
print("="*60)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {dev}")
if dev.type == 'cuda':
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# Tokenizer
print("\nLoading GPT2Tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
print(f"Vocab: {tokenizer.vocab_size}")

# Load WikiText-103
print("\nLoading WikiText-103...")
with open('wikitext103_raw.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Text: {len(text)/1e6:.1f}M chars")

# Tokenize (take first 10M chars to keep tokenization fast)
MAX_CHARS = 10_000_000  # 10M chars ~ 2.5M tokens
text = text[:MAX_CHARS]
print(f"Tokenizing {len(text)/1e6:.1f}M chars...")
tokens = tokenizer.encode(text)
print(f"Tokens: {len(tokens):,} ({len(tokens)/1e6:.2f}M)")

# Split train/val
val_tokens = 10000
train_tok = tokens[:-val_tokens]
val_tok = tokens[-val_tokens:]
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


# Config
S = 512
BS = 8
EPOCHS = 2
EVAL_EVERY = 200
SAVE_EVERY = 500

train_ds = TokenDataset(train_tok, S)
val_ds = TokenDataset(val_tok, S)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0, drop_last=True)
print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}")
print(f"Config: seq_len={S}, batch_size={BS}, epochs={EPOCHS}")

# Model
print("\nCreating model...")
model = HormonicForCausalLM.from_config(
    vocab_size=tokenizer.vocab_size,
    d_model=768, n_layers=12, n_heads=12,
    seq_len=S, n_steps=2, dt=0.02, dropout=0.1,
    use_hebbian=True, hebbian_lr=0.001,
    use_gradient_checkpointing=True
)
print(f"Parameters: {model.param_count()/1e6:.1f}M")
model = model.to(dev)

if dev.type == 'cuda':
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Optimizer + scheduler
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
total_steps = len(train_loader) * EPOCHS
warmup_steps = min(500, total_steps // 10)
print(f"Total steps: {total_steps}, LR warmup: {warmup_steps}")

def get_lr(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)


def evaluate():
    model.eval()
    total_loss, n_batches = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(dev), y.to(dev)
            loss, _ = model(x, labels=y)
            total_loss += loss.item()
            n_batches += 1
            if n_batches >= 50:  # cap eval
                break
    model.train()
    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()


# Resume from checkpoint if available
ckpt_path = None
for s in [9500, 9000, 8500, 8000]:
    p = f'ckpt_step{s}.pt'
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path:
    print(f"\nResuming from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['opt'])
    global_step = ckpt['step']
    best_val_ppl = ckpt['best_val_ppl']
    print(f"  Resumed at step {global_step}, best Val PPL: {best_val_ppl:.1f}")
else:
    global_step = 0
    best_val_ppl = float('inf')

# Training
model.train()
log = []
t0 = time.time()

for epoch in range(EPOCHS):
    h_lr = model.set_hebbian_warmup(epoch)
    print(f"\n=== Epoch {epoch} === Hebbian LR: {h_lr:.4f}")
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # Auto-stop at 50K
        if global_step >= 50000:
            print(f"\n--- Reached 50K steps, stopping ---")
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
                print(f"  OOM at step {global_step}!")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise
        
        train_ppl = torch.exp(loss.detach()).item()
        
        if global_step % 50 == 0:
            elapsed = time.time() - t0
            lr_now = opt.param_groups[0]['lr']
            if dev.type == 'cuda':
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  Step {global_step:5d}: Loss={loss.item():.3f} PPL={train_ppl:.1f} "
                      f"LR={lr_now:.2e} H-LR={h_lr:.4f} {elapsed:.0f}s VRAM={mem:.2f}GB")
            else:
                print(f"  Step {global_step:5d}: Loss={loss.item():.3f} PPL={train_ppl:.1f} "
                      f"LR={lr_now:.2e} {elapsed:.0f}s")
        
        if global_step > 0 and global_step % EVAL_EVERY == 0:
            val_loss, val_ppl = evaluate()
            print(f"  >>> Val Loss={val_loss:.3f} Val PPL={val_ppl:.1f} {'*BEST*' if val_ppl < best_val_ppl else ''}")
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
            log.append({'step': global_step, 'train_ppl': train_ppl, 'val_ppl': val_ppl, 'epoch': epoch})
        
        if global_step > 0 and global_step % SAVE_EVERY == 0:
            ckpt_path = f'ckpt_step{global_step}.pt'
            torch.save({
                'step': global_step, 'epoch': epoch,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'best_val_ppl': best_val_ppl
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")
        
        global_step += 1

# Final eval
val_loss, val_ppl = evaluate()
elapsed = time.time() - t0
print(f"\n--- Done in {elapsed:.0f}s ({elapsed/60:.1f}min) ---")
print(f"Final Val PPL: {val_ppl:.1f}")
print(f"Best Val PPL:  {best_val_ppl:.1f}")

if dev.type == 'cuda':
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# Save log
with open('wikitext103_train_log.json', 'w') as f:
    json.dump(log, f, indent=2)
print("Saved wikitext103_train_log.json")
