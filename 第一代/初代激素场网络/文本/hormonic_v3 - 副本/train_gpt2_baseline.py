"""
GPT-2 100M Baseline - Same data, same steps, same config.
Fair comparison with HormonicForCausalLM.
"""
import os, sys, time, json, gc
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

print("="*60)
print("GPT-2 100M Baseline - WikiText-103")
print("="*60)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {dev}")
if dev.type == 'cuda':
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# Tokenizer
print("\nLoading GPT2Tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load WikiText-103
print("Loading WikiText-103...")
with open('wikitext103_raw.txt', 'r', encoding='utf-8') as f:
    text = f.read()

MAX_CHARS = 10_000_000
text = text[:MAX_CHARS]
print(f"Tokenizing {len(text)/1e6:.1f}M chars...")
tokens = tokenizer.encode(text)
print(f"Tokens: {len(tokens):,}")

val_tokens = 10000
train_tok = tokens[:-val_tokens]
val_tok = tokens[-val_tokens:]


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


S = 512
BS = 8
MAX_STEPS = 22500

train_ds = TokenDataset(train_tok, S)
val_ds = TokenDataset(val_tok, S)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0, drop_last=True)

# GPT-2 100M (matched to HormonicForCausalLM)
print("\nCreating GPT-2 100M...")
config = GPT2Config(
    n_embd=768, n_layer=12, n_head=12,
    vocab_size=tokenizer.vocab_size,
    n_positions=S,
    resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1
)
model = GPT2LMHeadModel(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params/1e6:.1f}M")
model = model.to(dev)

if dev.type == 'cuda':
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Same optimizer as Hormonic
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
warmup_steps = 500

def get_lr(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, MAX_STEPS - warmup_steps)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)


def evaluate():
    model.eval()
    total_loss, n = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(dev), y.to(dev)
            out = model(x, labels=y)
            total_loss += out.loss.item()
            n += 1
            if n >= 50:
                break
    model.train()
    avg = total_loss / max(1, n)
    return avg, torch.exp(torch.tensor(avg)).item()


print(f"\n--- Training {MAX_STEPS} steps (BS={BS}, seq_len={S}) ---")
model.train()
log = []
best_val_ppl = float('inf')
t0 = time.time()
step = 0

for epoch in range(100):  # will break at MAX_STEPS
    for x, y in train_loader:
        if step >= MAX_STEPS:
            break
        x, y = x.to(dev), y.to(dev)

        opt.zero_grad()
        out = model(x, labels=y)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % 50 == 0:
            elapsed = time.time() - t0
            ppl = torch.exp(loss.detach()).item()
            lr_now = opt.param_groups[0]['lr']
            if dev.type == 'cuda':
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  Step {step:5d}: Loss={loss.item():.3f} PPL={ppl:.1f} "
                      f"LR={lr_now:.2e} {elapsed:.0f}s VRAM={mem:.2f}GB")
            else:
                print(f"  Step {step:5d}: Loss={loss.item():.3f} PPL={ppl:.1f} {elapsed:.0f}s")

        if step > 0 and step % 200 == 0:
            val_loss, val_ppl = evaluate()
            tag = '*BEST*' if val_ppl < best_val_ppl else ''
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
            print(f"  >>> Val Loss={val_loss:.3f} Val PPL={val_ppl:.1f} {tag}")
            log.append({'step': step, 'train_loss': loss.item(), 'val_ppl': val_ppl})

        if step > 0 and step % 2000 == 0:
            torch.save({'step': step, 'model': model.state_dict(),
                        'best_val_ppl': best_val_ppl},
                       f'gpt2_ckpt_step{step}.pt')
            print(f"  Saved gpt2_ckpt_step{step}.pt")

        step += 1
    if step >= MAX_STEPS:
        break

elapsed = time.time() - t0
val_loss, val_ppl = evaluate()
print(f"\n--- Done in {elapsed:.0f}s ({elapsed/60:.1f}min) ---")
print(f"Final Val PPL: {val_ppl:.1f}")
print(f"Best Val PPL:  {best_val_ppl:.1f}")
if dev.type == 'cuda':
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

with open('gpt2_train_log.json', 'w') as f:
    json.dump(log, f, indent=2)
print("Saved gpt2_train_log.json")
