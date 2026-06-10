"""
HormonicFormer Paper Experiment — 3-Way Comparison on WikiText-103
All on DCU K100_AI 64GB. Full scale, no miniaturization.

Run 1: Hormonic + Hebbian Warmup (107M params)
Run 2: Hormonic - No Hebbian (107M params)
Run 3: GPT-2 Transformer baseline (124M params)

Same data (118M tokens), same LR, same 20K steps.
"""
import os, sys, time, gc, json
import numpy as np
sys.path.insert(0, '/root/private_data/hormonic_v3/models')
sys.path.insert(0, '/root/private_data/hormonic_v3/field')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from hormonicformer_v3 import HormonicForCausalLM

print("="*70)
print("HormonicFormer Paper Experiment — 3-Way WikiText-103")
print("="*70)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {dev}")
if dev.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# Data
print("\nLoading pre-tokenized WikiText-103...")
all_tokens = np.load('/root/private_data/hormonic_v3/wikitext103_tokens.npy').astype(np.int32)
print(f"Total tokens: {len(all_tokens):,} ({len(all_tokens)/1e6:.1f}M)")

val_size = 500_000
train_tok = all_tokens[:-val_size].tolist()
val_tok = all_tokens[-val_size:].tolist()
print(f"Train: {len(train_tok):,}, Val: {len(val_tok):,}")

# Tokenizer (for vocab size only)
tokenizer = GPT2Tokenizer.from_pretrained('/root/private_data/hormonic_v3/gpt2_tokenizer')
VOCAB = tokenizer.vocab_size
print(f"Vocab: {VOCAB}")


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = np.array(tokens, dtype=np.int32)  # numpy for faster slice
        self.seq_len = seq_len
        self.n = len(tokens) - seq_len - 1
        print(f"  Dataset: {self.n:,} samples")
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return x, y


# Config
S = 1024
BS = 48
MAX_STEPS = 20000
EVAL_EVERY = 1000
LR = 3e-4
WARMUP_STEPS = 500

train_ds = TokenDataset(train_tok, S)
val_ds = TokenDataset(val_tok, S)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=0, drop_last=True)
print(f"\nConfig: seq_len={S}, BS={BS}, steps={MAX_STEPS}, LR={LR}")


def make_scheduler(optimizer):
    def get_lr(step):
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return max(0.1, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)


def evaluate(model, is_gpt2=False, max_batches=100):
    model.eval()
    total_loss, n = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(dev), y.to(dev)
            if is_gpt2:
                out = model(x, labels=y)
                loss = out.loss
            else:
                loss, _ = model(x, labels=y)
            total_loss += loss.item()
            n += 1
            if n >= max_batches:
                break
    model.train()
    avg = total_loss / max(1, n)
    return avg, torch.exp(torch.tensor(avg)).item()


def train_model(model, name, is_gpt2=False, use_hebbian_warmup=False):
    """Train one model for MAX_STEPS and record metrics."""
    print(f"\n{'='*70}")
    print(f"Training: {name}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M")
    print(f"{'='*70}")
    
    model = model.to(dev)
    torch.cuda.reset_peak_memory_stats()
    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = make_scheduler(opt)
    
    model.train()
    log = []
    best_val_ppl = float('inf')
    t0 = time.time()
    step = 0
    
    for epoch in range(100):
        if use_hebbian_warmup and hasattr(model, 'set_hebbian_warmup'):
            h_lr = model.set_hebbian_warmup(epoch)
            print(f"  Epoch {epoch}: Hebbian LR = {h_lr:.4f}")
        
        for x, y in train_loader:
            if step >= MAX_STEPS:
                break
            x, y = x.to(dev), y.to(dev)
            
            opt.zero_grad()
            try:
                if is_gpt2:
                    out = model(x, labels=y)
                    loss = out.loss
                else:
                    loss, _ = model(x, labels=y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  OOM at step {step}!")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
            
            train_ppl = torch.exp(loss.detach()).item()
            
            if step % 200 == 0:
                elapsed = time.time() - t0
                lr_now = opt.param_groups[0]['lr']
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  [{name}] Step {step:5d}: Loss={loss.item():.3f} PPL={train_ppl:.1f} "
                      f"LR={lr_now:.2e} {elapsed:.0f}s VRAM={mem:.1f}GB")
            
            if step > 0 and step % EVAL_EVERY == 0:
                val_loss, val_ppl = evaluate(model, is_gpt2)
                tag = '*BEST*' if val_ppl < best_val_ppl else ''
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                print(f"  [{name}] >>> Val Loss={val_loss:.3f} Val PPL={val_ppl:.1f} {tag}")
                log.append({
                    'step': step, 'train_loss': loss.item(),
                    'train_ppl': train_ppl, 'val_loss': val_loss,
                    'val_ppl': val_ppl
                })
            
            step += 1
        if step >= MAX_STEPS:
            break
    
    # Final eval
    val_loss, val_ppl = evaluate(model, is_gpt2)
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    result = {
        'name': name,
        'params_M': n_params / 1e6,
        'best_val_ppl': min(best_val_ppl, val_ppl),
        'final_val_ppl': val_ppl,
        'peak_vram_gb': peak_mem,
        'total_time_s': elapsed,
        'steps': step,
        'log': log
    }
    
    print(f"\n  [{name}] Done in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  [{name}] Final Val PPL: {val_ppl:.1f}, Best: {result['best_val_ppl']:.1f}")
    print(f"  [{name}] Peak VRAM: {peak_mem:.1f} GB")
    
    # Save checkpoint
    torch.save({
        'model': model.state_dict(),
        'result': result
    }, f'/root/private_data/hormonic_v3/ckpt_{name.replace(" ", "_")}.pt')
    
    # Free memory
    del model, opt, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    return result


# ====================================================================
# Run 1: Hormonic + Hebbian Warmup
# ====================================================================
print("\n" + "█"*70)
print("RUN 1: Hormonic + Hebbian Warmup")
print("█"*70)

model1 = HormonicForCausalLM.from_config(
    vocab_size=VOCAB,
    d_model=768, n_layers=12, n_heads=12,
    seq_len=S, n_steps=2, dt=0.02, dropout=0.1,
    use_hebbian=True, hebbian_lr=0.001,
    use_gradient_checkpointing=True
)
result1 = train_model(model1, "Hormonic+Hebbian", is_gpt2=False, use_hebbian_warmup=True)


# ====================================================================
# Run 2: Hormonic - No Hebbian
# ====================================================================
print("\n" + "█"*70)
print("RUN 2: Hormonic - No Hebbian")
print("█"*70)

model2 = HormonicForCausalLM.from_config(
    vocab_size=VOCAB,
    d_model=768, n_layers=12, n_heads=12,
    seq_len=S, n_steps=2, dt=0.02, dropout=0.1,
    use_hebbian=False, hebbian_lr=0.0,
    use_gradient_checkpointing=True
)
result2 = train_model(model2, "Hormonic-NoHebb", is_gpt2=False, use_hebbian_warmup=False)


# ====================================================================
# Run 3: GPT-2 Transformer (124M)
# ====================================================================
print("\n" + "█"*70)
print("RUN 3: GPT-2 Transformer Baseline")
print("█"*70)

config = GPT2Config(
    n_embd=768, n_layer=12, n_head=12,
    vocab_size=VOCAB,
    n_positions=S,
    resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1
)
model3 = GPT2LMHeadModel(config)
result3 = train_model(model3, "GPT2-124M", is_gpt2=True, use_hebbian_warmup=False)


# ====================================================================
# Final Comparison
# ====================================================================
print("\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)

results = [result1, result2, result3]
print(f"\n{'Model':<25}{'Params':<10}{'Best PPL':<12}{'Final PPL':<12}{'VRAM(GB)':<10}{'Time(h)':<10}")
print("-"*70)
for r in results:
    print(f"{r['name']:<25}{r['params_M']:<10.1f}{r['best_val_ppl']:<12.1f}"
          f"{r['final_val_ppl']:<12.1f}{r['peak_vram_gb']:<10.1f}{r['total_time_s']/3600:<10.1f}")

# Save all results
with open('/root/private_data/hormonic_v3/paper_experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved paper_experiment_results.json")
print("="*70)
