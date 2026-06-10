"""
HormonicForCausalLM - Large Scale Training
Uses GPT2Tokenizer via hf-mirror.com
Local text data, gradient checkpointing, Hebbian warmup.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import sys, time, gc
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch, torch.nn.functional as F
from transformers import GPT2Tokenizer
from hormonicformer_v3 import HormonicForCausalLM

print("="*60)
print("HormonicForCausalLM - Large Scale Training")
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

# Build training text from multiple local sources
text_sources = []

# 1. Shakespeare (本地)
shakespeare_path = r'C:\Users\MR\Desktop\初代激素场网络\文本\shakespeare.txt'
try:
    with open(shakespeare_path, 'r', encoding='utf-8') as f:
        text_sources.append(f.read())
    print(f"Shakespeare: {len(text_sources[-1])} chars")
except:
    print("Shakespeare not found")

# 2. Genesis (本地)
genesis_path = r'C:\Users\MR\Desktop\初代激素场网络\文本\genesis.txt'
try:
    with open(genesis_path, 'r', encoding='utf-8') as f:
        text_sources.append(f.read())
    print(f"Genesis: {len(text_sources[-1])} chars")
except:
    print("Genesis not found")

# 3. Hamlet (本地)
hamlet_path = r'C:\Users\MR\Desktop\初代激素场网络\text\hamlet.txt'
try:
    with open(hamlet_path, 'r', encoding='utf-8') as f:
        text_sources.append(f.read())
    print(f"Hamlet: {len(text_sources[-1])} chars")
except:
    print("Hamlet not found")

# 4. 如果没有本地文件，用内嵌文本
if not text_sources:
    text_sources.append("""
To be or not to be that is the question Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune Or to take arms against a sea of troubles
And by opposing end them. In the beginning God created the heaven and the earth.
Four score and seven years ago our fathers brought forth on this continent a new nation.
It was the best of times it was the worst of times it was the age of wisdom it was the age of foolishness.
Call me Ishmael. It is a truth universally acknowledged that a single man in possession
of a good fortune must be in want of a wife. All happy families are alike each unhappy
family is unhappy in its own way. The quick brown fox jumps over the lazy dog.
""".strip())

full_text = "\n\n".join(text_sources) * 200  # Repeat for more data
print(f"\nTotal training text: {len(full_text):,} chars ({len(full_text)/1024/1024:.1f} MB)")

# Tokenize
print("Tokenizing...")
tokens = tokenizer.encode(full_text, max_length=512*5000, truncation=True)
print(f"Total tokens: {len(tokens):,} ({len(tokens)/1024:.1f}K)")

# Config
S = 256
bs = 2  # Small batch for 768-dim model with 8GB VRAM
n_samples = len(tokens) - S - 1
print(f"\nPossible training samples: {n_samples:,}")
print(f"Config: seq_len={S}, batch_size={bs}")

# Model - full size
print("\nCreating model d_model=768, n_layers=12, n_heads=12...")
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

# Optimizer
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# Train
print("\n--- Training 1 epoch ---")
model.train()
losses = []
total_steps = min(n_samples // bs, 2000)  # Cap at 2000 steps
print(f"Total steps: {total_steps}")
print(f"Hebbian warmup: E0-2=full E3-5=decay E6+=off")

t0 = time.time()
h_lr = model.set_hebbian_warmup(0)
print(f"  Epoch 0: Hebbian LR = {h_lr:.4f}")

for step in range(total_steps):
    # Hebbian warmup at epoch boundary
    new_epoch = step // (total_steps // 6)  # ~6 epochs within 1 pass
    current_epoch = min(new_epoch, 6)
    lr = model.set_hebbian_warmup(current_epoch)
    
    # Get batch
    idx = torch.randint(0, n_samples, (bs,))
    batch = torch.stack([torch.tensor(tokens[i:i+S], dtype=torch.long) for i in idx]).to(dev)
    labels = batch.clone()
    
    opt.zero_grad()
    try:
        loss, _ = model(batch, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  OOM at step {step}! Clearing cache...")
            torch.cuda.empty_cache()
            gc.collect()
            # Reduce batch
            bs = max(1, bs // 2)
            continue
        raise
    
    losses.append(loss.item())
    
    if step % 50 == 0:
        elapsed = time.time() - t0
        ppl = torch.exp(torch.tensor(losses[-1])).item()
        if dev.type == 'cuda':
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Step {step:4d}: Loss={losses[-1]:.3f} PPL={ppl:.1f} "
                  f"LR={lr:.4f} Time={elapsed:.0f}s VRAM={mem:.2f}GB")
        else:
            print(f"  Step {step:4d}: Loss={losses[-1]:.3f} PPL={ppl:.1f} "
                  f"LR={lr:.4f} Time={elapsed:.0f}s")

elapsed = time.time() - t0
print(f"\n--- Done in {elapsed:.0f}s ({elapsed/60:.1f}min) ---")

loss0, lossF = losses[0], losses[-1]
ppl0 = torch.exp(torch.tensor(loss0)).item()
pplF = torch.exp(torch.tensor(lossF)).item()
print(f"\nLoss:  {loss0:.3f} -> {lossF:.3f}")
print(f"PPL:   {ppl0:.1f} -> {pplF:.1f}")
print(f"PPL reduction: {(1-pplF/ppl0)*100:.1f}%")
has_nan = any(torch.isnan(torch.tensor(l)) for l in losses)
print(f"NaN: {has_nan}")

if dev.type == 'cuda':
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

if pplF < ppl0 and not has_nan:
    print("\n[PASS] Large scale training passed!")
else:
    print("\n[FAIL] Something went wrong.")
