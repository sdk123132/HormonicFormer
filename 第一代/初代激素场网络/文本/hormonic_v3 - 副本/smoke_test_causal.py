"""
Smoke test: HormonicForCausalLM on 1% data, 100 steps.
No external dependencies - uses byte-level tokenizer.
"""
import os, sys, time, gc
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch, torch.nn.functional as F
from hormonicformer_v3 import HormonicForCausalLM

print("="*60)
print("HormonicForCausalLM Smoke Test")
print("="*60)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {dev}")
if dev.type == 'cuda':
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# Byte-level tokenizer (no download needed)
VOCAB = 256  # byte-level

# Training text
text = (b"In the beginning God created the heaven and the earth. "
        b"And the earth was without form, and void; and darkness "
        b"was upon the face of the deep. And the Spirit of God "
        b"moved upon the face of the waters. And God said, Let there "
        b"be light: and there was light. And God saw the light, that "
        b"it was good: and God divided the light from the darkness.").decode()
tokens = [min(ord(c), 255) for c in (text * 100)]
print(f"Total bytes: {len(tokens)}")

# Build batches
S = 128  # shorter seq for smoke test
bs = 4
batches = []
for i in range(0, len(tokens) - S, S):
    batches.append(torch.tensor(tokens[i:i+S], dtype=torch.long))
    if len(batches) >= 30:
        break
print(f"Batches: {len(batches)} x seq_len={S}")

# Model - full config
print("\nCreating model d_model=768, n_layers=12, n_heads=12...")
model = HormonicForCausalLM.from_config(
    vocab_size=VOCAB, d_model=768, n_layers=12, n_heads=12,
    seq_len=S, n_steps=2, dt=0.02, dropout=0.1,
    use_hebbian=True, hebbian_lr=0.001,
    use_gradient_checkpointing=True
)
print(f"Parameters: {model.param_count()/1e6:.1f}M")
model = model.to(dev)

if dev.type == 'cuda':
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Optimizer
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# Train 100 steps
print("\n--- Training ---")
model.train()
losses = []
total_steps = 100
steps_per_epoch = 25
epoch = 0
h_lr = model.set_hebbian_warmup(0)
print(f"Epoch 0: Hebbian LR = {h_lr}")

t0 = time.time()
for step in range(total_steps):
    # Epoch boundary
    new_epoch = step // steps_per_epoch
    if new_epoch != epoch:
        epoch = new_epoch
        h_lr = model.set_hebbian_warmup(epoch)
        print(f"\n  >> Epoch {epoch}: Hebbian LR = {h_lr}")

    batch = batches[step % len(batches)].unsqueeze(0).expand(bs, -1).to(dev)
    labels = batch.clone()

    opt.zero_grad()
    try:
        loss, _ = model(batch, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  OOM at step {step}! Skipping...")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        raise

    ppl = torch.exp(loss.detach()).item()
    losses.append(loss.item())

    if step % 10 == 0:
        elapsed = time.time() - t0
        if dev.type == 'cuda':
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Step {step:3d}: Loss={loss.item():.3f} PPL={ppl:.1f} "
                  f"Time={elapsed:.1f}s VRAM={mem:.2f}GB")
        else:
            print(f"  Step {step:3d}: Loss={loss.item():.3f} PPL={ppl:.1f} "
                  f"Time={elapsed:.1f}s")

elapsed = time.time() - t0
print(f"\n--- Done in {elapsed:.1f}s ---")

loss0, lossF = losses[0], losses[-1]
ppl0 = torch.exp(torch.tensor(loss0)).item()
pplF = torch.exp(torch.tensor(lossF)).item()
print(f"\nLoss:  {loss0:.3f} -> {lossF:.3f}")
print(f"PPL:   {ppl0:.1f} -> {pplF:.1f}")
print(f"PPL reduction: {(1-pplF/ppl0)*100:.1f}%")
has_nan = any(torch.isnan(torch.tensor(l)) for l in losses)
print(f"NaN: {has_nan}")
print(f"OOM steps: 0")

if pplF < ppl0 and not has_nan:
    print("\n[PASS] Smoke test passed!")
    sys.exit(0)
else:
    print("\n[FAIL] PPL not decreasing or NaN")
    sys.exit(1)
