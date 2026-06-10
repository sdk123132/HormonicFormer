"""
Post-50K Analysis:
1. Export PPL curve CSV with Hebbian Warmup phases
2. Inference test
3. Compare with GPT-2 baseline
"""
import os, sys, json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
from transformers import GPT2Tokenizer
from hormonicformer_v3 import HormonicForCausalLM

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 1. Export training curves ===
print("="*60)
print("1. Export PPL Curves")
print("="*60)

hormonic_log = 'wikitext103_train_log.json'
gpt2_log = 'gpt2_train_log.json'

if os.path.exists(hormonic_log):
    with open(hormonic_log) as f:
        h_log = json.load(f)
    print(f"Hormonic: {len(h_log)} eval points")
    
    # CSV export
    with open('hormonic_ppl_curve.csv', 'w') as f:
        f.write('step,val_ppl,hebbian_phase\n')
        for entry in h_log:
            step = entry['step']
            # Hebbian phases based on training schedule
            # Epoch boundary = total_steps / 6
            # E0-2: full, E3-5: decay, E6+: off
            epoch_approx = step // 91135  # ~546K/6
            if epoch_approx <= 2:
                phase = 'full'
            elif epoch_approx <= 5:
                phase = 'decay'
            else:
                phase = 'off'
            f.write(f"{step},{entry['val_ppl']:.1f},{phase}\n")
    print("Saved hormonic_ppl_curve.csv")
else:
    print("No Hormonic log found yet")

if os.path.exists(gpt2_log):
    with open(gpt2_log) as f:
        g_log = json.load(f)
    print(f"GPT-2: {len(g_log)} eval points")
    
    with open('gpt2_ppl_curve.csv', 'w') as f:
        f.write('step,val_ppl\n')
        for entry in g_log:
            f.write(f"{entry['step']},{entry['val_ppl']:.1f}\n")
    print("Saved gpt2_ppl_curve.csv")

# === 2. Compare if both exist ===
if os.path.exists(hormonic_log) and os.path.exists(gpt2_log):
    print("\n" + "="*60)
    print("2. Head-to-Head Comparison")
    print("="*60)
    
    with open(hormonic_log) as f:
        h_log = json.load(f)
    with open(gpt2_log) as f:
        g_log = json.load(f)
    
    # Build step->ppl maps
    h_map = {e['step']: e['val_ppl'] for e in h_log}
    g_map = {e['step']: e['val_ppl'] for e in g_log}
    
    # Compare at common checkpoints
    print(f"\n{'Step':<10}{'Hormonic PPL':<15}{'GPT-2 PPL':<15}{'Winner':<15}")
    print("-"*55)
    for step in sorted(set(list(h_map.keys()) + list(g_map.keys()))):
        if step in h_map and step in g_map:
            h = h_map[step]
            g = g_map[step]
            winner = "Hormonic" if h < g else "GPT-2" if g < h else "Tie"
            print(f"{step:<10}{h:<15.1f}{g:<15.1f}{winner:<15}")
    
    # Best PPL
    h_best = min(e['val_ppl'] for e in h_log)
    g_best = min(e['val_ppl'] for e in g_log)
    print(f"\nBest Val PPL: Hormonic={h_best:.1f}, GPT-2={g_best:.1f}")

# === 3. Inference Test ===
print("\n" + "="*60)
print("3. Inference Test")
print("="*60)

# Find latest checkpoint
ckpts = sorted([f for f in os.listdir('.') if f.startswith('ckpt_step') and f.endswith('.pt')],
               key=lambda x: int(x.split('step')[1].split('.')[0]))

if ckpts:
    latest = ckpts[-1]
    print(f"Loading {latest}...")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = HormonicForCausalLM.from_config(
        vocab_size=tokenizer.vocab_size,
        d_model=768, n_layers=12, n_heads=12,
        seq_len=512, n_steps=2, dt=0.02,
        use_hebbian=True, use_gradient_checkpointing=False
    ).to(dev)
    
    ckpt = torch.load(latest, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded step {ckpt['step']}, best Val PPL: {ckpt.get('best_val_ppl', 'N/A')}")
    
    # Generate
    prompts = [
        "The main difference between",
        "In the beginning",
        "The president of the United States",
        "Scientists have discovered that",
    ]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(dev)
        print(f"\nPrompt: '{prompt}'")
        
        with torch.no_grad():
            generated = input_ids
            for _ in range(50):
                if generated.shape[1] >= 512:
                    break
                logits = model(generated)
                if isinstance(logits, tuple):
                    logits = logits[1] if len(logits) > 1 else logits[0]
                next_token_logits = logits[:, -1, :]
                # Top-k sampling
                top_k = 40
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                probs = torch.softmax(next_token_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        
        output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Output: '{output}'")
else:
    print("No checkpoints found yet")

print("\n" + "="*60)
print("Done!")
print("="*60)
