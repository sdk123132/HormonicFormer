import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
import yaml
from hormonicformer_v3 import HormonicFormer

with open('local_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

device = 'cuda'
model = HormonicFormer(config).to(device)

img = torch.randn(4, 1, 28, 28).to(device)
label = torch.randint(0, 10, (4,)).to(device)

# Test 1: 无 AMP，纯 float32
print("=== Test 1: Float32, no AMP ===")
try:
    with torch.no_grad():
        logits, loss = model(img, label)
    print(f"Logits NaN: {torch.isnan(logits).any()}")
    print(f"Loss: {loss.item()}")
    print(f"Loss NaN: {torch.isnan(loss).any()}")
    print(f"Loss Inf: {torch.isinf(loss).any()}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: 有 AMP
print("\n=== Test 2: With AMP ===")
try:
    with torch.amp.autocast('cuda'):
        logits, loss = model(img, label)
    print(f"Logits NaN: {torch.isnan(logits).any()}")
    print(f"Loss: {loss.item()}")
    print(f"Loss NaN: {torch.isnan(loss).any()}")
    print(f"Loss Inf: {torch.isinf(loss).any()}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: 训练一步
print("\n=== Test 3: One training step (no AMP) ===")
optim = torch.optim.AdamW(model.parameters(), lr=0.001)
try:
    model.train()
    logits, loss = model(img, label)
    print(f"Forward OK. Loss: {loss.item()}")
    loss.backward()
    print(f"Backward OK.")
    
    # 检查梯度
    nan_grads = 0
    total_grads = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            total_grads += 1
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                nan_grads += 1
                print(f"  NaN/Inf grad: {name}")
    print(f"Gradients: {nan_grads}/{total_grads} have NaN/Inf")
    
    optim.step()
    optim.zero_grad()
    print("Step OK.")
except Exception as e:
    print(f"Error: {e}")

# Test 4: 训练一步 with AMP
print("\n=== Test 4: One training step (with AMP) ===")
model2 = HormonicFormer(config).to(device)
optim2 = torch.optim.AdamW(model2.parameters(), lr=0.001)
scaler = torch.amp.GradScaler('cuda')
try:
    model2.train()
    with torch.amp.autocast('cuda'):
        logits, loss = model2(img, label)
    print(f"Forward OK. Loss: {loss.item()}")
    
    scaler.scale(loss).backward()
    print(f"Backward OK.")
    
    scaler.step(optim2)
    scaler.update()
    optim2.zero_grad()
    print("Step OK.")
except Exception as e:
    print(f"Error: {e}")
