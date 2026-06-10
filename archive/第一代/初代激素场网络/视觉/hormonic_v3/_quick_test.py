import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import time
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from hormonicformer_v3 import HormonicFormer

with open('local_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 创建模型
model = HormonicFormer(config).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Params: {params/1e6:.2f}M")

optim = torch.optim.AdamW(model.parameters(), lr=0.001)

# 只用 256 张图片快速测试
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
small_ds = Subset(full_ds, range(256))
loader = DataLoader(small_ds, batch_size=64, shuffle=True, num_workers=0)

print(f"\nQuick test: {len(small_ds)} samples, batch_size=64, {len(loader)} batches")
print(f"{'='*60}")

model.train()
total_t = 0

for epoch in range(3):
    epoch_loss = 0
    epoch_acc = 0
    n = 0
    t0 = time.time()
    
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        
        logits, loss = model(images, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        
        with torch.no_grad():
            acc = (logits.argmax(1) == targets).float().mean()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        n += 1
    
    elapsed = time.time() - t0
    total_t += elapsed
    avg_loss = epoch_loss / n
    avg_acc = epoch_acc / n * 100
    
    mem = torch.cuda.memory_allocated() / 1024**2
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} Acc={avg_acc:.1f}% Time={elapsed:.1f}s GPU={mem:.0f}MB")

print(f"\n{'='*60}")
print(f"Total time: {total_t:.1f}s")
print(f"Per-epoch: {total_t/3:.1f}s")
print(f"Estimated full epoch (60000 samples): {total_t/3 * 60000/256:.0f}s = {total_t/3 * 60000/256/60:.0f}min")
print(f"\n>>> Quick test PASSED!")
