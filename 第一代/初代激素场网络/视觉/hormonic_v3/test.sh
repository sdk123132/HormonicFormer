#!/bin/bash
# 快速测试脚本 - 验证环境

set -e

echo "=========================================="
echo "  HormonicFormer v3 - Quick Test"
echo "=========================================="

# 测试1: 导入测试
echo -e "\n[1/4] Testing imports..."
python3 << 'EOF'
import torch
import torch.nn as nn
import yaml
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ Device: {torch.cuda.get_device_name(0)}")
EOF

# 测试2: DFT Laplacian
echo -e "\n[2/4] Testing DFT Laplacian..."
python3 << 'EOF'
import sys
import torch
sys.path.insert(0, 'field')
from laplacian_dft import DFTLaplacian

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

lap = DFTLaplacian(196, device).to(device)
x = torch.randn(2, 196).to(device)
y = lap(x)

print(f"Input: {x.shape}")
print(f"Output: {y.shape}")
print(f"✓ DFT Laplacian OK")
EOF

# 测试3: 模型前向
echo -e "\n[3/4] Testing model forward..."
python3 << 'EOF'
import sys
import torch
import yaml
sys.path.insert(0, 'models')
from hormonicformer_v3 import HormonicFormer

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 减小规模用于测试
config['model']['n_layers'] = 2
config['model']['d_model'] = 128
config['model']['n_heads'] = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = HormonicFormer(config).to(device)
print(f"Model created")

# 测试前向
dummy = torch.randn(2, 1, 28, 28).to(device)
dummy_target = torch.randint(0, 10, (2,)).to(device)

print(f"\nForward test:")
print(f"  Input: {dummy.shape}")

with torch.no_grad():
    logits = model(dummy)

print(f"  Logits: {logits.shape}")
print(f"  ✓ Model forward OK")

# 测试带损失的完整前向
logits, loss = model(dummy, dummy_target)
print(f"  Loss: {loss.item():.4f}")
print(f"  ✓ Model forward+loss OK")

# Hebbian 统计
stats = model.get_hebbian_stats()
print(f"\nHebbian stats:")
for s in stats:
    print(f"  Layer {s['layer']}: G_mean={s['G_mean']:.4f}, sparsity={s['G_sparsity']:.2%}")
EOF

# 测试4: 小批量训练
echo -e "\n[4/4] Testing mini training..."
python3 << 'EOF'
import sys
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, 'models')
from hormonicformer_v3 import HormonicFormer

# 配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['model']['n_layers'] = 2
config['model']['d_model'] = 128
config['model']['n_heads'] = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

# 模型
model = HormonicFormer(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=device == 'cuda')

# 训练2步
model.train()
for step, (images, targets) in enumerate(train_loader):
    if step >= 2:
        break
    images, targets = images.to(device), targets.to(device)

    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=device == 'cuda'):
        logits, loss = model(images, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    acc = (logits.argmax(dim=-1) == targets).float().mean().item()
    print(f"  Step {step+1}: Loss={loss.item():.4f}, Acc={acc:.2%}")

print(f"✓ Training step OK")
EOF

echo -e "\n=========================================="
echo "  All tests PASSED!"
echo "=========================================="
echo -e "\nReady to launch full training with:"
echo "  bash launch.sh"
