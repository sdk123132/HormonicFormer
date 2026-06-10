"""
简化版阶段 1 训练 - 用于调试
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, r'C:\Users\MR\Desktop')
sys.path.insert(0, r'C:\Users\MR\Desktop\论文\关于场物理的神经框架\第二代')

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from hormonic_cifar10 import HormonicCIFAR10

print("=" * 60)
print("阶段 1 简化训练")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 数据
print("\nLoading CIFAR-10...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)  # num_workers=0 避免多进程问题
print(f"Train: {len(trainset)} samples, {len(trainloader)} batches")

# 模型
print("\nBuilding model...")
config = {
    'model': {'d_model': 64, 'n_layers': 4, 'n_heads': 4, 'dropout': 0.1, 
              'n_cgl_steps': 10, 'D0_amp': 0.002, 'D0_phase': 0.002, 
              'cgl_dt': 0.02, 'noise_scale': 0.001},
    'use_neuromod': True, 'use_pac': True, 'use_pc': False,
    'g_coupling_strength': 0.1,
    'neuromod': {'da_init': 2.5, 'da_ema_alpha': 0.9, 'da_var_alpha': 0.9,
                 'da_min': 0.1, 'da_max': 0.9, 'use_cb': True, 'cb_gain': 2.0,
                 'cb_threshold': 0.25, 'tau_cb': 10.0, 'cb_dt': 0.05},
    'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05},
    'hebbian': {'eta_potentiate': 0.001, 'eta_depress': 0.0005, 
                'sync_threshold': 0.3, 'decay': 0.999},
}
model = HormonicCIFAR10(config).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

# 训练
print("\nTraining...")
model.train()
for epoch in range(2):  # 只跑2个epoch测试
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(trainloader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(images, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'  [{batch_idx:3d}/{len(trainloader)}] '
                  f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
    
    print(f"\nEpoch {epoch+1} complete!")
    print(f"  Avg Loss: {total_loss/len(trainloader):.4f}")
    print(f"  Accuracy: {100.*correct/total:.2f}%")

print("\n" + "=" * 60)
print("训练完成!")
print("=" * 60)
