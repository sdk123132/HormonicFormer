"""
阶段 1：视觉任务回归（CIFAR-10）
监控所有调质回路健康度
运行: python run_phase1_cifar10.py
"""
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

# 添加 hormonic_v7r3_validated.py 路径
sys.path.insert(0, r'C:\Users\MR\Desktop')

# 尝试导入，如果失败则使用内嵌简化版
try:
    from hormonic_v7r3_validated import HormonicFormerV7r3
    print("Loaded HormonicFormerV7r3 from hormonic_v7r3_validated.py")
except ImportError as e:
    print(f"Import error: {e}")
    print("Using embedded version...")
    # 这里会包含简化版模块定义
    raise

# 配置
CONFIG = {
    'model': {
        'd_model': 64,
        'seq_len': 196,  # 14x14 patches
        'n_layers': 4,
        'n_heads': 4,
        'vocab_size': 10,  # CIFAR-10 类别
        'dropout': 0.1,
        'n_cgl_steps': 10,
        'D0_amp': 0.002,
        'D0_phase': 0.002,
        'cgl_dt': 0.02,
        'noise_scale': 0.001,
    },
    'use_neuromod': True,
    'use_pac': True,
    'use_pc': True,
    'pc_weight': 0.01,
    'g_coupling_strength': 0.1,
    'neuromod': {
        'da_init': 2.5,
        'da_ema_alpha': 0.9,
        'da_var_alpha': 0.9,
        'da_min': 0.1,
        'da_max': 0.9,
        'use_cb': True,
        'cb_gain': 2.0,
        'cb_threshold': 0.25,
        'tau_cb': 10.0,
        'cb_dt': 0.05,
    },
    'stp': {
        'U': 0.2,
        'tau_f': 1.0,
        'tau_d': 3.0,
        'dt': 0.05,
    },
    'hebbian': {
        'eta_potentiate': 0.001,
        'eta_depress': 0.0005,
        'sync_threshold': 0.3,
        'decay': 0.999,
    },
}

# 训练配置
BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
WARMUP_EPOCHS = 2


def get_cifar10_loaders():
    """加载 CIFAR-10 数据集，转为 patch 序列"""
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=2, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=True)
    
    return trainloader, testloader


def image_to_patches(images, patch_size=2):
    """
    将图像转换为 patch 序列
    images: [B, 3, 32, 32]
    return: [B, 196, 12]  (14x14=196 patches, 每个patch 2x2x3=12)
    """
    B, C, H, W = images.shape
    # unfold: [B, C, H//p, W//p, p, p]
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # [B, C, H/p, W/p, p, p]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # [B, H/p, W/p, C, p, p]
    patches = patches.view(B, -1, C * patch_size * patch_size)
    # [B, num_patches, patch_dim]
    return patches


class PatchEmbedding(nn.Module):
    """Patch embedding for CIFAR-10"""
    def __init__(self, patch_dim=12, d_model=64):
        super().__init__()
        self.proj = nn.Linear(patch_dim, d_model * 2)
        
    def forward(self, patches):
        # patches: [B, 196, 12]
        return self.proj(patches)  # [B, 196, 128]


class HormonicCIFAR10(nn.Module):
    """HormonicFormer for CIFAR-10"""
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_dim=12, d_model=config['model']['d_model'])
        self.backbone = HormonicFormerV7r3(config)
        # 替换输出头为分类头
        self.backbone.lm_head = nn.Linear(config['model']['d_model'] * 2, 10)
        
    def forward(self, images, targets=None):
        # images: [B, 3, 32, 32]
        patches = image_to_patches(images)  # [B, 196, 12]
        x = self.patch_embed(patches)  # [B, 196, 128]
        
        # 转复数场 [B, S, D, 2]
        B, S, D2 = x.shape
        D = D2 // 2
        psi = x.view(B, S, D, 2)
        
        # 通过 backbone
        logits = self.backbone.forward_cifar(psi)  # 需要修改 backbone
        
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        return logits


def train_epoch(model, loader, optimizer, scheduler, epoch, writer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(images, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 每 50 batch 记录一次
        if batch_idx % 50 == 0:
            global_step = epoch * len(loader) + batch_idx
            
            # 基础指标
            writer.add_scalar('Train/loss', loss.item(), global_step)
            writer.add_scalar('Train/acc', 100. * correct / total, global_step)
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_step)
            
            # 调质回路指标
            try:
                diag = model.backbone.get_diagnostics()
                writer.add_scalar('Neuromod/DA', diag['DA'], global_step)
                writer.add_scalar('Neuromod/CB', diag['CB'], global_step)
                writer.add_scalar('Neuromod/STP_eff', diag['STP_eff'], global_step)
                writer.add_scalar('CGL/limit_cycle_r', diag['limit_cycle_r'], global_step)
                writer.add_scalar('Hebbian/G_sparsity', diag['G_sparsity'], global_step)
            except:
                pass
            
            print(f'  Epoch {epoch} [{batch_idx}/{len(loader)}] '
                  f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits, loss = model(images, targets)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def main():
    """主训练循环"""
    print("=" * 70)
    print("阶段 1：CIFAR-10 视觉任务回归")
    print("=" * 70)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 数据
    print("\nLoading CIFAR-10...")
    trainloader, testloader = get_cifar10_loaders()
    
    # 模型
    print("\nBuilding model...")
    # 需要修改 HormonicFormerV7r3 添加 forward_cifar 方法
    # 这里简化处理，直接使用分类模式
    CONFIG['model']['n_classes'] = 10
    model = HormonicFormerV7r3(CONFIG)
    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    
    # 学习率调度
    total_steps = EPOCHS * len(trainloader)
    warmup_steps = WARMUP_EPOCHS * len(trainloader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # TensorBoard
    log_dir = 'runs/hormonic_v7r3_cifar10'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard: tensorboard --logdir={log_dir}")
    
    # 训练
    print(f"\nTraining for {EPOCHS} epochs...")
    best_acc = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, scheduler, 
                                           epoch, writer, device)
        val_loss, val_acc = evaluate(model, testloader, device)
        
        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        writer.add_scalar('Epoch/train_acc', train_acc, epoch)
        writer.add_scalar('Epoch/val_loss', val_loss, epoch)
        writer.add_scalar('Epoch/val_acc', val_acc, epoch)
        
        print(f'  Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%')
        print(f'  Val   Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{log_dir}/best_model.pt')
            print(f'  *** Best model saved: {best_acc:.2f}%')
    
    writer.close()
    print(f"\n{'='*70}")
    print(f"Training complete! Best accuracy: {best_acc:.2f}%")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
