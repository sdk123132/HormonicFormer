"""
阶段 1 扩容版：CIFAR-10 视觉任务回归
配置: d_model=128, n_layers=6, epochs=50
修复: 添加 compute_da(loss) 调用
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, r'C:\Users\MR\Desktop')
sys.path.insert(0, r'C:\Users\MR\Desktop\论文\关于场物理的神经框架\第二代')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import math

from hormonic_cifar10 import HormonicCIFAR10

# ==================== 扩容配置 ====================
CONFIG = {
    'model': {
        'd_model': 128,  # 扩容: 64 -> 128
        'n_layers': 6,   # 扩容: 4 -> 6
        'n_heads': 4,
        'dropout': 0.1,
        'n_cgl_steps': 15,  # 扩容: 10 -> 15
        'D0_amp': 0.002,
        'D0_phase': 0.002,
        'cgl_dt': 0.02,
        'noise_scale': 0.001,
    },
    'use_neuromod': True,
    'use_pac': True,
    'use_pc': False,
    'g_coupling_strength': 0.1,
    'neuromod': {
        'da_init': 2.5,
        'da_ema_alpha': 0.9,
        'da_var_alpha': 0.9,
        'da_min': 0.1,
        'da_max': 0.9,
        'use_cb': True,
        'cb_gain': 2.0,
        'cb_threshold': 0.1,  # 降低: 0.25 -> 0.1
        'tau_cb': 10.0,
        'cb_dt': 0.05,
    },
    'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05},
    'hebbian': {'eta_potentiate': 0.001, 'eta_depress': 0.0005, 
                'sync_threshold': 0.3, 'decay': 0.999},
}

TRAIN_CONFIG = {
    'batch_size': 32,  # OOM修复: 64 -> 32
    'epochs': 50,      # 扩容: 20 -> 50
    'lr': 5e-4,        # 扩容: 3e-4 -> 5e-4
    'weight_decay': 0.05,
    'warmup_epochs': 3,
    'grad_clip': 1.0,
}

# ==================== 数据加载 ====================
def get_data_loaders():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
    testloader = DataLoader(testset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    
    return trainloader, testloader

# ==================== 训练函数 ====================
def train_epoch(model, loader, optimizer, scheduler, epoch, writer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(images, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['grad_clip'])
        optimizer.step()
        scheduler.step()
        
        # 关键修复：调用 compute_da 更新 da_ema
        if model.backbone.blocks[0].neuromod is not None:
            nm = model.backbone.blocks[0].neuromod
            if hasattr(nm, 'compute_da'):
                nm.compute_da(loss.item() if hasattr(loss, 'item') else loss)
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        global_step = epoch * len(loader) + batch_idx
        if batch_idx % 50 == 0:
            writer.add_scalar('Train/loss_step', loss.item(), global_step)
            writer.add_scalar('Train/acc_step', 100. * correct / total, global_step)
            
            # 记录诊断指标
            diag = {}
            try:
                diag = model.get_diagnostics()
                for k, v in diag.items():
                    writer.add_scalar(f'Modulation/{k}', v, global_step)
            except Exception as e:
                pass
            
            print(f'  [{batch_idx:3d}/{len(loader)}] Loss: {loss.item():.4f} | '
                  f'Acc: {100.*correct/total:5.2f}% | '
                  f'DA: {diag.get("DA", 0):.3f} | '
                  f'CB: {diag.get("CB", 0):.3f} | '
                  f'STP: {diag.get("STP_eff", 1):.3f}')
    
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, device):
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

# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("阶段 1 扩容版：CIFAR-10 (d_model=128, n_layers=6)")
    print("=" * 70)
    print(f"配置: d_model={CONFIG['model']['d_model']}, "
          f"n_layers={CONFIG['model']['n_layers']}, "
          f"epochs={TRAIN_CONFIG['epochs']}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    print("\n[1/4] Loading CIFAR-10...")
    trainloader, testloader = get_data_loaders()
    
    print("\n[2/4] Building model...")
    model = HormonicCIFAR10(CONFIG)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.2f}M")
    
    print("\n[3/4] Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['lr'], 
                                   weight_decay=TRAIN_CONFIG['weight_decay'])
    
    total_steps = TRAIN_CONFIG['epochs'] * len(trainloader)
    warmup_steps = TRAIN_CONFIG['warmup_epochs'] * len(trainloader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print("\n[4/4] Setting up TensorBoard...")
    log_dir = 'runs/phase1_cifar10_v2'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"  Log dir: {log_dir}")
    
    print(f"\n{'='*70}")
    print(f"开始训练 {TRAIN_CONFIG['epochs']} epochs")
    print(f"{'='*70}\n")
    
    best_acc = 0
    
    for epoch in range(TRAIN_CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, scheduler, epoch, writer, device)
        val_loss, val_acc = evaluate(model, testloader, device)
        
        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        writer.add_scalar('Epoch/train_acc', train_acc, epoch)
        writer.add_scalar('Epoch/val_loss', val_loss, epoch)
        writer.add_scalar('Epoch/val_acc', val_acc, epoch)
        
        print(f'  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG,
            }, f'{log_dir}/best_model.pt')
            print(f'  *** New best: {best_acc:.2f}% ***')
    
    writer.close()
    print(f"\n{'='*70}")
    print(f"训练完成! Best Acc: {best_acc:.2f}%")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
