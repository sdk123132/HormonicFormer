"""
阶段 1：CIFAR-10 视觉任务回归训练
监控所有调质回路健康度

运行: 
  $env:KMP_DUPLICATE_LIB_OK="TRUE"
  python run_phase1_train.py

查看 TensorBoard:
  tensorboard --logdir=runs/phase1_cifar10
"""
import os
import sys
import math
import time

# 必须在导入 torch 前设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

# 导入模型
sys.path.insert(0, r'C:\Users\MR\Desktop')
sys.path.insert(0, r'C:\Users\MR\Desktop\论文\关于场物理的神经框架\第二代')
from hormonic_cifar10 import HormonicCIFAR10

# ==================== 配置 ====================
CONFIG = {
    'model': {
        'd_model': 64,
        'n_layers': 4,
        'n_heads': 4,
        'dropout': 0.1,
        'n_cgl_steps': 10,
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

TRAIN_CONFIG = {
    'batch_size': 128,
    'epochs': 20,
    'lr': 3e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 2,
    'grad_clip': 1.0,
}

# ==================== 数据加载 ====================
def get_data_loaders():
    """加载 CIFAR-10 数据集"""
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    # 测试集标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(
        trainset, batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=True, num_workers=2, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)
    testloader = DataLoader(
        testset, batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=False, num_workers=2, pin_memory=True)
    
    return trainloader, testloader


# ==================== 训练函数 ====================
def train_epoch(model, loader, optimizer, scheduler, epoch, writer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 诊断指标累积
    da_sum = cb_sum = stp_sum = limit_r_sum = g_sparse_sum = 0
    num_batches = 0
    
    for batch_idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(images, targets)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['grad_clip'])
        
        optimizer.step()
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 获取诊断指标
        try:
            diag = model.get_diagnostics()
            da_sum += diag['DA']
            cb_sum += diag['CB']
            stp_sum += diag['STP_eff']
            limit_r_sum += diag['limit_cycle_r']
            g_sparse_sum += diag['G_sparsity']
            num_batches += 1
        except:
            pass
        
        # 记录到 TensorBoard
        global_step = epoch * len(loader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/loss_step', loss.item(), global_step)
            writer.add_scalar('Train/acc_step', 100. * correct / total, global_step)
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_step)
            
            if num_batches > 0:
                writer.add_scalar('Modulation/DA', da_sum / num_batches, global_step)
                writer.add_scalar('Modulation/CB', cb_sum / num_batches, global_step)
                writer.add_scalar('Modulation/STP_eff', stp_sum / num_batches, global_step)
                writer.add_scalar('CGL/limit_cycle_r', limit_r_sum / num_batches, global_step)
                writer.add_scalar('Hebbian/G_sparsity', g_sparse_sum / num_batches, global_step)
        
        # 打印进度
        if batch_idx % 50 == 0:
            print(f'  [{batch_idx:3d}/{len(loader)}] '
                  f'Loss: {loss.item():.4f} | '
                  f'Acc: {100.*correct/total:5.2f}% | '
                  f'DA: {da_sum/max(num_batches,1):.3f} | '
                  f'CB: {cb_sum/max(num_batches,1):.3f} | '
                  f'r: {limit_r_sum/max(num_batches,1):.3f}')
    
    avg_loss = total_loss / len(loader)
    avg_acc = 100. * correct / total
    
    return avg_loss, avg_acc


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
    
    avg_loss = total_loss / len(loader)
    avg_acc = 100. * correct / total
    
    return avg_loss, avg_acc


# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("阶段 1：CIFAR-10 视觉任务回归")
    print("=" * 70)
    print(f"配置: d_model={CONFIG['model']['d_model']}, "
          f"n_layers={CONFIG['model']['n_layers']}, "
          f"batch_size={TRAIN_CONFIG['batch_size']}")
    print("=" * 70)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 数据
    print("\n[1/4] Loading CIFAR-10...")
    trainloader, testloader = get_data_loaders()
    print(f"  Train: {len(trainloader.dataset)} samples")
    print(f"  Test:  {len(testloader.dataset)} samples")
    
    # 模型
    print("\n[2/4] Building model...")
    model = HormonicCIFAR10(CONFIG)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.2f}M")
    
    # 优化器
    print("\n[3/4] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=TRAIN_CONFIG['lr'], 
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # 学习率调度
    total_steps = TRAIN_CONFIG['epochs'] * len(trainloader)
    warmup_steps = TRAIN_CONFIG['warmup_epochs'] * len(trainloader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # TensorBoard
    print("\n[4/4] Setting up TensorBoard...")
    log_dir = 'runs/phase1_cifar10'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"  Log dir: {log_dir}")
    print(f"  View: tensorboard --logdir={log_dir}")
    
    # 训练
    print(f"\n{'='*70}")
    print(f"开始训练 {TRAIN_CONFIG['epochs']} epochs")
    print(f"{'='*70}\n")
    
    best_acc = 0
    start_epoch = 0
    
    # 尝试恢复 checkpoint
    checkpoint_path = f'{log_dir}/best_model.pt'
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['val_acc']
            print(f"*** Resumed from Epoch {start_epoch}, Best Acc: {best_acc:.2f}% ***\n")
        except:
            print("Could not resume, starting from scratch\n")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, TRAIN_CONFIG['epochs']):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        print("-" * 70)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, trainloader, optimizer, scheduler, epoch, writer, device)
        
        # 验证
        val_loss, val_acc = evaluate(model, testloader, device)
        
        # 记录 epoch 级别
        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        writer.add_scalar('Epoch/train_acc', train_acc, epoch)
        writer.add_scalar('Epoch/val_loss', val_loss, epoch)
        writer.add_scalar('Epoch/val_acc', val_acc, epoch)
        
        epoch_time = time.time() - epoch_start
        
        print("-" * 70)
        print(f'  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
        print(f'  Time: {epoch_time:.1f}s | Best: {best_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG,
            }, f'{log_dir}/best_model.pt')
            print(f'  *** New best model saved! ***')
    
    total_time = time.time() - start_time
    
    writer.close()
    
    print(f"\n{'='*70}")
    print(f"训练完成!")
    print(f"  总时间: {total_time/60:.1f} minutes")
    print(f"  最佳准确率: {best_acc:.2f}%")
    print(f"  模型保存: {log_dir}/best_model.pt")
    print(f"{'='*70}")
    
    print(f"\n查看训练过程:")
    print(f"  tensorboard --logdir={log_dir}")


if __name__ == '__main__':
    main()
