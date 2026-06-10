#!/usr/bin/env python3
"""
HormonicFormer v3 - 本地单卡训练脚本
适配: RTX 5070 8GB / Windows / Linux单卡

用法:
    python local_train.py --config local_config.yaml
    
显存监控:
    训练过程中会自动打印GPU显存使用情况
实时监控:
    启动后可通过 http://localhost:5000 查看训练进度
"""
import os
import sys
import time
import argparse
import yaml
import logging
import threading
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'models'))
sys.path.insert(0, str(PROJECT_ROOT / 'field'))
sys.path.insert(0, str(PROJECT_ROOT / 'train_monitor'))

from hormonicformer_v3 import HormonicFormer
from web_server import run_server, update_monitor_data


def setup_logger(log_dir):
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'local_train_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('hormonic')


def print_gpu_memory(device_id=0):
    """打印GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device_id) / 1024**2
        reserved = torch.cuda.memory_reserved(device_id) / 1024**2
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**2
        print(f"  [GPU] 已用: {allocated:.0f}MB / 预留: {reserved:.0f}MB / 总共: {total:.0f}MB")
        return allocated
    return 0


def get_dataloader(config, train=True):
    """获取数据加载器 (Windows兼容版)"""
    data_root = config['train']['data_root']
    batch_size = config['train']['batch_size']
    num_workers = config['train']['num_workers']  # Windows设为0
    pin_memory = config['train']['pin_memory']

    # 数据预处理
    if train:
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(
                lambda x: x + config['train']['noise_sigma'] * torch.randn_like(x)
                if config['train']['noise_sigma'] > 0 else x
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    # 数据集
    dataset_name = config['train']['dataset']
    if dataset_name == 'fashion_mnist':
        dataset = datasets.FashionMNIST(data_root, train=train, download=True, transform=transform)
    elif dataset_name == 'mnist':
        dataset = datasets.MNIST(data_root, train=train, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(data_root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train,
    )

    return loader


def train_epoch(model, loader, optimizer, scaler, scheduler, epoch, config, logger):
    """训练一个epoch"""
    model.train()
    device = next(model.parameters()).device
    accumulation_steps = max(config['train'].get('accumulation_steps', 1), 1)
    log_interval = config['train']['log_interval']
    nan_guard = config['train']['nan_guard']
    use_amp = config['train']['use_amp']

    total_loss = 0.0
    total_ce = 0.0
    correct = 0
    total_samples = 0

    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=False)
        targets = targets.to(device, non_blocking=False)

        # 前向传播
        with autocast(enabled=use_amp):
            logits, loss = model(images, targets)
            loss = loss / accumulation_steps

        # NaN检测
        if nan_guard and (torch.isnan(loss) or torch.isinf(loss)):
            logger.warning(f"  NaN/Inf at batch {batch_idx}, skipping")
            continue

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积更新
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # 统计
        with torch.no_grad():
            ce_loss = F.cross_entropy(logits, targets)
            total_ce += ce_loss.item()
            total_loss += loss.item() * accumulation_steps
            pred = logits.argmax(dim=-1)
            correct += (pred == targets).sum().item()
            total_samples += targets.size(0)

        # BWO: 定期剪枝重生 (只在 epoch 结束时调用，且每5个epoch一次)
        # 注意：实际调用在 train() 函数中，epoch 循环结束后
        pass  # 已移至 epoch 循环外

        # 日志
        if batch_idx % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            acc = 100.0 * correct / total_samples if total_samples > 0 else 0
            logger.info(
                f"  Batch[{batch_idx:4d}/{len(loader)}] "
                f"Loss: {total_loss/(batch_idx+1):.4f} "
                f"CE: {total_ce/(batch_idx+1):.4f} "
                f"Acc: {acc:.1f}% "
                f"LR: {lr:.6f}"
            )

    avg_loss = total_loss / len(loader)
    avg_acc = 100.0 * correct / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, config):
    """评估"""
    model.eval()
    device = next(model.parameters()).device
    use_amp = config['train']['use_amp']

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=False)
        targets = targets.to(device, non_blocking=False)

        with autocast(enabled=use_amp):
            logits, loss = model(images, targets)

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(loader)
    avg_acc = 100.0 * correct / total

    return avg_loss, avg_acc


def save_checkpoint(model, optimizer, scaler, epoch, config, is_best=False):
    """保存检查点"""
    checkpoint_dir = Path(config['train']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'config': config,
    }

    # 最新检查点
    torch.save(state, checkpoint_dir / 'latest.pt')

    # 定期保存
    if epoch % config['train'].get('save_interval', 5) == 0:
        torch.save(state, checkpoint_dir / f'epoch_{epoch}.pt')

    # 最佳模型
    if is_best:
        torch.save(state, checkpoint_dir / 'best.pt')


def main():
    parser = argparse.ArgumentParser(description='Train HormonicFormer v3 (Local Single GPU)')
    parser.add_argument('--config', type=str, default='local_config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点')
    parser.add_argument('--device', type=str, default='cuda', help='设备: cuda / cpu')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"HormonicFormer v3 - Local Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    print(f"Config: {config['name']}")
    print(f"{'='*60}")

    # 日志
    logger = setup_logger(config['train']['log_dir'])

    # 设置随机种子
    seed = config['train']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 数据
    logger.info("Loading dataset...")
    train_loader = get_dataloader(config, train=True)
    val_loader = get_dataloader(config, train=False)
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # 模型
    logger.info("Creating model...")
    model = HormonicFormer(config).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Parameters: {total_params:.2f}M total, {trainable_params:.2f}M trainable")

    # 显存检查
    print_gpu_memory()

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )

    # 学习率调度 (带warmup)
    epochs = config['train']['epochs']
    warmup_epochs = config['train']['warmup_epochs']
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(warmup_steps, 1)
    ) if warmup_epochs > 0 else None

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6
    )

    if warmup_scheduler:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        scheduler = cosine_scheduler

    # 混合精度
    scaler = GradScaler(enabled=config['train']['use_amp'])

    start_epoch = 1
    best_acc = 0.0
    stale_epochs = 0

    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # 通知监控服务器训练开始
    update_monitor_data({
        'is_training': True,
        'total_epochs': epochs
    })

    # 训练循环
    logger.info(f"Starting training: {epochs} epochs")
    print(f"\n{'='*60}")
    print(f"Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best  | Time")
    print(f"{'='*60}")

    for epoch in range(start_epoch, epochs + 1):
        # 更新当前epoch
        update_monitor_data({
            'current_epoch': epoch
        })
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler, scheduler,
            epoch, config, logger
        )

        # 评估
        val_loss, val_acc = evaluate(model, val_loader, config)

        # Hebbian 统计
        hebb_stats = model.get_hebbian_stats()
        if hebb_stats:
            sparsity_info = ", ".join(
                f"L{s['layer']}:sparsity={s['G_sparsity']:.1%}" for s in hebb_stats[:2]
            )
        else:
            sparsity_info = "N/A"

        elapsed = time.time() - epoch_start

        # 保存
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            stale_epochs = 0
        else:
            stale_epochs += 1

        save_checkpoint(model, optimizer, scaler, epoch, config, is_best)

        # BWO: 定期剪枝重生 (每5个epoch，在epoch结束后调用)
        if config['bwo'].get('use_bwo', True) and epoch % config['bwo'].get('evolve_interval', 5) == 0 and epoch > 0:
            model.prune_and_regrow(epoch)
            logger.info(f"BWO executed at epoch {epoch}")

        # 更新神经调质 DA/CB
        DA, CB = model.update_neuromod(val_loss)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 通过WebSocket推送监控数据
        update_monitor_data({
            'metrics': {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': current_lr,
                'da': DA,
                'cb': CB
            },
            'hebbian_stats': hebb_stats
        })

        # 打印 (含 DA/CB)
        print(f"  {epoch:3d} | {train_loss:.4f}     | {train_acc:5.1f}%    | {val_loss:.4f}   | {val_acc:5.1f}% | {best_acc:5.1f}%| {elapsed:.0f}s | DA:{DA:.3f} CB:{CB:.3f} | {sparsity_info}")
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.1f}% val_loss={val_loss:.4f} val_acc={val_acc:.1f}% DA={DA:.3f} CB={CB:.3f}")

        # 显存
        print_gpu_memory()

        # 早停
        if stale_epochs >= config['train'].get('max_stale_epochs', 10):
            print(f"Early stopping at epoch {epoch} (stale={stale_epochs})")
            break

    # 通知训练结束
    update_monitor_data({
        'is_training': False
    })

    print(f"{'='*60}")
    print(f"Training complete! Best val acc: {best_acc:.2f}%")
    print(f"{'='*60}")

    # 保存最终模型
    torch.save(model.state_dict(), Path(config['train']['checkpoint_dir']) / 'final.pt')
    print(f"Model saved to {config['train']['checkpoint_dir']}/final.pt")


if __name__ == '__main__':
    main()
