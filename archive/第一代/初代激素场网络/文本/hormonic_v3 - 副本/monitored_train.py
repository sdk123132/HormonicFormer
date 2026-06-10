#!/usr/bin/env python3
"""
HormonicFormer v3 - 带实时监控的训练脚本
支持: WebSocket实时推送、TensorBoard、CSV日志
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
from training_monitor import TrainingMonitor, GPUMonitor
from web_server import run_server, update_monitor_data


def setup_logger(log_dir):
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'monitored_train_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('hormonic')


def get_dataloader(config, train=True):
    """获取数据加载器"""
    data_root = config['train']['data_root']
    batch_size = config['train']['batch_size']
    num_workers = config['train']['num_workers']
    pin_memory = config['train']['pin_memory']

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

        with autocast(enabled=use_amp):
            logits, loss = model(images, targets)
            loss = loss / accumulation_steps

        if nan_guard and (torch.isnan(loss) or torch.isinf(loss)):
            logger.warning(f"  NaN/Inf at batch {batch_idx}, skipping")
            continue

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            ce_loss = F.cross_entropy(logits, targets)
            total_ce += ce_loss.item()
            total_loss += loss.item() * accumulation_steps
            pred = logits.argmax(dim=-1)
            correct += (pred == targets).sum().item()
            total_samples += targets.size(0)

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
    return avg_loss, avg_acc, optimizer.param_groups[0]['lr']


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

    torch.save(state, checkpoint_dir / 'latest.pt')

    if epoch % config['train'].get('save_interval', 5) == 0:
        torch.save(state, checkpoint_dir / f'epoch_{epoch}.pt')

    if is_best:
        torch.save(state, checkpoint_dir / 'best.pt')


def main():
    parser = argparse.ArgumentParser(description='Train HormonicFormer v3 with Monitor')
    parser.add_argument('--config', type=str, default='local_config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点')
    parser.add_argument('--device', type=str, default='cuda', help='设备: cuda / cpu')
    parser.add_argument('--monitor-port', type=int, default=5000, help='监控服务器端口')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"HormonicFormer v3 - Monitored Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    print(f"Config: {config['name']}")
    print(f"Monitor: http://localhost:{args.monitor_port}")
    print(f"{'='*60}")

    # 日志
    logger = setup_logger(config['train']['log_dir'])

    # 启动监控服务器（后台线程）
    server_thread = threading.Thread(
        target=run_server,
        kwargs={'host': '0.0.0.0', 'port': args.monitor_port, 'debug': False},
        daemon=True
    )
    server_thread.start()
    time.sleep(1)  # 等待服务器启动

    # 初始化监控器
    monitor = TrainingMonitor(config)

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

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )

    # 学习率调度
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
    print(f"\n{'='*80}")
    print(f"Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best  | Time | DA    | GPU MB")
    print(f"{'='*80}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # 更新当前epoch
        update_monitor_data({
            'current_epoch': epoch
        })

        # 训练
        train_loss, train_acc, current_lr = train_epoch(
            model, train_loader, optimizer, scaler, scheduler,
            epoch, config, logger
        )

        # 评估
        val_loss, val_acc = evaluate(model, val_loader, config)

        # Hebbian 统计
        hebb_stats = model.get_hebbian_stats()

        # 更新神经调质
        DA, CB = model.update_neuromod(val_loss)

        # 获取GPU显存
        gpu_mem = GPUMonitor.log_memory_usage(logger)

        elapsed = time.time() - epoch_start

        # 保存
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            stale_epochs = 0
        else:
            stale_epochs += 1

        save_checkpoint(model, optimizer, scaler, epoch, config, is_best)

        # BWO剪枝重生
        if config['bwo'].get('use_bwo', True) and epoch % config['bwo'].get('evolve_interval', 5) == 0 and epoch > 0:
            model.prune_and_regrow(epoch)
            logger.info(f"BWO executed at epoch {epoch}")

        # 记录监控数据
        monitor.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=current_lr,
            da=DA,
            cb=CB,
            gpu_memory=gpu_mem,
            epoch_time=elapsed
        )

        # 更新Hebbian统计
        if hebb_stats:
            monitor.log_hebbian_stats(epoch, hebb_stats)

        # 通过WebSocket推送更新
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
            'hebbian_stats': hebb_stats,
            'gpu_memory': gpu_mem
        })

        # 打印
        print(f"  {epoch:3d} | {train_loss:.4f}     | {train_acc:5.1f}%    | {val_loss:.4f}   | {val_acc:5.1f}% | {best_acc:5.1f}%| {elapsed:.0f}s | {DA:.3f} | {gpu_mem:5d}")
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.1f}% val_loss={val_loss:.4f} val_acc={val_acc:.1f}% DA={DA:.3f}")

        # 早停
        if stale_epochs >= config['train'].get('max_stale_epochs', 10):
            print(f"Early stopping at epoch {epoch} (stale={stale_epochs})")
            break

    # 通知训练结束
    update_monitor_data({
        'is_training': False
    })

    print(f"{'='*80}")
    print(f"Training complete! Best val acc: {best_acc:.2f}%")
    print(f"{'='*80}")

    # 保存最终模型
    torch.save(model.state_dict(), Path(config['train']['checkpoint_dir']) / 'final.pt')
    print(f"Model saved to {config['train']['checkpoint_dir']}/final.pt")

    # 关闭监控器
    monitor.close()


if __name__ == '__main__':
    main()