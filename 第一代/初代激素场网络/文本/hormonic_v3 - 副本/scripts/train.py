#!/usr/bin/env python3
"""
HormonicFormer v3 - 训练脚本
支持: 分布式训练(DDP), 混合精度(AMP), DCU兼容, BWO剪枝, 神经调质
"""
import os
import sys
import math
import time
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'models'))
sys.path.insert(0, str(PROJECT_ROOT / 'field'))

from hormonicformer_v3 import HormonicFormer


def setup_logger(log_dir, rank=0):
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'train_{timestamp}_rank{rank}.log'

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='[%(asctime)s] [Rank %(rank)d] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if rank == 0 else logging.NullHandler()
        ]
    )
    logger = logging.getLogger('hormonic')
    logger = logging.LoggerAdapter(logger, {'rank': rank})
    return logger


def setup_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    return rank, local_rank, world_size


def cleanup_distributed():
    """清理分布式"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_dataloader(config, rank, world_size, train=True):
    """获取数据加载器"""
    data_root = config['train']['data_root']
    batch_size = config['train']['batch_size']
    num_workers = config['train']['num_workers']
    pin_memory = config['train']['pin_memory']

    # 数据预处理
    if train:
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            # 添加高斯噪声
            transforms.Lambda(lambda x: x + config['train']['noise_sigma'] * torch.randn_like(x))
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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 分布式采样
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=train) if world_size > 1 else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train,
    )

    return loader, sampler


def get_scheduler(optimizer, config, steps_per_epoch):
    """获取学习率调度器"""
    epochs = config['train']['epochs']
    warmup_epochs = config['train']['warmup_epochs']
    scheduler_type = config['train']['scheduler']

    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if scheduler_type == 'cosine':
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        ) if warmup_steps > 0 else None

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
        )

        if warmup:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
            )
        return cosine

    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    else:
        return None


def train_epoch(model, loader, optimizer, scaler, scheduler, epoch, config, rank, logger):
    """训练一个epoch"""
    model.train()
    device = next(model.parameters()).device

    use_amp = config['train']['use_amp']
    accumulation_steps = config['train'].get('accumulation_steps', 1)
    log_interval = config['train']['log_interval']
    nan_guard = config['train']['nan_guard']

    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    correct = 0
    total = 0
    num_batches = 0

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 前向传播
        with autocast(enabled=use_amp):
            logits, loss = model(images, targets)
            loss = loss / accumulation_steps

        # NaN检测
        if nan_guard and (torch.isnan(loss) or torch.isinf(loss)):
            logger.warning(f"NaN/Inf detected at batch {batch_idx}, skipping!")
            continue

        # 反向传播
        scaler.scale(loss).backward()

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
            total += targets.size(0)

        num_batches += 1

        # BWO: 定期剪枝重生
        if hasattr(model, 'module'):
            model.module.prune_and_regrow(epoch)
        else:
            model.prune_and_regrow(epoch)

        # 日志
        if rank == 0 and batch_idx % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            acc = 100.0 * correct / total if total > 0 else 0
            logger.info(
                f"Epoch[{epoch}] Batch[{batch_idx}/{len(loader)}] "
                f"Loss: {total_loss/num_batches:.4f} "
                f"CE: {total_ce/num_batches:.4f} "
                f"Acc: {acc:.2f}% "
                f"LR: {lr:.6f}"
            )

    # Epoch 统计
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = 100.0 * correct / max(total, 1)

    return avg_loss, avg_acc, elapsed


@torch.no_grad()
def evaluate(model, loader, config, rank, logger):
    """评估"""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=config['train']['use_amp']):
            logits, loss = model(images, targets)

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    # 汇总分布式结果
    if dist.is_initialized():
        stats = torch.tensor([total_loss, correct, total], device=device)
        dist.all_reduce(stats)
        total_loss, correct, total = stats.tolist()

    avg_loss = total_loss / len(loader)
    avg_acc = 100.0 * correct / total

    return avg_loss, avg_acc


def save_checkpoint(model, optimizer, scaler, epoch, config, checkpoint_dir, is_best=False):
    """保存检查点"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        'epoch': epoch,
        'model': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'config': config,
    }

    # 最新检查点
    latest_path = checkpoint_dir / 'latest.pt'
    torch.save(state, latest_path)

    # 定期保存
    if epoch % config['train'].get('save_interval', 1) == 0:
        epoch_path = checkpoint_dir / f'epoch_{epoch}.pt'
        torch.save(state, epoch_path)

    # 最佳模型
    if is_best:
        best_path = checkpoint_dir / 'best.pt'
        torch.save(state, best_path)


def main():
    parser = argparse.ArgumentParser(description='Train HormonicFormer v3')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 分布式设置
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # 日志
    logger = setup_logger(config['train']['log_dir'], rank)
    if rank == 0:
        logger.info(f"HormonicFormer v3 Training Start")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info(f"Config: {config['name']}")

    # 设置随机种子
    seed = config['train']['seed'] + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 数据
    train_loader, train_sampler = get_dataloader(config, rank, world_size, train=True)
    val_loader, _ = get_dataloader(config, rank, world_size, train=False)

    if rank == 0:
        logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # 模型
    model = HormonicFormer(config).to(device)
    if rank == 0:
        total = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Model params: {total:.2f}M")

    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=config['dist'].get('find_unused_parameters', False))

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )

    # 学习率调度
    scheduler = get_scheduler(optimizer, config, len(train_loader))

    # 混合精度
    scaler = GradScaler(enabled=config['train']['use_amp'])

    start_epoch = 1
    best_acc = 0.0
    stale_epochs = 0

    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # 训练循环
    for epoch in range(start_epoch, config['train']['epochs'] + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 训练
        train_loss, train_acc, elapsed = train_epoch(
            model, train_loader, optimizer, scaler, scheduler,
            epoch, config, rank, logger
        )

        # 评估
        val_loss, val_acc = evaluate(model, val_loader, config, rank, logger)

        # Hebbian 统计
        hebb_stats = ""
        if rank == 0 and hasattr(model, 'module'):
            stats = model.module.get_hebbian_stats()
            if stats:
                hebb_stats = f" | Hebb: " + ", ".join(
                    f"L{s['layer']}:sparsity={s['G_sparsity']:.1%}" for s in stats[:2]
                )

        # 日志
        if rank == 0:
            logger.info(
                f"Epoch[{epoch}] Summary: "
                f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
                f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}% | "
                f"Time={elapsed:.1f}s"
                f"{hebb_stats}"
            )

        # 保存检查点
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            stale_epochs = 0
        else:
            stale_epochs += 1

        if rank == 0:
            save_checkpoint(model, optimizer, scaler, epoch, config,
                          config['train']['checkpoint_dir'], is_best)
            logger.info(f"Best val acc: {best_acc:.2f}%, Stale: {stale_epochs}")

        # 早停
        if stale_epochs >= config['train'].get('max_stale_epochs', 10):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    cleanup_distributed()
    if rank == 0:
        logger.info("Training complete!")


if __name__ == '__main__':
    main()
