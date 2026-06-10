"""
HormonicFormer v6.1 - DCU训练脚本
v6.1改进:
  - 每个epoch开始时重置STP/稳态/胶质状态 (防止资源耗尽)
  - 保持v4所有功能: checkpoint, 早停, warmup, CSV日志, DDP
"""
import os
import sys
import yaml
import time
import random
import csv
import math
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
from hormonicformer_v3 import HormonicFormer
from torchvision import datasets, transforms


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / (1024 ** 3)
        print(f"[Device] {torch.cuda.get_device_name(0)} | {total_mem:.1f} GB")
        return device
    print("[Device] CPU")
    return torch.device('cpu')


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr_scale = self._get_lr_scale()
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * lr_scale

    def _get_lr_scale(self):
        if self.current_step < self.warmup_steps:
            return max(self.current_step / max(self.warmup_steps, 1), 1e-6)
        else:
            progress = (self.current_step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def get_dataloaders(config, rank, world_size):
    tc = config['train']
    noise_sigma = tc.get('noise_sigma', 0.0)

    train_transforms = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    if noise_sigma > 0:
        train_transforms.append(AddGaussianNoise(sigma=noise_sigma))
    test_transforms = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]

    train_ds = datasets.FashionMNIST(
        tc.get('data_root', './data'), train=True, download=True,
        transform=transforms.Compose(train_transforms))
    test_ds = datasets.FashionMNIST(
        tc.get('data_root', './data'), train=False, download=True,
        transform=transforms.Compose(test_transforms))

    train_sampler = None
    shuffle = True
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=tc['batch_size'], shuffle=shuffle,
        sampler=train_sampler, num_workers=tc.get('num_workers', 4),
        pin_memory=tc.get('pin_memory', True), drop_last=True)
    test_loader = DataLoader(
        test_ds, batch_size=tc['batch_size'] * 2, shuffle=False,
        num_workers=tc.get('num_workers', 4), pin_memory=tc.get('pin_memory', True))

    return train_loader, test_loader, train_sampler


class AddGaussianNoise:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.sigma


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, config, path):
    state = {
        'epoch': epoch, 'best_acc': best_acc, 'config': config,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_step': scheduler.current_step if hasattr(scheduler, 'current_step') else 0,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cuda'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_step' in ckpt:
        scheduler.current_step = ckpt['scheduler_step']
    print(f"[Checkpoint] Loaded: {path} (epoch {ckpt['epoch']}, best_acc {ckpt['best_acc']:.4f})")
    return ckpt['epoch'], ckpt['best_acc']


def train_epoch(model, loader, optimizer, scaler, scheduler, device, epoch, rank, config):
    model.train()
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    tc = config['train']
    accumulation_steps = tc.get('accumulation_steps', 1)
    use_amp = tc.get('use_amp', True)
    grad_clip = tc.get('grad_clip', 0.0)
    nan_guard = tc.get('nan_guard', True)

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    nan_skips = 0

    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, loss = model(images, targets)
            loss_scaled = loss / accumulation_steps

        scaler.scale(loss_scaled).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            has_nan = any(
                torch.isnan(p.grad).any() for p in model.parameters()
                if p.grad is not None
            ) if nan_guard else False

            if has_nan:
                nan_skips += 1
                optimizer.zero_grad()
                scaler.update()
                if rank == 0 and nan_skips <= 5:
                    print(f"  [WARN] NaN grad at batch {batch_idx}, skipping")
                continue

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == targets).float().mean()

        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1

        if batch_idx % tc.get('log_interval', 50) == 0 and rank == 0:
            lr = scheduler.get_lr()
            print(f"  [{batch_idx}/{len(loader)}] loss={loss.item():.4f} acc={acc.item():.3f} lr={lr:.2e}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = total_acc / max(num_batches, 1)
    if nan_skips > 0 and rank == 0:
        print(f"  [WARN] Total NaN skips this epoch: {nan_skips}")
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    use_amp = config['train'].get('use_amp', True)
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, loss = model(images, targets)
        pred = logits.argmax(dim=1)
        acc = (pred == targets).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1

    return total_loss / max(num_batches, 1), total_acc / max(num_batches, 1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='HormonicFormer v6.1 Training')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    rank, world_size, local_rank = setup_distributed()
    device = get_device()

    if rank == 0:
        print("=" * 70)
        print("HormonicFormer v6.1 - DCU Training")
        print(f"  Fixes: Laplacian scale + CFL stability + DA init + STP reset + G cache")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  World Size: {world_size}")
        print("=" * 70)

    seed = config['train'].get('seed', 42)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    train_loader, test_loader, train_sampler = get_dataloaders(config, rank, world_size)
    model = HormonicFormer(config).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=config['dist'].get('find_unused_parameters', False))

    tc = config['train']
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc['lr'], weight_decay=tc['weight_decay'])
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=tc.get('warmup_epochs', 3),
        total_epochs=tc['epochs'],
        steps_per_epoch=len(train_loader) // tc.get('accumulation_steps', 1))
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch += 1

    ckpt_dir = tc.get('checkpoint_dir', './checkpoints')
    log_dir = tc.get('log_dir', './logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, 'train_log.csv')
    csv_exists = os.path.exists(csv_path)
    if rank == 0:
        csv_file = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow([
                'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                'lr', 'DA', 'CB', 'G_sparsity',
                'stp_efficacy', 'homeo_gain', 'homeo_activity', 'epoch_time'
            ])

    stale_epochs = 0
    max_stale = tc.get('max_stale_epochs', 10)

    for epoch in range(start_epoch, tc['epochs']):
        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{tc['epochs']}")
            print(f"{'=' * 70}")

        # ═══════════════════════════════════════════════════
        # v6.1 FIX: 每个epoch开始时重置可塑性状态
        # 防止STP资源耗尽、稳态增益漂移
        # ═══════════════════════════════════════════════════
        base_model = model.module if hasattr(model, 'module') else model
        base_model.reset_neuromod_for_epoch()

        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler, scheduler,
            device, epoch, rank, config)
        epoch_time = time.time() - t0

        if rank == 0:
            val_loss, val_acc = evaluate(model, test_loader, device, config)
            base_model = model.module if hasattr(model, 'module') else model
            DA, CB = base_model.update_neuromod(val_loss)
            diag = base_model.get_diagnostics()
            lr_now = scheduler.get_lr()

            print(f"\n[Epoch {epoch + 1}]")
            print(f"  Train: loss={train_loss:.4f} acc={train_acc:.3f}")
            print(f"  Val:   loss={val_loss:.4f} acc={val_acc:.3f}")
            print(f"  LR: {lr_now:.2e} | DA: {DA:.3f} | CB: {CB:.3f} | "
                  f"G_sparsity: {diag['G_sparsity']:.1%}")
            print(f"  STP: u={diag.get('stp_u_mean',0):.3f} r={diag.get('stp_r_mean',0):.3f} "
                  f"eff={diag.get('stp_efficacy_mean',0):.3f} | "
                  f"Homeo: gain={diag.get('homeo_gain_mean',1):.3f}"
                  f"±{diag.get('homeo_gain_std',0):.3f} "
                  f"activity={diag.get('homeo_activity_mean',0.5):.3f}")
            print(f"  Energy: {diag.get('energy_mean', 0):.3f} | "
                  f"Alive: {diag.get('alive_ratio', 1):.1%} | "
                  f"Time: {epoch_time:.1f}s")

            csv_writer.writerow([
                epoch + 1, f'{train_loss:.4f}', f'{train_acc:.4f}',
                f'{val_loss:.4f}', f'{val_acc:.4f}', f'{lr_now:.6f}',
                f'{DA:.4f}', f'{CB:.4f}', f'{diag["G_sparsity"]:.4f}',
                f'{diag.get("stp_efficacy_mean",0):.4f}',
                f'{diag.get("homeo_gain_mean",1):.4f}',
                f'{diag.get("homeo_activity_mean",0.5):.4f}',
                f'{epoch_time:.1f}',
            ])
            csv_file.flush()

            improved = val_acc > best_acc
            if improved:
                best_acc = val_acc
                stale_epochs = 0
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, config,
                                os.path.join(ckpt_dir, 'best.pt'))
            else:
                stale_epochs += 1

            if (epoch + 1) % tc.get('save_interval', 5) == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, config,
                                os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pt'))

            if stale_epochs >= max_stale:
                print(f"\n[Early Stop] No improvement for {max_stale} epochs. Stopping.")
                break

        base_model = model.module if hasattr(model, 'module') else model
        bwo_cfg = config.get('bwo', {})
        if bwo_cfg.get('use_bwo', False):
            base_model.prune_and_regrow(
                epoch, interval=bwo_cfg.get('evolve_interval', 5),
                regrow_ratio=bwo_cfg.get('flip_ratio', 0.02))

        if world_size > 1:
            dist.barrier()

    if rank == 0:
        csv_file.close()
        print(f"\n{'=' * 70}")
        print(f"Training Complete! Best Val Acc: {best_acc:.4f}")
        print(f"Log: {csv_path}")
        print(f"Best model: {os.path.join(ckpt_dir, 'best.pt')}")
        print(f"{'=' * 70}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
