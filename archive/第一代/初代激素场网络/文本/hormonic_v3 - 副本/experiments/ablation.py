#!/usr/bin/env python3
"""
消融实验 (Ablation Study)
=========================
改善六：证明每个组件的因果贡献

配置：
  1. 完整模型 (Full)                    - 基线
  2. 关闭扩散项 (No Diffusion)          - 证明扩散是信息混合的核心
  3. 关闭反应项 (No Reaction)           - 证明非线性反应不是可有可无
  4. 关闭 E/I 平衡 (No EI)              - 证明侧抑制的作用
  5. 关闭感觉反馈 (No Feedback)         - 证明外部驱动的锚定作用
  6. 关闭 Hebbian 可塑性 (No Hebbian)   - 证明实时突触学习的作用
  7. 纯嵌入基线 (Embedding Only)        - 证明场演化在真正起作用

数据集：Fashion-MNIST (默认) 或 CIFAR-10

用法：
  python experiments/ablation.py --configs all --dataset fashion_mnist --epochs 10
  python experiments/ablation.py --configs no_diffusion no_reaction --epochs 5
"""
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'models'))
sys.path.insert(0, str(PROJECT_ROOT / 'field'))

from hormonicformer_v3 import HormonicFormer


# =============================================================================
# 消融配置生成器
# =============================================================================
def make_ablation_configs(base_config):
    """为每个消融条件生成配置"""
    configs = {}

    # 1. 完整模型
    configs['full'] = deepcopy(base_config)

    # 2. 关闭扩散项 (No Diffusion)
    cfg = deepcopy(base_config)
    cfg['model']['D0_amp'] = 0.0
    cfg['model']['D0_phase'] = 0.0
    configs['no_diffusion'] = cfg

    # 3. 关闭反应项 (No Reaction)
    cfg = deepcopy(base_config)
    cfg['model']['dt'] = 0.0  # 冻结反应
    configs['no_reaction'] = cfg

    # 4. 关闭 E/I 平衡
    cfg = deepcopy(base_config)
    if 'ei_balance' in cfg['model']:
        cfg['model']['ei_balance']['enabled'] = False
    configs['no_ei'] = cfg

    # 5. 关闭感觉反馈
    cfg = deepcopy(base_config)
    if 'sensory_feedback' in cfg['model']:
        cfg['model']['sensory_feedback']['enabled'] = False
    configs['no_feedback'] = cfg

    # 6. 关闭 Hebbian 可塑性
    cfg = deepcopy(base_config)
    if 'hebbian' in cfg['model']:
        cfg['model']['hebbian']['enabled'] = False
    configs['no_hebbian'] = cfg

    # 7. 纯嵌入基线 (Embedding Only)
    cfg = deepcopy(base_config)
    cfg['model']['n_steps'] = 0  # 不演化
    cfg['model']['n_layers'] = 1  # 单层
    if 'ei_balance' in cfg['model']:
        cfg['model']['ei_balance']['enabled'] = False
    if 'sensory_feedback' in cfg['model']:
        cfg['model']['sensory_feedback']['enabled'] = False
    if 'hebbian' in cfg['model']:
        cfg['model']['hebbian']['enabled'] = False
    configs['embedding_only'] = cfg

    return configs


# =============================================================================
# 数据加载
# =============================================================================
def get_dataloader(dataset_name, batch_size, num_workers, data_root='./data', train=True):
    """获取数据集"""
    if train:
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5) if dataset_name == 'cifar10' else (0.5,),
                                 (0.5, 0.5, 0.5) if dataset_name == 'cifar10' else (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5) if dataset_name == 'cifar10' else (0.5,),
                                 (0.5, 0.5, 0.5) if dataset_name == 'cifar10' else (0.5,))
        ])

    if dataset_name == 'fashion_mnist':
        dataset = datasets.FashionMNIST(data_root, train=train, download=True, transform=transform)
        in_channels = 1
        img_size = 28
    elif dataset_name == 'mnist':
        dataset = datasets.MNIST(data_root, train=train, download=True, transform=transform)
        in_channels = 1
        img_size = 28
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(data_root, train=train, download=True, transform=transform)
        in_channels = 3
        img_size = 32
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
                       num_workers=num_workers, pin_memory=True)

    return loader, in_channels, img_size


# =============================================================================
# 训练与评估
# =============================================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, loss = model(images, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits, loss = model(images, targets)
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


# =============================================================================
# 主实验
# =============================================================================
def run_ablation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 基础配置
    if args.dataset == 'cifar10':
        seq_len = (32 // args.patch_size) ** 2
        n_classes = 10
        in_channels = 3
    else:
        seq_len = (28 // args.patch_size) ** 2
        n_classes = 10
        in_channels = 1

    base_config = {
        'model': {
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'seq_len': seq_len,
            'n_steps': args.n_steps,
            'patch_size': args.patch_size,
            'n_classes': n_classes,
            'D0_amp': 0.1,
            'D0_phase': 0.1,
            'dt': 0.05,
            'ei_balance': {
                'enabled': True,
                'tau_e': 2.0,
                'tau_i': 1.0,
                'gamma_e': 1.0,
                'gamma_i': 0.8,
                'w_inh': 0.3,
                'inh_radius': 3,
            },
            'sensory_feedback': {
                'enabled': True,
                'feedback_strength': 0.3,
                'feedback_freq': 1,
            },
            'hebbian': {
                'enabled': True,
                'eta_hebb': 0.001,
                'eta_anti': 0.0005,
                'sync_threshold': 0.3,
                'tau_hebb': 10,
                'decay': 0.999,
            },
        },
        'pc': {
            'use_pc': True,
            'pred_hidden_mult': 4,
            'aux_weight': 0.01,
        }
    }

    # 生成消融配置
    all_configs = make_ablation_configs(base_config)

    # 选择要运行的配置
    if args.configs == ['all']:
        selected_configs = list(all_configs.keys())
    else:
        selected_configs = [c for c in args.configs if c in all_configs]

    print(f"\nRunning ablation configs: {selected_configs}")
    print(f"Dataset: {args.dataset}, Epochs: {args.epochs}")

    # 数据
    train_loader, _, _ = get_dataloader(args.dataset, args.batch_size, args.num_workers, args.data_root, train=True)
    val_loader, _, _ = get_dataloader(args.dataset, args.batch_size, args.num_workers, args.data_root, train=False)

    results = {}

    for config_name in selected_configs:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")

        config = all_configs[config_name]

        # 创建模型
        model = HormonicFormer(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Parameters: {total_params:.2f}M")

        # 训练
        best_val_acc = 0
        best_epoch = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, device)
            scheduler.step()

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            if epoch % args.log_interval == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                      f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        results[config_name] = {
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'final_train_acc': history[-1]['train_acc'],
            'final_val_acc': history[-1]['val_acc'],
            'params_M': total_params,
            'history': history
        }

        print(f"  Best: Val Acc={best_val_acc:.4f} @ epoch {best_epoch}")

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = output_dir / f'ablation_{args.dataset}_{timestamp}.json'

    summary = {
        name: {k: v for k, v in data.items() if k != 'history'}
        for name, data in results.items()
    }

    with open(result_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # 打印总结
    print(f"\n{'='*70}")
    print(f"ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':>20} | {'Best Val Acc':>12} | {'Params(M)':>10} | {'Delta':>8}")
    print(f"{'-'*20}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")

    full_acc = results.get('full', {}).get('best_val_acc', 0)
    for name in selected_configs:
        data = results[name]
        delta = data['best_val_acc'] - full_acc
        marker = ""
        if name != 'full':
            if delta < -0.05:
                marker = "<<< SIGNIFICANT DROP"
            elif delta < -0.02:
                marker = "< MODERATE DROP"
            elif delta > 0.02:
                marker = "> IMPROVEMENT"
        print(f"{name:>20} | {data['best_val_acc']:>12.4f} | {data['params_M']:>10.2f} | {delta:>+8.4f} {marker}")

    print(f"{'='*70}")
    print(f"Results saved to: {result_file}")
    print(f"\nInterpretation:")
    print(f"  - 如果 'no_diffusion' 显著下降 (< -5%): 扩散是核心信息混合机制")
    print(f"  - 如果 'no_reaction' 显著下降: 非线性反应不可忽略")
    print(f"  - 如果 'embedding_only' 接近随机: 场演化真正在起作用")

    return results


def main():
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--configs', nargs='+', default=['all'],
                       choices=['all', 'full', 'no_diffusion', 'no_reaction',
                               'no_ei', 'no_feedback', 'no_hebbian', 'embedding_only'],
                       help='要测试的消融配置')
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                       choices=['fashion_mnist', 'mnist', 'cifar10'])
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=3)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--log_interval', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    run_ablation(args)


if __name__ == '__main__':
    main()
