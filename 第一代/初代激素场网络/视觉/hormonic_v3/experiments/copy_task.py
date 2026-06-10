#!/usr/bin/env python3
"""
长程探针实验 (Long-Range Probe)
===============================
改善一：证明长程依赖能力

任务：序列复制 (Copy Task)
输入：随机token序列 [t1, t2, ..., tS]
目标：复制相同序列 [t1, t2, ..., tS]

变量：S = 16, 32, 64, 128, 256, 512

对比对象：
  - HormonicFormer v3 (你的)
  - LSTM (负对照)
  - Embedding Lookup (随机基线)

成功标准：
  - S=256 时准确率 > 90%
  - S=512 时显著优于 LSTM
  - 相位同步度随S增大保持稳定

用法：
  python experiments/copy_task.py --seq_lengths 16 64 256 512 --epochs 20
"""
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'models'))
sys.path.insert(0, str(PROJECT_ROOT / 'field'))

from hormonicformer_v3 import HormonicFormer


# =============================================================================
# Copy Task 数据
# =============================================================================
class CopyTaskDataset(Dataset):
    """序列复制任务数据集"""
    def __init__(self, vocab_size, seq_len, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # 预生成随机序列
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 输入和目标完全相同（复制任务）
        seq = self.data[idx]
        return seq, seq.clone()


# =============================================================================
# 对比模型
# =============================================================================
class LSTMCopy(nn.Module):
    """LSTM 基准模型（负对照：已知长程会崩）"""
    def __init__(self, vocab_size, d_model=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(
            d_model, d_model, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [B, S]
        emb = self.embedding(x)  # [B, S, D]
        out, _ = self.lstm(emb)  # [B, S, D]
        logits = self.fc(out)  # [B, S, V]
        return logits


class EmbeddingOnly(nn.Module):
    """纯嵌入查找（随机基线）"""
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        logits = self.fc(emb)
        return logits


# =============================================================================
# 将序列包装为类图像输入供 HormonicFormer 使用
# =============================================================================
def seq_to_patches(seq, patch_size=2, vocab_size=100):
    """
    将token序列转换为类似图像的patch格式
    seq: [B, S] -> [B, 1, H, W]
    将每个token ID 映射到一个标量值，然后reshape为方形
    """
    B, S = seq.shape
    # token ID 归一化到 [-1, 1]
    vals = (seq.float() / (vocab_size - 1)) * 2 - 1  # [B, S]
    # 将序列reshape为方形（假设S是完美平方数）
    H = W = int(S ** 0.5)
    if H * W != S:
        # 填充到最近的完美平方数
        pad_len = (H + 1) ** 2 - S if H * W < S else H * W - S
        if pad_len > 0:
            vals = F.pad(vals, (0, pad_len), value=0)
        H = W = int(vals.shape[1] ** 0.5)
    img = vals.view(B, 1, H, W)
    return img


def patches_to_seq(logits, seq_len, vocab_size=100):
    """
    将HormonicFormer的分类输出映射回序列预测
    logits: [B, num_classes] -> [B, S] 的预测token
    这里简化处理：假设num_classes >= vocab_size
    """
    # 取前vocab_size个logits
    logits = logits[:, :vocab_size]
    preds = logits.argmax(dim=-1)  # [B]
    return preds


class HormonicWrapper(nn.Module):
    """
    包装 HormonicFormer 用于序列任务
    将序列 -> patches -> HormonicFormer -> 逐token分类
    """
    def __init__(self, config, vocab_size, seq_len, patch_size=2):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.patch_size = patch_size

        # 创建 HormonicFormer 配置
        self.config = config
        self.model = HormonicFormer(config)

        # 替换分类头以支持 vocab_size 输出
        d_model = config['model']['d_model']
        self.model.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, vocab_size)
        )

    def forward(self, seq):
        """seq: [B, S] -> logits: [B, S, V]"""
        B, S = seq.shape

        # 序列 -> 伪图像patches
        img = seq_to_patches(seq, self.patch_size, self.vocab_size)  # [B, 1, H, W]

        # 确保尺寸匹配 (pad if needed)
        _, _, H, W = img.shape
        expected_size = self.config['model']['seq_len']
        actual_patches = (H // self.patch_size) * (W // self.patch_size)

        # 通过 HormonicFormer
        # 返回每个patch的预测
        x = self.model.patch_embed(img)  # [B, d_model, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, seq_len, d_model]

        # 通过每个 block
        x_embed = x.detach().clone()
        for block in self.model.blocks:
            x = block(x, x_embed=x_embed if block.use_feedback else None)

        # 对每个位置预测token
        logits = self.model.classifier(x)  # [B, seq_len, vocab_size]

        # 取前S个位置 (去掉padding)
        logits = logits[:, :S, :]

        return logits


# =============================================================================
# 训练与评估
# =============================================================================
def train_epoch(model, loader, optimizer, device, vocab_size):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)  # [B, S, V]

        # 计算交叉熵损失
        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * S, V),
            y.reshape(B * S)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)  # [B, S]
        correct += (pred == y).sum().item()
        total += B * S

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device, vocab_size):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_accuracies = []  # 每个样本的准确率

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        B, S, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * S, V),
            y.reshape(B * S)
        )

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)  # [B, S]

        # 每个位置的准确率
        token_correct = (pred == y).float()  # [B, S]
        all_accuracies.extend(token_correct.mean(dim=1).cpu().tolist())

        correct += (pred == y).sum().item()
        total += B * S

    avg_acc = correct / total
    seq_acc = sum(1 for a in all_accuracies if a > 0.99) / len(all_accuracies)  # 完全正确的序列比例

    return total_loss / len(loader), avg_acc, seq_acc


# =============================================================================
# 相位同步度分析
# =============================================================================
def analyze_phase_sync(model, loader, device):
    """分析 HormonicFormer 各层的相位同步度"""
    if not hasattr(model, 'model'):
        return None

    model.eval()
    sync_by_layer = {i: [] for i in range(len(model.model.blocks))}

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            # 前向传播并收集 Hebbian 统计
            logits = model(x)
            for stat in model.model.get_hebbian_stats():
                sync_by_layer[stat['layer']].append(stat['G_mean'])

    result = {}
    for layer, values in sync_by_layer.items():
        if values:
            result[f'layer_{layer}'] = {
                'mean_sync': sum(values) / len(values),
                'std_sync': (sum((v - sum(values)/len(values))**2 for v in values) / len(values)) ** 0.5
            }
    return result


# =============================================================================
# 主实验
# =============================================================================
def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    for seq_len in args.seq_lengths:
        print(f"\n{'='*60}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'='*60}")

        # 数据
        train_ds = CopyTaskDataset(args.vocab_size, seq_len, num_samples=args.train_samples)
        val_ds = CopyTaskDataset(args.vocab_size, seq_len, num_samples=args.val_samples)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        results[seq_len] = {}

        # 对每个模型
        for model_name in args.models:
            print(f"\n  Training: {model_name}...")

            if model_name == 'hormonic':
                # 计算需要的配置
                H = W = int((seq_len ** 0.5))
                if H * H < seq_len:
                    H = W = H + 1
                num_patches = (H // 2) * (W // 2)  # patch_size=2

                config = {
                    'model': {
                        'd_model': args.d_model,
                        'n_heads': args.n_heads,
                        'n_layers': args.n_layers,
                        'seq_len': num_patches,
                        'n_steps': args.n_steps,
                        'patch_size': 2,
                        'n_classes': args.vocab_size,
                        'D0_amp': 0.1, 'D0_phase': 0.1, 'dt': 0.05,
                        'ei_balance': {
                            'enabled': True, 'tau_e': 2.0, 'tau_i': 1.0,
                            'gamma_e': 1.0, 'gamma_i': 0.8, 'w_inh': 0.3, 'inh_radius': 3
                        },
                        'sensory_feedback': {
                            'enabled': True, 'feedback_strength': 0.3, 'feedback_freq': 1
                        },
                        'hebbian': {
                            'enabled': True, 'eta_hebb': 0.001, 'eta_anti': 0.0005,
                            'sync_threshold': 0.3, 'tau_hebb': 10, 'decay': 0.999
                        },
                    },
                    'pc': {'use_pc': False, 'pred_hidden_mult': 4, 'aux_weight': 0.01}
                }
                model = HormonicWrapper(config, args.vocab_size, seq_len)

            elif model_name == 'lstm':
                model = LSTMCopy(args.vocab_size, d_model=args.d_model,
                                n_layers=args.n_layers, dropout=0.1)

            elif model_name == 'embedding':
                model = EmbeddingOnly(args.vocab_size, d_model=args.d_model)

            else:
                raise ValueError(f"Unknown model: {model_name}")

            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6
            )

            # 训练
            best_val_acc = 0
            best_epoch = 0
            history = []

            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, args.vocab_size)
                val_loss, val_acc, val_seq_acc = evaluate(model, val_loader, device, args.vocab_size)
                scheduler.step()

                history.append({
                    'epoch': epoch,
                    'train_loss': train_loss, 'train_acc': train_acc,
                    'val_loss': val_loss, 'val_acc': val_acc, 'val_seq_acc': val_seq_acc
                })

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch

                if epoch % args.log_interval == 0 or epoch == 1:
                    print(f"    Epoch {epoch:3d}: Train Acc={train_acc:.4f}, "
                          f"Val Acc={val_acc:.4f}, Val SeqAcc={val_seq_acc:.4f}")

            # 最终评估
            final_loss, final_acc, final_seq_acc = evaluate(model, val_loader, device, args.vocab_size)

            results[seq_len][model_name] = {
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'final_acc': final_acc,
                'final_seq_acc': final_seq_acc,
                'history': history
            }

            print(f"    Final: Token Acc={final_acc:.4f}, Seq Acc={final_seq_acc:.4f} "
                  f"(best={best_val_acc:.4f} @ epoch {best_epoch})")

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = output_dir / f'copy_task_results_{timestamp}.json'

    # 序列化时去掉 history 以减小文件大小
    summary = {}
    for seq_len, models in results.items():
        summary[str(seq_len)] = {
            name: {k: v for k, v in data.items() if k != 'history'}
            for name, data in models.items()
        }

    with open(result_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {result_file}")
    print(f"{'='*60}")

    # 打印总结表格
    print(f"\n{'='*60}")
    print(f"SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'SeqLen':>8} | {'Model':>12} | {'Token Acc':>10} | {'Seq Acc':>10}")
    print(f"{'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")
    for seq_len in args.seq_lengths:
        for model_name in args.models:
            data = results[seq_len][model_name]
            print(f"{seq_len:>8} | {model_name:>12} | {data['final_acc']:>10.4f} | {data['final_seq_acc']:>10.4f}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Long-Range Probe: Copy Task')
    parser.add_argument('--seq_lengths', nargs='+', type=int,
                       default=[16, 64, 256],
                       help='序列长度列表')
    parser.add_argument('--models', nargs='+', default=['hormonic', 'lstm', 'embedding'],
                       choices=['hormonic', 'lstm', 'embedding'],
                       help='要测试的模型')
    parser.add_argument('--vocab_size', type=int, default=50,
                       help='词表大小')
    parser.add_argument('--d_model', type=int, default=128,
                       help='模型维度')
    parser.add_argument('--n_heads', type=int, default=4,
                       help='注意力头数 (仅Hormonic)')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='层数')
    parser.add_argument('--n_steps', type=int, default=3,
                       help='CGL演化步数')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train_samples', type=int, default=10000)
    parser.add_argument('--val_samples', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    run_experiment(args)


if __name__ == '__main__':
    main()
