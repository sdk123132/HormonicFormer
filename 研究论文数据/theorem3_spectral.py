"""
定理 3 实验验证：Hebbian G 矩阵谱分析
目的：分析 G 矩阵的特征值分布如何随数据量变化
纯 CPU，不影响 GPU 训练

实验设计：
- 固定 seq_len=64, d_model=64
- 数据量: 100, 500, 1000, 5000, 10000 样本
- 每个数据量训练固定 epoch，记录 G 矩阵
- 分析: 特征值分布、条件数、有效秩
"""
import os, sys, json, math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制 CPU

sys.path.insert(0, r'C:\Users\MR\Desktop')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hormonic_v7r3_validated import HormonicFormerV7r3

# ==================== 配置 ====================
SEQ_LEN = 64
D_MODEL = 64
VOCAB_SIZE = 50
N_LAYERS = 1
EPOCHS_PER_SCALE = 5
DATA_SCALES = [100, 500, 1000, 5000, 10000, 50000]

CONFIG = {
    'model': {
        'd_model': D_MODEL,
        'n_layers': N_LAYERS,
        'n_heads': 4,
        'seq_len': SEQ_LEN - 1,  # 因为 input = seq[:-1]
        'vocab_size': VOCAB_SIZE,
        'dropout': 0.0,
        'n_cgl_steps': 10,
        'D0_amp': 0.002,
        'D0_phase': 0.002,
        'cgl_dt': 0.02,
        'noise_scale': 0.001,
    },
    'use_neuromod': True,
    'use_pac': False,
    'use_pc': False,
    'g_coupling_strength': 0.1,
    'neuromod': {
        'da_init': 2.5, 'da_ema_alpha': 0.9, 'da_var_alpha': 0.9,
        'da_min': 0.1, 'da_max': 0.9,
        'use_cb': True, 'cb_gain': 2.0, 'cb_threshold': 0.1,
        'tau_cb': 10.0, 'cb_dt': 0.05,
    },
    'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05},
    'hebbian': {'eta_potentiate': 0.001, 'eta_depress': 0.0005,
                'sync_threshold': 0.3, 'decay': 0.999},
}


def generate_synthetic_data(n_samples, seq_len, vocab_size):
    """生成合成序列数据（有结构的，不是纯随机）"""
    data = []
    for _ in range(n_samples):
        # 模式1: 重复模式 (30%)
        if np.random.random() < 0.3:
            pattern_len = np.random.randint(2, min(8, seq_len//2))
            pattern = np.random.randint(0, vocab_size, pattern_len)
            seq = np.tile(pattern, seq_len // pattern_len + 1)[:seq_len]
        # 模式2: 递增/递减 (20%)
        elif np.random.random() < 0.3:
            start = np.random.randint(0, vocab_size)
            step = np.random.choice([-1, 1])
            seq = np.array([(start + i * step) % vocab_size for i in range(seq_len)])
        # 模式3: 随机 (50%)
        else:
            seq = np.random.randint(0, vocab_size, seq_len)
        data.append(seq)
    return torch.tensor(np.array(data), dtype=torch.long)


def analyze_G_matrix(G):
    """分析 G 矩阵的谱性质"""
    G_np = G.detach().cpu().numpy()
    
    # 特征值分解
    eigenvalues = np.linalg.eigvalsh(G_np)
    eigenvalues = np.sort(eigenvalues)[::-1]  # 降序
    
    # 条件数（避免零除）
    nonzero_eigs = eigenvalues[eigenvalues > 1e-10]
    if len(nonzero_eigs) > 1:
        condition_number = nonzero_eigs[0] / nonzero_eigs[-1]
    else:
        condition_number = float('inf')
    
    # 有效秩 (effective rank)
    # 定义: exp(entropy of normalized eigenvalues)
    abs_eigs = np.abs(eigenvalues)
    total = abs_eigs.sum()
    if total > 1e-10:
        p = abs_eigs / total
        p = p[p > 1e-15]
        entropy = -np.sum(p * np.log(p))
        effective_rank = np.exp(entropy)
    else:
        effective_rank = 0
    
    # 稀疏度
    sparsity = (np.abs(G_np) < 1e-6).mean()
    
    # Frobenius 范数
    fro_norm = np.linalg.norm(G_np, 'fro')
    
    # 核范数（特征值绝对值之和）
    nuclear_norm = np.sum(np.abs(eigenvalues))
    
    # Top-k 能量占比
    total_energy = np.sum(eigenvalues**2)
    if total_energy > 0:
        top1_ratio = eigenvalues[0]**2 / total_energy
        top5_ratio = np.sum(eigenvalues[:5]**2) / total_energy
        top10_ratio = np.sum(eigenvalues[:10]**2) / total_energy
    else:
        top1_ratio = top5_ratio = top10_ratio = 0
    
    return {
        'eigenvalues': eigenvalues.tolist(),
        'condition_number': float(condition_number),
        'effective_rank': float(effective_rank),
        'sparsity': float(sparsity),
        'fro_norm': float(fro_norm),
        'nuclear_norm': float(nuclear_norm),
        'top1_energy': float(top1_ratio),
        'top5_energy': float(top5_ratio),
        'top10_energy': float(top10_ratio),
        'max_eigenvalue': float(eigenvalues[0]),
        'min_eigenvalue': float(eigenvalues[-1]),
    }


def train_and_analyze(n_samples, config, epochs):
    """训练模型并分析 G 矩阵"""
    print(f"\n{'='*60}")
    print(f"N = {n_samples}, N/S^2 = {n_samples / SEQ_LEN**2:.3f}")
    print(f"{'='*60}")
    
    model = HormonicFormerV7r3(config)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    data = generate_synthetic_data(n_samples, SEQ_LEN, VOCAB_SIZE)
    
    # 训练
    total_loss = 0
    n_batches = 0
    bs = min(32, n_samples)
    
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0
        for i in range(0, n_samples, bs):
            batch = data[perm[i:i+bs]]
            if batch.shape[0] < 2:
                continue
            
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(1, n_samples // bs)
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    # 分析 G 矩阵
    G = model.blocks[0].hebbian.G
    analysis = analyze_G_matrix(G)
    analysis['n_samples'] = n_samples
    analysis['n_over_s2'] = n_samples / (SEQ_LEN ** 2)
    analysis['final_loss'] = avg_loss
    
    print(f"\n  G 矩阵分析:")
    print(f"    条件数:    {analysis['condition_number']:.2f}")
    print(f"    有效秩:    {analysis['effective_rank']:.2f}")
    print(f"    稀疏度:    {analysis['sparsity']:.4f}")
    print(f"    Top-1 能量: {analysis['top1_energy']:.4f}")
    print(f"    Top-5 能量: {analysis['top5_energy']:.4f}")
    print(f"    Top-10 能量:{analysis['top10_energy']:.4f}")
    print(f"    最大特征值: {analysis['max_eigenvalue']:.6f}")
    print(f"    最小特征值: {analysis['min_eigenvalue']:.6f}")
    
    return analysis


def main():
    print("=" * 60)
    print("定理 3 验证：Hebbian G 矩阵谱分析")
    print(f"seq_len={SEQ_LEN}, d_model={D_MODEL}, vocab={VOCAB_SIZE}")
    print(f"S^2 = {SEQ_LEN**2}, 临界点预测: N* ≈ {SEQ_LEN**2}")
    print("=" * 60)
    
    results = []
    for n in DATA_SCALES:
        analysis = train_and_analyze(n, CONFIG, EPOCHS_PER_SCALE)
        # 不保存完整特征值列表到汇总
        summary = {k: v for k, v in analysis.items() if k != 'eigenvalues'}
        results.append(summary)
    
    # 保存结果
    out_path = r'C:\Users\MR\Desktop\论文\关于场物理的神经框架\研究论文数据\theorem3_G_spectral.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")
    
    # 打印汇总表
    print(f"\n{'='*80}")
    print(f"{'N':>8} {'N/S²':>8} {'条件数':>12} {'有效秩':>8} {'稀疏度':>8} {'Top5能量':>10} {'Loss':>8}")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['n_samples']:>8d} {r['n_over_s2']:>8.3f} {r['condition_number']:>12.2f} "
              f"{r['effective_rank']:>8.2f} {r['sparsity']:>8.4f} {r['top5_energy']:>10.4f} {r['final_loss']:>8.4f}")
    
    # 关键观察
    print(f"\n{'='*60}")
    print("关键观察:")
    print(f"{'='*60}")
    
    cond_nums = [r['condition_number'] for r in results]
    ns = [r['n_samples'] for r in results]
    
    # 找条件数跃升最大的点
    max_jump = 0
    jump_idx = 0
    for i in range(1, len(cond_nums)):
        jump = cond_nums[i] / max(cond_nums[i-1], 1e-10)
        if jump > max_jump:
            max_jump = jump
            jump_idx = i
    
    if max_jump > 2:
        print(f"  条件数最大跃升: N={ns[jump_idx-1]}→{ns[jump_idx]}, "
              f"条件数 {cond_nums[jump_idx-1]:.1f}→{cond_nums[jump_idx]:.1f} ({max_jump:.1f}x)")
        print(f"  临界点估计: N* ≈ {(ns[jump_idx-1] + ns[jump_idx]) // 2}")
        print(f"  理论预测: N* ≈ S² = {SEQ_LEN**2}")
    else:
        print(f"  未检测到明显的相变跳跃（最大比值 {max_jump:.2f}x）")
        print(f"  可能需要更大的数据规模范围")

if __name__ == '__main__':
    main()
