"""
定理 1 验证：CGL 极限环半径 vs 表征容量
扫描 alpha 值，观察极限环半径对模型性能的影响
纯 CPU，不影响 GPU 训练

核心假设：
  极限环半径 r* = sqrt(alpha) 控制表征容量
  alpha < 0: 无极限环（表征崩塌）
  alpha > 0: r* = sqrt(alpha)，越大容量越高，但太大会不稳定
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
SEQ_LEN = 63  # 64-1 for next-token prediction
D_MODEL = 64
VOCAB_SIZE = 50
N_LAYERS = 1
N_SAMPLES = 2000
EPOCHS = 8
BS = 32

# alpha 扫描范围：从接近 0 到较大值
# softplus(x) = ln(1+exp(x)), 初始 x=0.54 时 softplus≈1.0
# 要控制 alpha，需要设置 CGL 内部的 alpha 初始值
ALPHA_INITS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]


def make_config(alpha_init):
    return {
        'model': {
            'd_model': D_MODEL,
            'n_layers': N_LAYERS,
            'n_heads': 4,
            'seq_len': SEQ_LEN,
            'vocab_size': VOCAB_SIZE,
            'dropout': 0.0,
            'n_cgl_steps': 10,
            'D0_amp': 0.002,
            'D0_phase': 0.002,
            'cgl_dt': 0.02,
            'noise_scale': 0.001,
        },
        'use_neuromod': False,  # 关闭调质，隔离 CGL 效应
        'use_pac': False,
        'use_pc': False,
        'g_coupling_strength': 0.1,
        'neuromod': {},
        'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05},
        'hebbian': {'eta_potentiate': 0.001, 'eta_depress': 0.0005,
                    'sync_threshold': 0.3, 'decay': 0.999},
    }


def generate_data(n_samples, seq_len, vocab_size):
    """生成有结构的合成序列"""
    data = []
    for _ in range(n_samples):
        r = np.random.random()
        if r < 0.3:
            # 重复模式
            plen = np.random.randint(2, min(8, seq_len // 2))
            pat = np.random.randint(0, vocab_size, plen)
            seq = np.tile(pat, seq_len // plen + 1)[:seq_len + 1]
        elif r < 0.5:
            # 递增
            start = np.random.randint(0, vocab_size)
            step = np.random.choice([-1, 1])
            seq = np.array([(start + i * step) % vocab_size for i in range(seq_len + 1)])
        else:
            seq = np.random.randint(0, vocab_size, seq_len + 1)
        data.append(seq)
    return torch.tensor(np.array(data), dtype=torch.long)


def set_alpha(model, alpha_val):
    """强制设置 CGL 的 alpha 参数
    注意: actual_alpha = softplus(alpha_raw)
    要得到 actual_alpha = alpha_val, 需要 alpha_raw = softplus_inv(alpha_val)
    """
    for block in model.blocks:
        if alpha_val > 20:
            raw = alpha_val
        elif alpha_val > 0.001:
            raw = math.log(math.exp(alpha_val) - 1)
        else:
            raw = -10.0
        block.cgl.alpha_raw.data.fill_(raw)
        # 验证
        actual = F.softplus(block.cgl.alpha_raw).item()
        print(f"  set alpha_raw={raw:.4f} -> softplus={actual:.4f} (target={alpha_val:.4f})")


def measure_representation(model, data):
    """测量表征质量：相位多样性和幅度稳定性"""
    model.eval()
    with torch.no_grad():
        inputs = data[:200, :-1]
        x = model.token_embed(inputs)
        x = x + model.pos_embed[:, :inputs.shape[1], :]
        psi = x.reshape(inputs.shape[0], inputs.shape[1], model.d_model, 2)
        
        # 过一层 block
        psi_out = model.blocks[0](psi)
        
        # 幅度
        amp = torch.sqrt(psi_out[..., 0]**2 + psi_out[..., 1]**2 + 1e-8)
        # 相位
        phase = torch.atan2(psi_out[..., 1], psi_out[..., 0] + 1e-8)
        
        # 指标 1: 幅度均值和标准差
        amp_mean = amp.mean().item()
        amp_std = amp.std().item()
        
        # 指标 2: 相位多样性（entropy-like）
        # 将相位离散化到 32 个 bin
        phase_flat = phase.reshape(-1)
        bins = torch.histc(phase_flat, bins=32, min=-math.pi, max=math.pi)
        bins = bins / bins.sum()
        bins = bins[bins > 0]
        phase_entropy = -torch.sum(bins * torch.log(bins)).item()
        max_entropy = math.log(32)  # uniform distribution
        phase_diversity = phase_entropy / max_entropy  # normalized [0, 1]
        
        # 指标 3: 极限环半径（实际）
        limit_cycle_r = amp_mean
        
        # 指标 4: 幅度变异系数（CV = std/mean）
        amp_cv = amp_std / (amp_mean + 1e-8)
        
        # 指标 5: 有效维度（基于 singular values of psi）
        psi_flat = psi_out.reshape(inputs.shape[0] * inputs.shape[1], -1)
        try:
            sv = torch.linalg.svdvals(psi_flat[:500])
            sv_norm = sv / sv.sum()
            sv_entropy = -torch.sum(sv_norm * torch.log(sv_norm + 1e-15)).item()
            eff_dim = math.exp(sv_entropy)
        except:
            eff_dim = 0
        
    return {
        'amp_mean': amp_mean,
        'amp_std': amp_std,
        'amp_cv': amp_cv,
        'phase_diversity': phase_diversity,
        'limit_cycle_r': limit_cycle_r,
        'effective_dim': eff_dim,
    }


def run_experiment(alpha_init):
    """对一个 alpha 值跑完整实验"""
    print(f"\n{'='*60}")
    r_star = math.sqrt(alpha_init) if alpha_init > 0 else 0
    print(f"alpha = {alpha_init:.3f}, r* = {r_star:.3f}")
    print(f"{'='*60}")
    
    config = make_config(alpha_init)
    model = HormonicFormerV7r3(config)
    set_alpha(model, alpha_init)
    
    # 验证 alpha 设置
    actual_alpha = model.blocks[0].cgl.alpha.item()
    actual_softplus = math.log(1 + math.exp(actual_alpha)) if actual_alpha < 20 else actual_alpha
    print(f"  raw_alpha = {actual_alpha:.4f}, softplus(raw) = {actual_softplus:.4f}")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    data = generate_data(N_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    
    # 训练前测量
    pre_repr = measure_representation(model, data)
    
    # 训练
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(N_SAMPLES)
        epoch_loss = 0
        n_batch = 0
        correct = 0
        total = 0
        
        for i in range(0, N_SAMPLES, BS):
            batch = data[perm[i:i+BS]]
            if batch.shape[0] < 2:
                continue
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            
            # 检查 NaN
            if torch.isnan(loss):
                print(f"  NaN at epoch {epoch+1}!")
                return {
                    'alpha_init': alpha_init,
                    'r_star': r_star,
                    'status': 'NaN',
                    'best_loss': float('inf'),
                    'best_acc': 0,
                }
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            _, pred = logits.reshape(-1, VOCAB_SIZE).max(1)
            correct += pred.eq(targets.reshape(-1)).sum().item()
            total += targets.numel()
            n_batch += 1
        
        avg_loss = epoch_loss / max(n_batch, 1)
        acc = 100.0 * correct / max(total, 1)
        loss_history.append(avg_loss)
        best_loss = min(best_loss, avg_loss)
        
        if epoch % 3 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    
    # 训练后测量
    post_repr = measure_representation(model, data)
    
    # 训练后的 alpha（可能被优化器改变）
    final_alpha = model.blocks[0].cgl.alpha.item()  # property 已经做了 softplus
    final_r = math.sqrt(max(final_alpha, 0))
    
    result = {
        'alpha_init': alpha_init,
        'r_star_init': r_star,
        'alpha_final': final_alpha,
        'r_star_final': final_r,
        'best_loss': best_loss,
        'final_loss': loss_history[-1],
        'best_acc': acc,
        'loss_curve': loss_history,
        'status': 'OK',
        # 表征指标
        'pre_amp_mean': pre_repr['amp_mean'],
        'pre_phase_diversity': pre_repr['phase_diversity'],
        'pre_effective_dim': pre_repr['effective_dim'],
        'post_amp_mean': post_repr['amp_mean'],
        'post_phase_diversity': post_repr['phase_diversity'],
        'post_effective_dim': post_repr['effective_dim'],
        'post_amp_cv': post_repr['amp_cv'],
        'post_limit_cycle_r': post_repr['limit_cycle_r'],
    }
    
    print(f"\n  结果:")
    print(f"    alpha: {alpha_init:.3f} -> {final_alpha:.3f}")
    print(f"    r*:    {r_star:.3f} -> {final_r:.3f}")
    print(f"    Loss:  {best_loss:.4f}")
    print(f"    Acc:   {acc:.2f}%")
    print(f"    相位多样性: {post_repr['phase_diversity']:.4f}")
    print(f"    有效维度:   {post_repr['effective_dim']:.2f}")
    print(f"    幅度均值:   {post_repr['amp_mean']:.4f}")
    
    return result


def main():
    print("=" * 60)
    print("定理 1 验证：CGL 极限环半径 vs 表征容量")
    print(f"seq_len={SEQ_LEN}, d_model={D_MODEL}, vocab={VOCAB_SIZE}")
    print(f"N={N_SAMPLES}, epochs={EPOCHS}, neuromod=OFF")
    print("=" * 60)
    
    results = []
    for alpha in ALPHA_INITS:
        r = run_experiment(alpha)
        results.append(r)
    
    # 保存
    out_path = r'C:\Users\MR\Desktop\论文\关于场物理的神经框架\研究论文数据\theorem1_alpha_sweep.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 汇总
    print(f"\n{'='*80}")
    print(f"{'alpha':>8} {'r*':>8} {'alpha_f':>8} {'r*_f':>8} {'Loss':>8} {'Acc':>8} {'PhasDiv':>8} {'EffDim':>8}")
    print(f"{'='*80}")
    for r in results:
        if r['status'] == 'NaN':
            print(f"{r['alpha_init']:>8.3f} {r['r_star']:>8.3f} {'NaN':>8} {'NaN':>8} {'NaN':>8} {'NaN':>8} {'NaN':>8} {'NaN':>8}")
        else:
            print(f"{r['alpha_init']:>8.3f} {r['r_star_init']:>8.3f} "
                  f"{r['alpha_final']:>8.3f} {r['r_star_final']:>8.3f} "
                  f"{r['best_loss']:>8.4f} {r['best_acc']:>8.2f} "
                  f"{r['post_phase_diversity']:>8.4f} {r['post_effective_dim']:>8.2f}")
    
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
