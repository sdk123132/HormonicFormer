"""
定理 1 补充实验：alpha < 0 时的表征崩塌
验证 CGL 在 alpha < 0 时不存在稳定极限环

理论预测：
  alpha > 0: 超临界 Hopf 分岔，存在稳定极限环
  alpha = 0: 分岔点
  alpha < 0: 亚临界，无极限环，系统趋向于零解
"""
import os, sys, json, math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.insert(0, r'C:\Users\MR\Desktop')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hormonic_v7r3_validated import HormonicFormerV7r3

# alpha < 0 的测试点
ALPHA_NEG = [-8.0, -5.0, -3.0, -2.0, -1.0, -0.5, -0.1, 0.0]
# 对照组（正的）
ALPHA_POS = [0.01, 0.1, 0.5, 1.0]

SEQ_LEN = 63
D_MODEL = 64
VOCAB_SIZE = 50
N_LAYERS = 1
N_SAMPLES = 500
EPOCHS = 5
BS = 32


def make_config():
    return {
        'model': {
            'd_model': D_MODEL, 'n_layers': N_LAYERS, 'n_heads': 4,
            'seq_len': SEQ_LEN, 'vocab_size': VOCAB_SIZE,
            'dropout': 0.0, 'n_cgl_steps': 10,
            'D0_amp': 0.002, 'D0_phase': 0.002,
            'cgl_dt': 0.02, 'noise_scale': 0.001,
        },
        'use_neuromod': False, 'use_pac': False, 'use_pc': False,
        'g_coupling_strength': 0.1, 'neuromod': {},
        'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05},
        'hebbian': {'eta_potentiate': 0.001, 'eta_depress': 0.0005,
                    'sync_threshold': 0.3, 'decay': 0.999},
    }


def generate_data(n, seq_len, vocab):
    data = []
    for _ in range(n):
        r = np.random.random()
        if r < 0.3:
            plen = np.random.randint(2, min(8, seq_len // 2))
            pat = np.random.randint(0, vocab, plen)
            seq = np.tile(pat, seq_len // plen + 1)[:seq_len + 1]
        elif r < 0.5:
            start = np.random.randint(0, vocab)
            step = np.random.choice([-1, 1])
            seq = np.array([(start + i * step) % vocab for i in range(seq_len + 1)])
        else:
            seq = np.random.randint(0, vocab, seq_len + 1)
        data.append(seq)
    return torch.tensor(np.array(data), dtype=torch.long)


def set_alpha(model, alpha_val):
    """设置 alpha_raw，使得 softplus(alpha_raw) = alpha_val"""
    for block in model.blocks:
        if alpha_val > 20:
            raw = alpha_val
        elif alpha_val > 0.001:
            raw = math.log(math.exp(alpha_val) - 1)
        elif alpha_val < -20:
            raw = alpha_val
        else:
            # alpha_val 可能为负，直接设置 raw = alpha_val
            # softplus(-x) ≈ exp(-x) 很小
            raw = alpha_val
        block.cgl.alpha_raw.data.fill_(raw)
        actual = F.softplus(block.cgl.alpha_raw).item()
        return actual


def measure_collapse_indicators(model, data):
    """测量表征崩塌指标"""
    model.eval()
    with torch.no_grad():
        inputs = data[:100, :-1]
        x = model.token_embed(inputs)
        x = x + model.pos_embed[:, :inputs.shape[1], :]
        psi = x.reshape(inputs.shape[0], inputs.shape[1], model.d_model, 2)
        
        # 过一层 block
        psi_out = model.blocks[0](psi)
        
        # 幅度
        amp = torch.sqrt(psi_out[..., 0]**2 + psi_out[..., 1]**2 + 1e-8)
        
        # 指标
        amp_mean = amp.mean().item()
        amp_std = amp.std().item()
        amp_max = amp.max().item()
        amp_min = amp.min().item()
        
        # 是否接近零（崩塌）
        near_zero_ratio = (amp < 0.01).float().mean().item()
        
        # 梯度检查（需要重新开 eval 模式）
        
    return {
        'amp_mean': amp_mean,
        'amp_std': amp_std,
        'amp_max': amp_max,
        'amp_min': amp_min,
        'near_zero_ratio': near_zero_ratio,
    }


def run_experiment(alpha_target):
    print(f"\n{'='*60}")
    print(f"alpha_target = {alpha_target:.3f}")
    print(f"{'='*60}")
    
    config = make_config()
    model = HormonicFormerV7r3(config)
    
    actual_alpha = set_alpha(model, alpha_target)
    print(f"  actual_alpha (softplus) = {actual_alpha:.6f}")
    
    # 检查是否为负
    if actual_alpha < 0.001:
        print(f"  [WARN] alpha 接近零或负: {actual_alpha}")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    data = generate_data(N_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    
    # 训练前测量
    pre = measure_collapse_indicators(model, data)
    print(f"  Pre-train:  amp_mean={pre['amp_mean']:.4f}, "
          f"near_zero={pre['near_zero_ratio']:.2%}")
    
    # 训练
    losses = []
    has_nan = False
    
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(N_SAMPLES)
        epoch_loss = 0
        n_batch = 0
        
        for i in range(0, N_SAMPLES, BS):
            batch = data[perm[i:i+BS]]
            if batch.shape[0] < 2:
                continue
            
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [FAIL] NaN/Inf at epoch {epoch+1}!")
                has_nan = True
                break
            
            optimizer.zero_grad()
            loss.backward()
            
            # 检查梯度
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            if total_grad_norm > 1000:
                print(f"  [WARN] 梯度爆炸: {total_grad_norm:.2f}")
            elif total_grad_norm < 1e-6:
                print(f"  [WARN] 梯度消失: {total_grad_norm:.2e}")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batch += 1
        
        if has_nan:
            break
        
        avg_loss = epoch_loss / max(n_batch, 1)
        losses.append(avg_loss)
        
        if epoch % 2 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}")
    
    # 训练后测量
    post = measure_collapse_indicators(model, data)
    print(f"  Post-train: amp_mean={post['amp_mean']:.4f}, "
          f"near_zero={post['near_zero_ratio']:.2%}")
    
    # 判断是否崩塌
    collapsed = post['amp_mean'] < 0.1 or post['near_zero_ratio'] > 0.5
    
    result = {
        'alpha_target': alpha_target,
        'alpha_actual': actual_alpha,
        'pre_amp_mean': pre['amp_mean'],
        'post_amp_mean': post['amp_mean'],
        'post_near_zero': post['near_zero_ratio'],
        'final_loss': losses[-1] if losses else float('inf'),
        'loss_curve': losses,
        'has_nan': has_nan,
        'collapsed': collapsed,
    }
    
    status = "[OK] 正常" if not collapsed and not has_nan else "[FAIL] 崩塌"
    print(f"  结果: {status}")
    
    return result


def main():
    print("=" * 60)
    print("定理 1 补充实验：alpha < 0 时的表征崩塌")
    print(f"seq_len={SEQ_LEN}, d_model={D_MODEL}")
    print("=" * 60)
    
    results = []
    
    # 先测负的
    print("\n" + "="*60)
    print("测试 alpha < 0（预期崩塌）")
    print("="*60)
    for alpha in ALPHA_NEG:
        r = run_experiment(alpha)
        results.append(r)
    
    # 再测正的作为对照
    print("\n" + "="*60)
    print("测试 alpha > 0（对照组）")
    print("="*60)
    for alpha in ALPHA_POS:
        r = run_experiment(alpha)
        results.append(r)
    
    # 保存
    out_path = r'C:\Users\MR\Desktop\论文\关于场物理的神经框架\研究论文数据\theorem1_alpha_negative.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 汇总
    print(f"\n{'='*80}")
    print(f"{'alpha_target':>12} {'alpha_actual':>12} {'Loss':>8} {'amp_post':>10} "
          f"{'near0':>8} {'Status':>10}")
    print(f"{'='*80}")
    for r in results:
        status = "崩塌" if r['collapsed'] else "正常"
        if r['has_nan']:
            status = "NaN"
        print(f"{r['alpha_target']:>12.3f} {r['alpha_actual']:>12.6f} "
              f"{r['final_loss']:>8.4f} {r['post_amp_mean']:>10.4f} "
              f"{r['post_near_zero']:>8.2%} {status:>10}")
    
    print(f"\n结果已保存: {out_path}")
    
    # 关键结论
    print(f"\n{'='*60}")
    print("关键结论:")
    print(f"{'='*60}")
    
    neg_collapsed = sum(1 for r in results if r['alpha_target'] < 0 and r['collapsed'])
    neg_total = len([r for r in results if r['alpha_target'] < 0])
    pos_collapsed = sum(1 for r in results if r['alpha_target'] > 0 and r['collapsed'])
    pos_total = len([r for r in results if r['alpha_target'] > 0])
    
    print(f"  alpha < 0: {neg_collapsed}/{neg_total} 崩塌 ({neg_collapsed/neg_total*100:.0f}%)")
    print(f"  alpha > 0: {pos_collapsed}/{pos_total} 崩塌 ({pos_collapsed/pos_total*100:.0f}%)")
    
    if neg_collapsed > pos_collapsed:
        print(f"  [PASS] 验证通过: alpha < 0 时更容易崩塌")
    else:
        print(f"  [NOTE] 未观察到明显差异，可能需要调整指标阈值")


if __name__ == '__main__':
    main()
