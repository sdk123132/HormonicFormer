"""
定理 2 验证：G 矩阵谱收敛
验证 G 的特征向量与数据相位相关矩阵的特征向量对齐
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

SEQ_LEN = 63
D_MODEL = 64
VOCAB_SIZE = 50
N_LAYERS = 1

CONFIG = {
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


def generate_structured_data(n_samples, seq_len, vocab_size, pattern='periodic'):
    """生成具有已知相位结构的数据"""
    data = []
    
    if pattern == 'periodic':
        # 周期性模式：相位差固定
        for _ in range(n_samples):
            base = np.random.randint(0, vocab_size)
            period = np.random.randint(3, 8)
            seq = np.array([(base + i % period) % vocab_size for i in range(seq_len + 1)])
            data.append(seq)
    
    elif pattern == 'linear':
        # 线性递增：相位差恒定
        for _ in range(n_samples):
            start = np.random.randint(0, vocab_size)
            step = np.random.randint(1, 5)
            seq = np.array([(start + i * step) % vocab_size for i in range(seq_len + 1)])
            data.append(seq)
    
    elif pattern == 'random_walk':
        # 随机游走：相位差高斯分布
        for _ in range(n_samples):
            seq = [np.random.randint(0, vocab_size)]
            for _ in range(seq_len):
                step = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                seq.append((seq[-1] + step) % vocab_size)
            data.append(np.array(seq))
    
    return torch.tensor(np.array(data), dtype=torch.long)


def extract_phase_correlation(data, vocab_size):
    """从数据中提取相位相关矩阵 Sigma"""
    # 将 token ID 映射到相位 [0, 2*pi)
    phases = (data.float() / vocab_size) * 2 * math.pi
    
    n = phases.shape[0]
    S = phases.shape[1] - 1  # 去掉最后一个位置
    
    # 计算相位差矩阵
    Sigma = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            delta_phi = phases[:, i] - phases[:, j]
            # cos(相位差) 的期望
            Sigma[i, j] = torch.cos(delta_phi).mean().item()
    
    return Sigma


def compute_spectral_alignment(G, Sigma, k=5):
    """计算 G 和 Sigma 的前 k 个特征向量的对齐度"""
    # 特征分解
    eig_G, vec_G = np.linalg.eigh(G)
    eig_S, vec_S = np.linalg.eigh(Sigma)
    
    # 降序排列
    idx_G = np.argsort(eig_G)[::-1]
    idx_S = np.argsort(eig_S)[::-1]
    
    eig_G = eig_G[idx_G]
    vec_G = vec_G[:, idx_G]
    eig_S = eig_S[idx_S]
    vec_S = vec_S[:, idx_S]
    
    # 只考虑非零特征值
    nonzero_G = np.abs(eig_G) > 1e-6
    nonzero_S = np.abs(eig_S) > 1e-6
    
    results = {
        'eigenvalues_G': eig_G.tolist(),
        'eigenvalues_Sigma': eig_S.tolist(),
        'top_k_alignments': [],
        'subspace_angle': None,
    }
    
    # 计算前 k 个特征向量的对齐度（cosine of principal angles）
    k_eff = min(k, np.sum(nonzero_G), np.sum(nonzero_S))
    
    if k_eff > 0:
        # 取前 k_eff 个特征向量
        V_G = vec_G[:, :k_eff]
        V_S = vec_S[:, :k_eff]
        
        # 计算 canonical angles: cos(theta_i) = sigma_i(V_G^T V_S)
        M = V_G.T @ V_S
        _, s, _ = np.linalg.svd(M)
        
        # s[i] = cos(theta_i)，theta_i 是第 i 个主角度
        angles = np.arccos(np.clip(s, -1, 1))
        
        results['subspace_angle'] = float(np.mean(angles))
        results['top_k_alignments'] = s.tolist()
        results['k_effective'] = int(k_eff)
    
    return results


def train_and_track(model, data, n_epochs=10, checkpoint_epochs=None):
    """训练并定期记录 G 矩阵"""
    if checkpoint_epochs is None:
        checkpoint_epochs = [1, 2, 5, 10]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # 预计算数据相位相关矩阵
    Sigma = extract_phase_correlation(data, VOCAB_SIZE)
    
    results = {
        'checkpoints': [],
        'Sigma': Sigma.tolist(),
    }
    
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(data))
        epoch_loss = 0
        n_batch = 0
        
        for i in range(0, len(data), 32):
            batch = data[perm[i:i+32]]
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
            n_batch += 1
        
        avg_loss = epoch_loss / max(n_batch, 1)
        
        # 记录 checkpoint
        if (epoch + 1) in checkpoint_epochs:
            G = model.blocks[0].hebbian.G.detach().cpu().numpy()
            alignment = compute_spectral_alignment(G, Sigma, k=5)
            
            results['checkpoints'].append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'G_norm': float(np.linalg.norm(G, 'fro')),
                'alignment': alignment,
            })
            
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                  f"subspace_angle={alignment.get('subspace_angle', 'N/A'):.4f}")
    
    return results


def main():
    print("=" * 60)
    print("定理 2 验证：G 矩阵谱收敛")
    print(f"seq_len={SEQ_LEN}, d_model={D_MODEL}")
    print("=" * 60)
    
    patterns = ['periodic', 'linear', 'random_walk']
    all_results = {}
    
    for pattern in patterns:
        print(f"\n{'='*60}")
        print(f"Pattern: {pattern}")
        print(f"{'='*60}")
        
        # 生成数据
        data = generate_structured_data(1000, SEQ_LEN, VOCAB_SIZE, pattern)
        
        # 创建模型
        model = HormonicFormerV7r3(CONFIG)
        
        # 训练并跟踪
        result = train_and_track(model, data, n_epochs=10)
        all_results[pattern] = result
    
    # 保存
    out_path = r'C:\Users\MR\Desktop\论文\关于场物理的神经框架\研究论文数据\theorem2_spectral_alignment.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {out_path}")
    
    # 汇总
    print(f"\n{'='*80}")
    print("谱对齐度随训练进展:")
    print(f"{'='*80}")
    
    for pattern, res in all_results.items():
        print(f"\n{pattern}:")
        for cp in res['checkpoints']:
            angle = cp['alignment'].get('subspace_angle')
            if angle is not None:
                # 转换为度数
                angle_deg = angle * 180 / math.pi
                print(f"  Epoch {cp['epoch']:2d}: angle={angle_deg:6.2f}°, "
                      f"cos(angle)={math.cos(angle):.4f}")
            else:
                print(f"  Epoch {cp['epoch']:2d}: N/A")


if __name__ == '__main__':
    main()
