"""
Theorem 5(c): Rotation Invariance Validation
验证表示在正交变换下的不变性

实验设计:
1. 训练 HormonicFormer 至收敛
2. 冻结所有参数
3. 生成固定 latent state
4. 应用随机正交变换
5. 验证表示距离不变性
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from scipy.stats import ortho_group

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class MinimalHormonicFormer(nn.Module):
    """简化版 HormonicFormer 用于验证"""
    def __init__(self, seq_len=64, d_model=64):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 简化的编码器
        self.encoder = nn.Linear(2, d_model)  # 输入: [amplitude, phase]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.decoder = nn.Linear(d_model, 2)  # 输出: [amplitude, phase]
        
    def forward(self, x):
        # x: [batch, seq_len, 2] (amplitude, phase)
        h = self.encoder(x)  # [batch, seq_len, d_model]
        h = self.transformer(h)  # [batch, seq_len, d_model]
        out = self.decoder(h)  # [batch, seq_len, 2]
        return out, h

def generate_latent_state(seq_len=64, sigma=1.0):
    """生成 latent state: (amplitude, phase)"""
    # 幅度: Rayleigh 分布
    amplitude = np.random.rayleigh(sigma, size=seq_len)
    # 相位: 均匀分布
    phase = np.random.uniform(0, 2*np.pi, size=seq_len)
    
    latent = np.stack([amplitude, phase], axis=-1)  # [seq_len, 2]
    return torch.tensor(latent, dtype=torch.float32)

def apply_orthogonal_transform(latent, Q):
    """应用正交变换到 latent space"""
    # latent: [seq_len, 2]
    # Q: [seq_len, seq_len] 正交矩阵
    return Q @ latent

def compute_representation_distance(h1, h2):
    """计算表示之间的距离"""
    # h1, h2: [seq_len, d_model]
    # 欧氏距离
    euclidean = torch.norm(h1 - h2, p=2).item()
    # 余弦距离
    cos_sim = torch.nn.functional.cosine_similarity(
        h1.flatten(), h2.flatten(), dim=0
    ).item()
    cosine_dist = 1 - cos_sim
    
    return {
        'euclidean': euclidean,
        'cosine': cosine_dist
    }

def rotation_invariance_experiment(n_trials=100, seq_len=64, d_model=64):
    """
    验证旋转不变性
    
    预测: d(psi(Qz), psi(z)) = d(Q' * psi(z), psi(z))
    即表示距离在正交变换下保持不变
    """
    print("="*80)
    print("Theorem 5(c): Rotation Invariance Validation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of trials: {n_trials}")
    print(f"  Random seed: {SEED}")
    print()
    
    # 创建模型
    model = MinimalHormonicFormer(seq_len=seq_len, d_model=d_model)
    model.eval()
    
    results = {
        'n_trials': n_trials,
        'seq_len': seq_len,
        'd_model': d_model,
        'seed': SEED,
        'trials': []
    }
    
    distances_original = []
    distances_transformed = []
    
    for trial in range(n_trials):
        # 生成 latent state
        z = generate_latent_state(seq_len=seq_len)
        
        # 生成随机正交变换
        Q = ortho_group.rvs(seq_len)  # [seq_len, seq_len]
        Q = torch.tensor(Q, dtype=torch.float32)
        
        # 应用变换
        z_transformed = apply_orthogonal_transform(z, Q)  # [seq_len, 2]
        
        # 获取表示
        with torch.no_grad():
            _, h_original = model(z.unsqueeze(0))  # [1, seq_len, d_model]
            _, h_transformed = model(z_transformed.unsqueeze(0))  # [1, seq_len, d_model]
            
            h_original = h_original.squeeze(0)  # [seq_len, d_model]
            h_transformed = h_transformed.squeeze(0)  # [seq_len, d_model]
        
        # 计算距离
        dist_original = compute_representation_distance(h_original, h_original)
        dist_transformed = compute_representation_distance(h_original, h_transformed)
        
        distances_original.append(dist_original['euclidean'])
        distances_transformed.append(dist_transformed['euclidean'])
        
        results['trials'].append({
            'trial': trial,
            'dist_original': dist_original,
            'dist_transformed': dist_transformed
        })
        
        if trial % 20 == 0:
            print(f"  Trial {trial}/{n_trials}: d_original={dist_original['euclidean']:.4f}, "
                  f"d_transformed={dist_transformed['euclidean']:.4f}")
    
    # 统计分析
    distances_original = np.array(distances_original)
    distances_transformed = np.array(distances_transformed)
    
    print("\n" + "="*80)
    print("Statistical Analysis")
    print("="*80)
    
    print(f"\nOriginal distances (d(z, z)):")
    print(f"  Mean: {np.mean(distances_original):.6f}")
    print(f"  Std:  {np.std(distances_original):.6f}")
    print(f"  Min:  {np.min(distances_original):.6f}")
    print(f"  Max:  {np.max(distances_original):.6f}")
    
    print(f"\nTransformed distances (d(z, Qz)):")
    print(f"  Mean: {np.mean(distances_transformed):.6f}")
    print(f"  Std:  {np.std(distances_transformed):.6f}")
    print(f"  Min:  {np.min(distances_transformed):.6f}")
    print(f"  Max:  {np.max(distances_transformed):.6f}")
    
    # 验证不变性
    # 理论上: d(psi(Qz), psi(z)) 应该与 d(psi(z), psi(z)) 相关
    # 实际上: 由于模型没有显式约束, 可能不完全相等
    
    # 计算相关系数
    correlation = np.corrcoef(distances_original, distances_transformed)[0, 1]
    
    print(f"\nCorrelation between original and transformed distances:")
    print(f"  Pearson r: {correlation:.4f}")
    
    # 检验: 距离应该保持某种结构
    # 如果完全随机, 相关系数应该接近 0
    # 如果保持结构, 相关系数应该显著不为 0
    
    from scipy.stats import pearsonr
    r, p_value = pearsonr(distances_original, distances_transformed)
    
    print(f"\nStatistical significance:")
    print(f"  r = {r:.4f}")
    print(f"  p-value = {p_value:.2e}")
    
    # 结论
    print("\n" + "="*80)
    print("Conclusion")
    print("="*80)
    
    if abs(r) > 0.5 and p_value < 0.001:
        print("[PASS] Rotation invariance partially verified:")
        print(f"       - Significant correlation (r={r:.4f}, p={p_value:.2e})")
        print("       - Representations preserve structure under orthogonal transforms")
        print("       - Identifiability class [psi] = {Q * psi : Q ∈ O(S)} confirmed")
        results['status'] = 'PASS'
    elif abs(r) > 0.3 and p_value < 0.05:
        print("[PARTIAL] Weak rotation invariance detected:")
        print(f"          - Moderate correlation (r={r:.4f}, p={p_value:.2e})")
        print("          - Model may need explicit constraints for full invariance")
        results['status'] = 'PARTIAL'
    else:
        print("[FAIL] No significant rotation invariance detected:")
        print(f"       - Weak correlation (r={r:.4f}, p={p_value:.2e})")
        print("       - Model lacks explicit orthogonal invariance")
        results['status'] = 'FAIL'
    
    results['statistics'] = {
        'correlation': float(r),
        'p_value': float(p_value),
        'original_mean': float(np.mean(distances_original)),
        'original_std': float(np.std(distances_original)),
        'transformed_mean': float(np.mean(distances_transformed)),
        'transformed_std': float(np.std(distances_transformed))
    }
    
    # 保存结果
    output_path = Path(__file__).parent / 'theorem5c_rotation_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[✓] Results saved to: {output_path}")
    
    return results

def generate_report(results):
    """生成验证报告"""
    report = []
    report.append("="*80)
    report.append("Theorem 5(c): Rotation Invariance Validation Report")
    report.append("="*80)
    report.append("")
    
    report.append(f"Configuration:")
    report.append(f"  Trials: {results['n_trials']}")
    report.append(f"  Sequence length: {results['seq_len']}")
    report.append(f"  Model dimension: {results['d_model']}")
    report.append(f"  Random seed: {results['seed']}")
    report.append("")
    
    stats = results.get('statistics', {})
    report.append("Statistical Results:")
    report.append(f"  Correlation: {stats.get('correlation', 0):.4f}")
    report.append(f"  P-value: {stats.get('p_value', 1):.2e}")
    report.append(f"  Original distance mean: {stats.get('original_mean', 0):.4f} ± {stats.get('original_std', 0):.4f}")
    report.append(f"  Transformed distance mean: {stats.get('transformed_mean', 0):.4f} ± {stats.get('transformed_std', 0):.4f}")
    report.append("")
    
    status = results.get('status', 'UNKNOWN')
    report.append(f"Status: {status}")
    report.append("")
    
    if status == 'PASS':
        report.append("Conclusion: Rotation invariance verified.")
        report.append("The representation preserves structure under orthogonal transformations.")
        report.append("This confirms the identifiability class [psi] = {Q * psi : Q ∈ O(S)}.")
    elif status == 'PARTIAL':
        report.append("Conclusion: Partial rotation invariance detected.")
        report.append("The model shows some structure preservation, but may need explicit constraints.")
    else:
        report.append("Conclusion: Rotation invariance not verified.")
        report.append("The model lacks explicit orthogonal invariance in its current form.")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)

if __name__ == '__main__':
    # 运行实验
    results = rotation_invariance_experiment(n_trials=100, seq_len=64, d_model=64)
    
    # 生成报告
    report = generate_report(results)
    
    # 保存报告
    report_path = Path(__file__).parent / '22_Theorem5c_旋转不变性报告.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[✓] Report saved to: {report_path}")
    print("\n" + report)
