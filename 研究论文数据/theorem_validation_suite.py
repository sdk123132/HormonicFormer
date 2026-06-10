"""
HormonicFormer Rigorous Theory Validation Suite
验证 HormonicFormer_Rigorous_Theory_v2.docx 中的数学公式和定理
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, stats
from scipy.optimize import minimize_scalar
import json
import os
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. 基础 CGL 和 Hebbian 实现
# ============================================================================

class CGLLayer(nn.Module):
    """Complex Ginzburg-Landau 动力学层"""
    def __init__(self, d_model, alpha=1.0, beta=1.0, gamma=1.0, dt=0.01, n_steps=10):
        super().__init__()
        self.d_model = d_model
        self.alpha = alpha  # 增长率 r
        self.beta = beta      # 饱和参数
        self.gamma = gamma    # 扩散系数
        self.dt = dt
        self.n_steps = n_steps
        
    def forward(self, z):
        """
        z: complex tensor [batch, seq_len, d_model]
        返回演化后的复数场
        """
        for _ in range(self.n_steps):
            # |z|^2
            z_sq = torch.abs(z) ** 2
            # CGL 更新: dz/dt = alpha*z - beta*|z|^2*z + gamma*Laplacian(z)
            # 简化为局部动力学（无空间耦合）
            dz = self.alpha * z - self.beta * z_sq * z
            z = z + self.dt * dz
        return z

class HebbianPlasticity(nn.Module):
    """Hebbian 学习规则"""
    def __init__(self, seq_len, eta_hebb=0.001, eta_anti=0.0005, 
                 lambda_decay=0.99, theta=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.eta_hebb = eta_hebb
        self.eta_anti = eta_anti
        self.lambda_decay = lambda_decay
        self.theta = theta
        
        # 初始化 G 矩阵
        self.G = nn.Parameter(torch.zeros(seq_len, seq_len), requires_grad=False)
        
    def update(self, phi):
        """
        phi: phase tensor [batch, seq_len]
        执行 Hebbian 更新
        """
        batch_size = phi.shape[0]
        
        for b in range(batch_size):
            phi_b = phi[b]  # [seq_len]
            
            # 计算相位差
            delta_phi = phi_b.unsqueeze(0) - phi_b.unsqueeze(1)  # [seq_len, seq_len]
            
            # 同步检测
            sync_mask = (torch.abs(delta_phi) < self.theta).float()
            
            # Hebbian 更新
            delta_G = self.eta_hebb * sync_mask - self.eta_anti * (1 - sync_mask)
            
            # 衰减和更新
            self.G.data = self.lambda_decay * self.G.data + delta_G
            
            # 对角线置零
            self.G.data.fill_diagonal_(0)
            
            # 对称化
            self.G.data = 0.5 * (self.G.data + self.G.data.T)

# ============================================================================
# 2. 实验 I: 幅度恢复实验 (Theorem 5a)
# ============================================================================

def experiment_amplitude_recovery():
    """
    实验 I: 验证幅度可识别性
    预测: r* = sqrt(alpha/beta) 时，编码器可以恢复幅度
    """
    print("\n" + "="*60)
    print("实验 I: 幅度恢复实验 (Theorem 5a)")
    print("="*60)
    
    # 参数设置
    d_model = 64
    seq_len = 32
    batch_size = 100
    n_epochs = 50
    
    # 测试不同的 alpha 值
    alphas = np.linspace(0.5, 3.0, 10)
    betas = np.ones_like(alphas)  # beta = 1
    
    results = {
        'alphas': alphas.tolist(),
        'theoretical_r_star': [],
        'measured_amplitude': [],
        'recovery_error': []
    }
    
    for alpha, beta in zip(alphas, betas):
        print(f"\nTesting alpha={alpha:.3f}, beta={beta:.3f}")
        
        # 理论极限环半径
        r_star_theory = np.sqrt(alpha / beta)
        results['theoretical_r_star'].append(r_star_theory)
        
        # 创建 CGL 层
        cgl = CGLLayer(d_model, alpha=alpha, beta=beta, n_steps=20).to(device)
        
        # 生成测试数据: 已知幅度
        true_amplitudes = torch.rand(batch_size, seq_len, 1).to(device) * 2.0
        
        # 随机相位
        phases = torch.rand(batch_size, seq_len, d_model).to(device) * 2 * np.pi
        
        # 构建复数场: z = r * exp(i*phi)
        z = true_amplitudes * torch.exp(1j * phases)
        
        # CGL 演化
        z_evolved = cgl(z)
        
        # 测量演化后的幅度
        measured_amp = torch.abs(z_evolved).mean(dim=(0, 2)).cpu().numpy()
        
        # 计算恢复误差
        recovery_error = np.abs(measured_amp.mean() - r_star_theory) / r_star_theory
        
        results['measured_amplitude'].append(measured_amp.mean())
        results['recovery_error'].append(recovery_error)
        
        print(f"  Theory r* = {r_star_theory:.4f}")
        print(f"  Measured amp = {measured_amp.mean():.4f}")
        print(f"  Recovery error = {recovery_error:.4f}")
    
    # 保存结果
    with open('experiment_I_amplitude_recovery.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n实验 I 完成。结果保存到 experiment_I_amplitude_recovery.json")
    return results

# ============================================================================
# 3. 实验 II: 相位恢复实验 (Theorem 5b)
# ============================================================================

def experiment_phase_recovery():
    """
    实验 II: 验证相位可识别性
    预测: 相位可以恢复，但存在全局旋转不确定性
    """
    print("\n" + "="*60)
    print("实验 II: 相位恢复实验 (Theorem 5b)")
    print("="*60)
    
    seq_len = 32
    batch_size = 100
    n_trials = 10
    
    results = {
        'phase_errors': [],
        'relative_phase_errors': [],
        'global_offsets': []
    }
    
    for trial in range(n_trials):
        print(f"\nTrial {trial+1}/{n_trials}")
        
        # 生成真实相位
        true_phases = torch.rand(batch_size, seq_len).to(device) * 2 * np.pi
        
        # 添加 Laplacian 耦合模拟相位同步
        # 简化的相位动力学
        L = torch.diag(torch.ones(seq_len)) - torch.ones(seq_len, seq_len) / seq_len
        L = L.to(device)
        
        # 演化后的相位（带扩散）
        evolved_phases = []
        for b in range(batch_size):
            phi = true_phases[b]
            for _ in range(10):  # 扩散步骤
                phi = phi - 0.01 * (L @ phi)
            evolved_phases.append(phi)
        
        evolved_phases = torch.stack(evolved_phases)
        
        # 估计全局偏移
        global_offset = (evolved_phases - true_phases).mean(dim=1, keepdim=True)
        
        # 校正后的相位
        corrected_phases = evolved_phases - global_offset
        
        # 计算误差
        phase_error = torch.abs(corrected_phases - true_phases).mean().item()
        
        # 相对相位误差（不受全局偏移影响）
        rel_true = torch.diff(true_phases, dim=1)
        rel_evolved = torch.diff(evolved_phases, dim=1)
        rel_error = torch.abs(rel_evolved - rel_true).mean().item()
        
        results['phase_errors'].append(phase_error)
        results['relative_phase_errors'].append(rel_error)
        results['global_offsets'].append(global_offset.mean().item())
        
        print(f"  Phase error (corrected): {phase_error:.6f}")
        print(f"  Relative phase error: {rel_error:.6f}")
        print(f"  Global offset: {global_offset.mean().item():.6f}")
    
    # 统计结果
    print("\n统计结果:")
    print(f"  平均相位误差: {np.mean(results['phase_errors']):.6f}")
    print(f"  平均相对相位误差: {np.mean(results['relative_phase_errors']):.6f}")
    print(f"  全局偏移标准差: {np.std(results['global_offsets']):.6f}")
    
    with open('experiment_II_phase_recovery.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n实验 II 完成。结果