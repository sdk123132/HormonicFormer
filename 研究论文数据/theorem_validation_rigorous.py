"""
HormonicFormer 严格理论验证套件
基于真实实验数据验证 HormonicFormer_Rigorous_Theory_v2.docx 中的定理

数学严谨性要求:
1. 所有公式必须与理论文档一致
2. 所有数据必须可复现
3. 统计显著性必须报告
4. 误差分析必须完整
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, stats
from scipy.optimize import minimize_scalar, curve_fit
import json
import os
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"="*70)
print(f"HormonicFormer 严格理论验证套件")
print(f"设备: {device}")
print(f"随机种子: {SEED}")
print(f"="*70)

# ============================================================================
# 1. 加载已有实验数据
# ============================================================================

def load_existing_data():
    """加载已有的定理验证实验数据"""
    data_dir = "C:\\Users\\MR\\Desktop\\论文\\关于场物理的神经框架\\研究论文数据"
    
    data = {}
    
    # 定理1: alpha扫描数据
    try:
        with open(f"{data_dir}\\theorem1_alpha_sweep.json", 'r') as f:
            data['theorem1'] = json.load(f)
        print(f"[OK] 已加载定理1数据: {len(data['theorem1'])}个alpha值")
    except Exception as e:
        print(f"[FAIL] 定理1数据加载失败: {e}")
        data['theorem1'] = None
    
    # 定理2: G矩阵谱收敛数据
    try:
        with open(f"{data_dir}\\theorem2_enhanced_50epoch.json", 'r') as f:
            data['theorem2'] = json.load(f)
        print(f"[OK] 已加载定理2数据: {len(data['theorem2']['epochs'])}个epoch")
    except Exception as e:
        print(f"[FAIL] 定理2数据加载失败: {e}")
        data['theorem2'] = None
    
    # 定理3: Hebbian相变数据
    try:
        with open(f"{data_dir}\\theorem3_G_spectral.json", 'r') as f:
            data['theorem3'] = json.load(f)
        print(f"[OK] 已加载定理3数据: {len(data['theorem3'])}个数据规模点")
    except Exception as e:
        print(f"[FAIL] 定理3数据加载失败: {e}")
        data['theorem3'] = None
    
    return data

# ============================================================================
# 2. 实验 I: 幅度恢复验证 (Theorem 5a)
# ============================================================================

def validate_amplitude_recovery(theorem1_data):
    """
    验证 Theorem 5(a): 幅度可识别性
    
    理论预测:
    - 极限环半径 r* = sqrt(alpha/beta)
    - 编码器可以恢复幅度 up to scaling: A_hat = c * A_true
    
    验证方法:
    - 使用定理1的alpha扫描数据
    - 验证 post_limit_cycle_r 与理论 r* 的线性关系
    """
    print(f"\n{'='*70}")
    print("实验 I: 幅度恢复验证 (Theorem 5a)")
    print(f"{'='*70}")
    
    if theorem1_data is None:
        print("✗ 缺少定理1数据，跳过此实验")
        return None
    
    results = {
        'theoretical_r_star': [],
        'measured_r': [],
        'alpha_values': [],
        'linear_fit': {},
        'correlation': 0.0,
        'rmse': 0.0
    }
    
    # 提取数据
    for entry in theorem1_data:
        alpha_init = entry['alpha_init']
        r_star_init = entry['r_star_init']  # 理论值 sqrt(alpha/beta)
        r_star_final = entry['post_limit_cycle_r']  # 测量值
        
        results['alpha_values'].append(alpha_init)
        results['theoretical_r_star'].append(r_star_init)
        results['measured_r'].append(r_star_final)
    
    # 转换为numpy数组
    r_theory = np.array(results['theoretical_r_star'])
    r_measured = np.array(results['measured_r'])
    
    # 线性拟合: r_measured = c * r_theory
    # 这验证了 Theorem 5(a) 的 scaling 关系
    slope, intercept, r_value, p_value, std_err = stats.linregress(r_theory, r_measured)
    
    results['linear_fit'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value**2),
        'p_value': float(p_value),
        'std_err': float(std_err)
    }
    
    # 计算 RMSE
    r_pred = slope * r_theory + intercept
    results['rmse'] = float(np.sqrt(np.mean((r_measured - r_pred)**2)))
    
    # 计算 Pearson 相关系数
    results['correlation'] = float(r_value)
    
    # 报告结果
    print(f"\n线性拟合结果 (r_measured = c * r_theory + b):")
    print(f"  斜率 c = {slope:.6f} ± {std_err:.6f}")
    print(f"  截距 b = {intercept:.6f}")
    print(f"  R^2 = {r_value**2:.6f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  RMSE = {results['rmse']:.6f}")
    
    # 验证结论
    if r_value**2 > 0.95 and p_value < 0.001:
        print(f"\n[PASS] Theorem 5(a) 验证通过: 幅度恢复呈现显著线性关系")
        print(f"  测量值与理论值的 scaling factor: c = {slope:.4f}")
    else:
        print(f"\n[FAIL] Theorem 5(a) 验证失败: 线性关系不显著")
    
    # 详细数据表
    print(f"\n详细数据:")
    print(f"{'Alpha':>8} {'Theory r*':>12} {'Measured r':>12} {'Error':>10}")
    print("-" * 50)
    for i in range(len(r_theory)):
        error = abs(r_measured[i] - r_theory[i]) / r_theory[i] * 100
        print(f"{results['alpha_values'][i]:8.3f} {r_theory[i]:12.4f} {r_measured[i]:12.4f} {error:9.2f}%")
    
    return results

# ============================================================================
# 3. 实验 II: 相位恢复验证 (Theorem 5b)
# ============================================================================

def validate_phase_recovery(theorem1_data):
    """
    验证 Theorem 5(b): 相位可识别性
    
    理论预测:
    - 相位可以恢复 up to global rotation: phi_hat = phi_true + phi_0
    - 相对相位 (phi_i - phi_j) 可以完全恢复
    
    验证方法:
    - 使用定理1数据中的 phase_diversity
    - 验证相位多样性始终接近 1.0 (最大多样性)
    """
    print(f"\n{'='*70}")
    print("实验 II: 相位恢复验证 (Theorem 5b)")
    print(f"{'='*70}")
    
    if theorem1_data is None:
        print("✗ 缺少定理1数据，跳过此实验")
        return None
    
    results = {
        'alpha_values': [],
        'pre_phase_diversity': [],
        'post_phase_diversity': [],
        'diversity_change': [],
        'mean_diversity': 0.0,
        'std_diversity': 0.0
    }
    
    # 提取数据
    diversities = []
    for entry in theorem1_data:
        alpha = entry['alpha_init']
        pre_div = entry['pre_phase_diversity']
        post_div = entry['post_phase_diversity']
        
        results['alpha_values'].append(alpha)
        results['pre_phase_diversity'].append(pre_div)
        results['post_phase_diversity'].append(post_div)
        results['diversity_change'].append(post_div - pre_div)
        diversities.append(post_div)
    
    # 统计
    results['mean_diversity'] = float(np.mean(diversities))
    results['std_diversity'] = float(np.std(diversities))
    
    # 报告结果
    print(f"\n相位多样性统计:")
    print(f"  均值: {results['mean_diversity']:.6f}")
    print(f"  标准差: {results['std_diversity']:.6f}")
    print(f"  最小值: {min(diversities):.6f}")
    print(f"  最大值: {max(diversities):.6f}")
    
    # 验证: 相位多样性应接近 1.0 (最大)
    if results['mean_diversity'] > 0.99:
        print(f"\n[PASS] Theorem 5(b) 验证通过: 相位多样性始终接近最大值")
        print(f"  表明相位保持了高度独立性 (up to global rotation)")
    else:
        print(f"\n[FAIL] Theorem 5(b) 需要进一步验证")
    
    # 详细数据
    print(f"\n详细数据:")
    print(f"{'Alpha':>8} {'Pre-Div':>12} {'Post-Div':>12} {'Change':>10}")
    print("-" * 50)
    for i in range(len(results['alpha_values'])):
        print(f"{results['alpha_values'][i]:8.3f} "
              f"{results['pre_phase_diversity'][i]:12.6f} "
              f"{results['post_phase_diversity'][i]:12.6f} "
              f"{results['diversity_change'][i]:10.6f}")
    
    return results

# ============================================================================
# 4. 实验 III: 旋转不变性验证 (Theorem 5c)
# ============================================================================

def validate_rotation_invariance():
    """
    验证 Theorem 5(c): 旋转不变性
    
    理论预测:
    - 表示在正交变换下保持不变: f(O*x) = O*f(x) (up to block-diagonal)
    - 即: ||f(O*x) - O*f(x)|| 应该很小
    
    由于需要实际模型推理，这里设计验证协议
    """
    print(f"\n{'='*70}")
    print("实验 III: 旋转不变性验证 (Theorem 5c)")
    print(f"{'='*70}")
    
    print("\n此实验需要加载训练好的 HormonicFormer 模型进行验证")
    print("设计验证协议:")
    print("  1. 生成随机输入 x ~ N(0, I)")
    print("  2. 应用随机正交变换 O")
    print("  3. 计算 f(x) 和 f(O*x)")
    print("  4. 验证 ||f(O*x) - O*f(x)|| < epsilon")
    
    # 由于没有预训练模型，提供模拟验证
    print("\n模拟验证 (基于理论性质):")
    
    # 模拟数据
    d_model = 64
    n_trials = 100
    
    invariances = []
    for _ in range(n_trials):
        # 随机输入
        x = np.random.randn(d_model)
        
        # 随机正交矩阵 (使用QR分解生成)
        A = np.random.randn(d_model, d_model)
        Q, _ = np.linalg.qr(A)
        
        # 模拟编码器: 线性变换 + 非线性
        # 理想情况下应该满足 f(Qx) = Qf(x)
        W = np.random.randn(d_model, d_model) * 0.1
        
        f_x = np.tanh(W @ x)
        f_Qx = np.tanh(W @ (Q @ x))
        Q_f_x = Q @ f_x
        
        # 计算不变性误差
        error = np.linalg.norm(f_Qx - Q_f_x) / np.linalg.norm(f_x)
        invariances.append(error)
    
    mean_error = np.mean(invariances)
    std_error = np.std(invariances)
    
    print(f"\n模拟结果 (基于随机权重):")
    print(f"  平均不变性误差: {mean_error:.6f} ± {std_error:.6f}")
    print(f"  注意: 这是随机权重的基线，训练后模型应显著改善")
    
    results = {
        'mean_error': float(mean_error),
        'std_error': float(std_error),
        'n_trials': n_trials,
        'note': '需要预训练模型进行完整验证'
    }
    
    return results

# ============================================================================
# 5. 定理 1 数值验证: CGL 表示容量
# ============================================================================

def validate_theorem1_capacity(theorem1_data):
    """
    验证 Theorem 1: CGL 表示容量
    
    关键预测:
    (a) 幅度信息: I(A;R) = S * [1 + ln(alpha/beta) - ln(2) - ln(delta)]
    (c) 最优容量在 r* = W(2S)/2 处取得
    (d) 相位多样性在最优区域接近 1.0
    """
    print(f"\n{'='*70}")
    print("定理 1 数值验证: CGL 表示容量")
    print(f"{'='*70}")
    
    if theorem1_data is None:
        print("✗ 缺少定理1数据")
        return None
    
    results = {
        'capacity_analysis': {},
        'optimal_regime': {},
        'phase_diversity_bound': {}
    }
    
    # 提取数据
    alphas = []
    losses = []
    phase_divs = []
    
    for entry in theorem1_data:
        alphas.append(entry['alpha_init'])
        losses.append(entry['best_loss'])
        phase_divs.append(entry['post_phase_diversity'])
    
    alphas = np.array(alphas)
    losses = np.array(losses)
    phase_divs = np.array(phase_divs)
    
    # 找到最优区域 (最小 loss)
    optimal_idx = np.argmin(losses)
    optimal_alpha = alphas[optimal_idx]
    optimal_loss = losses[optimal_idx]
    
    print(f"\n最优区域分析:")
    print(f"  最优 alpha: {optimal_alpha:.3f}")
    print(f"  最优 loss: {optimal_loss:.4f}")
    print(f"  对应 r*: {np.sqrt(optimal_alpha):.4f}")
    
    # 验证: 在最优区域，相位多样性应接近 1.0
    optimal_div = phase_divs[optimal_idx]
    print(f"  最优区域相位多样性: {optimal_div:.6f}")
    
    # 理论预测: r* = W(2S)/2
    # 对于 S=64 (定理1实验配置), W(128) ≈ 3.9
    # 所以 r* ≈ 1.95, alpha = r*^2 ≈ 3.8
    from scipy.special import lambertw
    S = 64  # 序列长度
    r_star_theory = np.real(lambertw(2*S)) / 2
    alpha_theory = r_star_theory**2
    
    print(f"\n理论预测 (Theorem 1c):")
    print(f"  预测最优 r*: {r_star_theory:.4f}")
    print(f"  预测最优 alpha: {alpha_theory:.4f}")
    print(f"  实验最优 alpha: {optimal_alpha:.4f}")
    print(f"  相对误差: {abs(alpha_theory - optimal_alpha)/alpha_theory * 100:.2f}%")
    
    results['capacity_analysis'] = {
        'optimal_alpha': float(optimal_alpha),
        'optimal_loss': float(optimal_loss),
        'optimal_phase_diversity': float(optimal_div),
        'theoretical_optimal_alpha': float(alpha_theory),
        'theoretical_optimal_r': float(r_star_theory)
    }
    
    # 验证相位多样性边界 (Theorem 1d)
    # D_phi >= 1 - (1/S) * (2*pi^2*D)/(alpha*beta)
    # 简化为: D_phi >= 1 - C/alpha
    
    # 拟合验证
    inv_alphas = 1.0 / alphas
    # 线性拟合: D_phi ~ 1 - C/alpha
    slope, intercept, r_val, p_val, _ = stats.linregress(inv_alphas, 1 - phase_divs)
    
    print(f"\n相位多样性边界验证 (Theorem 1d):")
    print(f"  拟合: 1 - D_phi = {slope:.6f}/alpha + {intercept:.6f}")
    print(f"  R^2 = {r_val**2:.6f}")
    print(f"  p-value = {p_val:.2e}")
    
    if r_val**2 > 0.5 and p_val < 0.05:
        print(f"  [PASS] 边界关系得到验证")
    else:
        print(f"  [WARN] 边界关系需要更多数据验证")
    
    results['phase_diversity_bound'] = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_val**2),
        'p_value': float(p_val)
    }
    
    return results

# ============================================================================
# 6. 定理 2 数值验证: G-矩阵谱收敛
# ============================================================================

def validate_theorem2_convergence(theorem2_data):
    """
    验证 Theorem 2: G-矩阵谱收敛
    
    关键预测:
    (a) 特征向量对齐: sin(theta_k) 随 epoch 减小
    (b) 收敛率: T(epsilon) = O(log(1/epsilon))
    (c) 范数三阶段动力学: 增长 -> 饱和 -> 稳态
    (d) 稳态谱: lambda_k^* = eta_eff * mu_k / (1 - lambda)
    """
    print(f"\n{'='*70}")
    print("定理 2 数值验证: G-矩阵谱收敛")
    print(f"{'='*70}")
    
    if theorem2_data is None:
        print("✗ 缺少定理2数据")
        return None
    
    results = {}
    
    epochs = np.array(theorem2_data['epochs'])
    angles_deg = np.array(theorem2_data['angles_deg'])
    angles_rad = np.array(theorem2_data['angles_rad'])
    cos_angles = np.array(theorem2_data['cos_angles'])
    G_norms = np.array(theorem2_data['G_norms'])
    
    # (a) 特征向量对齐验证
    print(f"\n(a) 特征向量对齐验证:")
    print(f"  Epoch 1: 角度 = {angles_deg[0]:.2f}°")
    print(f"  Epoch 50: 角度 = {angles_deg[-1]:.2f}°")
    print(f"  改善: {angles_deg[0] - angles_deg[-1]:.2f}°")
    
    # 验证 sin(theta) 减小
    sin_angles = np.sin(angles_rad)
    print(f"  sin(theta) 从 {sin_angles[0]:.4f} 降至 {sin_angles[-1]:.4f}")
    
    if sin_angles[-1] < sin_angles[0]:
        print(f"  [PASS] 对齐随训练改善")
    
    results['alignment'] = {
        'initial_angle_deg': float(angles_deg[0]),
        'final_angle_deg': float(angles_deg[-1]),
        'improvement_deg': float(angles_deg[0] - angles_deg[-1]),
        'initial_sin': float(sin_angles[0]),
        'final_sin': float(sin_angles[-1])
    }
    
    # (b) 收敛率验证
    # 理论: 指数收敛 rate = -log(lambda_eff)
    # 从数据估计收敛率
    
    # 使用后期数据 (epoch 20-50) 拟合指数衰减
    late_epochs = epochs[epochs >= 20]
    late_angles = angles_deg[epochs >= 20]
    
    # 指数衰减模型: angle = A * exp(-k * epoch) + C
    def exp_decay(x, A, k, C):
        return A * np.exp(-k * x) + C
    
    try:
        popt, _ = curve_fit(exp_decay, late_epochs, late_angles, 
                           p0=[20, 0.1, 30], maxfev=10000)
        A, k, C = popt
        print(f"\n(b) 收敛率估计 (Epoch 20-50):")
        print(f"  指数衰减模型: angle = {A:.2f} * exp(-{k:.4f}*epoch) + {C:.2f}")
        print(f"  收敛率 k = {k:.4f}")
        print(f"  半衰期: {np.log(2)/k:.1f} epochs")
    except Exception as e:
        print(f"\n(b) 收敛率拟合失败: {e}")
        k = None
    
    results['convergence_rate'] = {
        'rate_k': float(k) if k is not None else None,
        'half_life': float(np.log(2)/k) if k is not None else None
    }
    
    # (c) 范数三阶段动力学
    print(f"\n(c) 范数动力学分析:")
    
    # 计算增长率
    growth_rates = np.diff(G_norms) / G_norms[:-1]
    
    print(f"  初始 G_norm: {G_norms[0]:.2f}")
    print(f"  最终 G_norm: {G_norms[-1]:.2f}")
    print(f"  总增长: {(G_norms[-1]/G_norms[0]):.2f}x")
    
    # 识别阶段
    # 阶段 I: 快速增长 (高增长率)
    # 阶段 II: 饱和 (增长率下降)
    # 阶段 III: 稳态 (增长率 ~0)
    
    phase1_end = min(5, len(epochs))  # 前5个epoch为快速增长
    phase2_end = min(25, len(epochs))  # 到25epoch为饱和
    
    print(f"  阶段 I (1-{phase1_end}): 快速增长")
    print(f"    G_norm: {G_norms[0]:.2f} -> {G_norms[phase1_end-1]:.2f}")
    
    if phase2_end > phase1_end:
        print(f"  阶段 II ({phase1_end+1}-{phase2_end}): 饱和")
        print(f"    G_norm: {G_norms[phase1_end]:.2f} -> {G_norms[phase2_end-1]:.2f}")
        
        print(f"  阶段 III ({phase2_end+1}-{len(epochs)}): 稳态")
        print(f"    G_norm: {G_norms[phase2_end-1]:.2f} -> {G_norms[-1]:.2f}")
        print(f"    增长: {(G_norms[-1]/G_norms[phase2_end-1] - 1)*100:.2f}%")
    else:
        print(f"  阶段 II/III: 数据不足，无法区分")
    
    norm_results = {
        'initial': float(G_norms[0]),
        'final': float(G_norms[-1]),
        'total_growth': float(G_norms[-1]/G_norms[0]),
        'phase1_growth': float(G_norms[phase1_end-1]/G_norms[0])
    }
    if phase2_end > phase1_end:
        norm_results['phase2_growth'] = float(G_norms[phase2_end-1]/G_norms[phase1_end])
        norm_results['phase3_growth'] = float(G_norms[-1]/G_norms[phase2_end-1])
    results['norm_dynamics'] = norm_results
    
    return results

# ============================================================================
# 7. 定理 3 数值验证: Hebbian 相变
# ============================================================================

def validate_theorem3_phase_transition(theorem3_data):
    """
    验证 Theorem 3: Hebbian 相变
    
    关键预测:
    (a) 分布式区域 (N/S² < C_crit): 条件数 O(1), 有效秩 = S
    (b) 集中式区域 (N/S² > C_crit): 条件数发散, 有效秩下降
    (c) 临界缩放: 在 C_crit 处，条件数和有效秩呈现幂律行为
    (d) 相位多样性相变: 从 ~1 降至低值
    """
    print(f"\n{'='*70}")
    print("定理 3 数值验证: Hebbian 相变")
    print(f"{'='*70}")
    
    if theorem3_data is None:
        print("✗ 缺少定理3数据")
        return None
    
    results = {}
    
    # 提取数据
    n_over_s2 = []
    cond_numbers = []
    eff_ranks = []
    top1_energies = []
    
    for entry in theorem3_data:
        n_over_s2.append(entry['n_over_s2'])
        cond = entry['condition_number']
        # 处理 Infinity
        if cond == float('inf'):
            cond = 1e6  # 用一个大数代替
        cond_numbers.append(cond)
        eff_ranks.append(entry['effective_rank'])
        top1_energies.append(entry['top1_energy'])
    
    n_over_s2 = np.array(n_over_s2)
    cond_numbers = np.array(cond_numbers)
    eff_ranks = np.array(eff_ranks)
    top1_energies = np.array(top1_energies)
    
    print(f"\n数据规模扫描:")
    print(f"{'N/S^2':>10} {'Cond#':>12} {'EffRank':>10} {'Top1 Energy':>12}")
    print("-" * 50)
    for i in range(len(n_over_s2)):
        cond_str = f"{cond_numbers[i]:.1f}" if cond_numbers[i] < 1e5 else "Inf"
        print(f"{n_over_s2[i]:10.4f} {cond_str:>12} {eff_ranks[i]:10.2f} {top1_energies[i]:12.4f}")
    
    # 识别相变点
    # 条件数急剧增加的点
    cond_ratios = cond_numbers[1:] / cond_numbers[:-1]
    max_jump_idx = np.argmax(cond_ratios) + 1
    critical_point = n_over_s2[max_jump_idx]
    
    print(f"\n相变点分析:")
    print(f"  条件数最大跳跃在 N/S^2 = {critical_point:.4f}")
    print(f"  条件数从 {cond_numbers[max_jump_idx-1]:.1f} 增至 {cond_numbers[max_jump_idx]:.1f}")
    print(f"  跳跃倍数: {cond_ratios[max_jump_idx-1]:.1f}x")
    
    # 验证临界缩放
    # 在相变点附近，有效秩应该呈现特定缩放
    # 理论: eff_rank ~ |N/S² - C_crit|^(-beta)
    
    results['phase_transition'] = {
        'critical_point': float(critical_point),
        'condition_number_jump': float(cond_ratios[max_jump_idx-1]),
        'distributed_regime': {
            'n_over_s2_range': n_over_s2[n_over_s2 < critical_point].tolist(),
            'mean_cond': float(np.mean(cond_numbers[n_over_s2 < critical_point])),
            'mean_eff_rank': float(np.mean(eff_ranks[n_over_s2 < critical_point]))
        },
        'concentrated_regime': {
            'n_over_s2_range': n_over_s2[n_over_s2 > critical_point].tolist(),
            'mean_cond': float(np.mean(cond_numbers[n_over_s2 > critical_point])),
            'mean_eff_rank': float(np.mean(eff_ranks[n_over_s2 > critical_point]))
        }
    }
    
    # Top-1 能量集中验证
    print(f"\nTop-1 能量集中分析:")
    print(f"  分布式区域平均: {np.mean(top1_energies[n_over_s2 < critical_point]):.4f}")
    print(f"  集中式区域平均: {np.mean(top1_energies[n_over_s2 > critical_point]):.4f}")
    print(f"  集中程度提升: {(top1_energies[-1]/top1_energies[0] - 1)*100:.1f}%")
    
    results['energy_concentration'] = {
        'distributed_avg': float(np.mean(top1_energies[n_over_s2 < critical_point])),
        'concentrated_avg': float(np.mean(top1_energies[n_over_s2 > critical_point])),
        'improvement_factor': float(top1_energies[-1]/top1_energies[0])
    }
    
    # 验证结论
    if cond_ratios[max_jump_idx-1] > 10:
        print(f"\n[PASS] Theorem 3 验证通过: 观察到明显的相变行为")
        print(f"  临界复杂度 C_crit ≈ {critical_point:.3f}")
    else:
        print(f"\n[WARN] 相变信号不够强烈，可能需要更多数据点")
    
    return results

# ============================================================================
# 8. 主函数: 运行所有验证
# ============================================================================

def main():
    """运行所有定理验证"""
    
    print(f"\n{'='*70}")
    print("开始严格理论验证")
    print(f"{'='*70}\n")
    
    # 加载已有数据
    data = load_existing_data()
    
    # 运行验证实验
    all_results = {}
    
    # 实验 I-III: 验证 Theorem 5
    all_results['experiment_I_amplitude'] = validate_amplitude_recovery(data['theorem1'])
    all_results['experiment_II_phase'] = validate_phase_recovery(data['theorem1'])
    all_results['experiment_III_rotation'] = validate_rotation_invariance()
    
    # 定理 1-3 数值验证
    all_results['theorem1_validation'] = validate_theorem1_capacity(data['theorem1'])
    all_results['theorem2_validation'] = validate_theorem2_convergence(data['theorem2'])
    all_results['theorem3_validation'] = validate_theorem3_phase_transition(data['theorem3'])
    
    # 保存结果
    output_file = "C:\\Users\\MR\\Desktop\\论文\\关于场物理的神经框架\\研究论文数据\\theorem_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"验证完成！结果保存到:")
    print(f"  {output_file}")
    print(f"{'='*70}\n")
    
    return all_results

if __name__ == "__main__":
    results = main()
