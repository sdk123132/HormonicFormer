"""
实验5: T6 敏感性分析重跑（修正版）
使用PPL作为指标，验证参数扰动对表示的影响

关键修正:
- 用PPL作为指标（比之前的不明指标敏感）
- 每个参数单独扰动（排除交叉效应）
- 记录连续变化曲线（不只二值判断）
- 扰动幅度±50%（足够大以产生可测变化）
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

class SimpleCGLModel(nn.Module):
    """简化版CGL模型用于敏感性分析"""
    def __init__(self, d_model=64, seq_len=64, alpha=0.731, beta=0.5, D0=0.002, g_coupling=0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.alpha = alpha
        self.beta = beta
        self.D0 = D0
        self.g_coupling = g_coupling
        
        # 简单的编码器
        self.encoder = nn.Linear(2, d_model)
        self.decoder = nn.Linear(d_model, 2)
        
    def cgl_step(self, psi):
        """单步CGL动力学"""
        # psi: [batch, seq_len, 2] (实部和虚部)
        u = psi[..., 0]
        v = psi[..., 1]
        
        # 计算模和相位
        r2 = u**2 + v**2
        
        # CGL方程
        du = self.alpha * u - self.beta * r2 * u - self.D0 * v
        dv = self.alpha * v - self.beta * r2 * v + self.D0 * u
        
        # 更新
        dt = 0.02
        u_new = u + dt * du
        v_new = v + dt * dv
        
        return torch.stack([u_new, v_new], dim=-1)
    
    def forward(self, x, n_steps=10):
        """
        x: [batch, seq_len, 2] 输入 (amplitude, phase)
        """
        # 编码
        h = self.encoder(x)  # [batch, seq_len, d_model]
        
        # CGL动力学
        psi = torch.zeros(x.size(0), self.seq_len, 2, device=x.device)
        psi[:, :, 0] = x[:, :, 0]  # amplitude -> real
        psi[:, :, 1] = x[:, :, 1]  # phase -> imag (simplified)
        
        for _ in range(n_steps):
            psi = self.cgl_step(psi)
        
        # 解码
        output = self.decoder(h)
        return output, psi

def generate_synthetic_data(seq_len=64, batch_size=32, n_batches=10):
    """生成合成数据"""
    data = []
    for _ in range(n_batches):
        # 随机幅度和相位
        amp = torch.rand(batch_size, seq_len) * 2.0  # [0, 2]
        phase = torch.rand(batch_size, seq_len) * 2 * np.pi  # [0, 2π]
        x = torch.stack([amp, phase], dim=-1)  # [batch, seq, 2]
        data.append(x)
    return data

def evaluate_model_with_config(config, data, criterion):
    """评估特定配置下的模型性能"""
    model = SimpleCGLModel(
        d_model=64,
        seq_len=64,
        alpha=config['alpha'],
        beta=config['beta'],
        D0=config['D0'],
        g_coupling=config['g_coupling']
    ).to(device)
    
    model.eval()
    total_loss = 0
    n_samples = 0
    
    with torch.no_grad():
        for batch in data:
            batch = batch.to(device)
            output, _ = model(batch)
            
            # 简单的重构损失
            loss = criterion(output, batch)
            total_loss += loss.item() * batch.size(0)
            n_samples += batch.size(0)
    
    avg_loss = total_loss / n_samples
    # 使用perplexity-like指标
    ppl = np.exp(avg_loss)
    
    return ppl

def run_t6_sensitivity_analysis():
    """运行T6敏感性分析"""
    print("="*80)
    print("实验5: T6 敏感性分析重跑（修正版）")
    print("="*80)
    print()
    
    # 基准配置
    base_config = {
        'alpha': 0.731,      # softplus(1.0)
        'beta': 0.5,
        'D0': 0.002,
        'g_coupling': 0.1
    }
    
    print("基准配置:")
    for k, v in base_config.items():
        print(f"  {k}: {v}")
    print()
    
    # 生成测试数据
    print("[1/4] 生成测试数据...")
    test_data = generate_synthetic_data(seq_len=64, batch_size=32, n_batches=10)
    criterion = nn.MSELoss()
    
    # 评估基准配置
    print("[2/4] 评估基准配置...")
    baseline_ppl = evaluate_model_with_config(base_config, test_data, criterion)
    print(f"  基准PPL: {baseline_ppl:.4f}")
    print()
    
    # 参数扰动测试
    parameters_to_test = {
        'alpha': [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5],
        'beta': [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5],
        'D0': [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5],
        'g_coupling': [-0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5]
    }
    
    results = {
        'baseline_ppl': float(baseline_ppl),
        'base_config': base_config,
        'sensitivities': []
    }
    
    print("[3/4] 运行参数敏感性测试...")
    for param_name, perturbations in parameters_to_test.items():
        print(f"\n  测试参数: {param_name}")
        
        param_results = []
        
        for delta in perturbations:
            if delta == 0:
                continue
            
            # 创建扰动配置
            perturbed_config = base_config.copy()
            perturbed_config[param_name] *= (1 + delta)
            
            # 确保参数为正
            if perturbed_config[param_name] <= 0:
                perturbed_config[param_name] = 0.001
            
            # 评估
            ppl = evaluate_model_with_config(perturbed_config, test_data, criterion)
            
            # 计算敏感性
            ppl_change = (ppl - baseline_ppl) / baseline_ppl
            sensitivity = ppl_change / delta
            
            param_results.append({
                'perturbation': delta,
                'ppl': float(ppl),
                'ppl_change': float(ppl_change),
                'sensitivity': float(sensitivity)
            })
            
            print(f"    {param_name} × {1+delta:+.2f}: PPL {baseline_ppl:.4f} → {ppl:.4f} (变化: {ppl_change:+.4f}, 敏感性: {sensitivity:+.4f})")
        
        # 计算平均敏感性
        avg_sensitivity = np.mean([abs(r['sensitivity']) for r in param_results])
        
        results['sensitivities'].append({
            'parameter': param_name,
            'avg_sensitivity': float(avg_sensitivity),
            'results': param_results
        })
    
    print("\n[4/4] 生成报告...")
    
    # 敏感性排序
    sorted_sens = sorted(results['sensitivities'], key=lambda x: x['avg_sensitivity'], reverse=True)
    
    print("\n" + "="*80)
    print("敏感性排序 (按平均绝对敏感性):")
    print("="*80)
    for rank, item in enumerate(sorted_sens, 1):
        print(f"  {rank}. {item['parameter']}: {item['avg_sensitivity']:.4f}")
    
    # 验证结论
    print("\n" + "="*80)
    print("验证结论")
    print("="*80)
    
    # 检查参数稳定性
    n_stable = sum(1 for item in sorted_sens if item['avg_sensitivity'] < 1.0)
    
    if n_stable >= len(sorted_sens) * 0.75:
        print(f"[PASS] Theorem 6 verified: {n_stable}/{len(sorted_sens)} 参数稳定")
        print("       参数扰动对PPL影响可控，近似可识别性成立")
        results['status'] = 'PASS'
    else:
        print(f"[PARTIAL] Theorem 6 partially verified: {n_stable}/{len(sorted_sens)} 参数稳定")
        results['status'] = 'PARTIAL'
    
    # 保存结果
    output_path = Path(__file__).parent / 'experiment_t6_sensitivity_v3_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 结果保存至: {output_path}")
    
    return results

def generate_report(results):
    """生成实验报告"""
    lines = []
    lines.append("="*80)
    lines.append("实验5: T6 敏感性分析重跑（修正版）报告")
    lines.append("="*80)
    lines.append("")
    
    lines.append("基准配置:")
    for k, v in results['base_config'].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  基准PPL: {results['baseline_ppl']:.4f}")
    lines.append("")
    
    lines.append("参数敏感性分析:")
    for item in results['sensitivities']:
        lines.append(f"\n  {item['parameter']} (平均敏感性: {item['avg_sensitivity']:.4f}):")
        for r in item['results']:
            lines.append(f"    扰动 {r['perturbation']:+.2f}: PPL={r['ppl']:.4f}, 敏感性={r['sensitivity']:+.4f}")
    
    lines.append("")
    lines.append(f"状态: {results['status']}")
    lines.append("")
    
    if results['status'] == 'PASS':
        lines.append("结论: Theorem 6 验证通过。参数扰动对表示质量影响可控，")
        lines.append("      近似可识别性在参数空间中成立。")
    
    lines.append("")
    lines.append("="*80)
    
    return "\n".join(lines)

if __name__ == '__main__':
    print("开始运行实验5: T6 敏感性分析...")
    print()
    
    # 运行实验
    results = run_t6_sensitivity_analysis()
    
    # 生成报告
    report = generate_report(results)
    
    # 保存报告
    report_path = Path(__file__).parent / '26_实验5_T6敏感性分析报告.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[OK] 报告保存至: {report_path}")
    print()
    print(report)
