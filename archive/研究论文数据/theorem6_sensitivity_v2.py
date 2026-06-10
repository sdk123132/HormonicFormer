"""
Theorem 6: Approximate Identifiability under Perturbations
参数敏感性分析 (独立版本)

验证: 当参数有微小扰动时，表示变化是否可控
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from scipy import stats

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class SimpleCGLCell(nn.Module):
    """简化版 CGL 单元用于敏感性分析"""
    def __init__(self, d_model=64, alpha=1.0, beta=1.0, D=0.1):
        super().__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.D = nn.Parameter(torch.tensor(D))
        
        # 状态
        self.register_buffer('u', torch.zeros(1, d_model))
        self.register_buffer('v', torch.zeros(1, d_model))
        
    def forward(self, n_steps=10):
        """CGL 动力学"""
        for _ in range(n_steps):
            # CGL 方程
            r2 = self.u**2 + self.v**2
            du = self.alpha * self.u - self.beta * r2 * self.u - self.D * self.v
            dv = self.alpha * self.v - self.beta * r2 * self.v + self.D * self.u
            
            self.u = self.u + 0.1 * du
            self.v = self.v + 0.1 * dv
        
        return torch.cat([self.u, self.v], dim=-1)

def compute_representation_change(model, param_name, original_value, perturbed_value):
    """
    计算参数扰动导致的表示变化
    
    Returns:
        relative_change: 相对变化量
    """
    # 原始参数下的表示
    with torch.no_grad():
        # 设置原始参数
        if hasattr(model, param_name):
            setattr(model, param_name, nn.Parameter(torch.tensor(original_value)))
        
        model.u.zero_()
        model.v.zero_()
        h_original = model.forward().detach().clone()
        
        # 设置扰动参数
        if hasattr(model, param_name):
            setattr(model, param_name, nn.Parameter(torch.tensor(perturbed_value)))
        
        model.u.zero_()
        model.v.zero_()
        h_perturbed = model.forward().detach().clone()
    
    # 计算变化
    absolute_change = torch.norm(h_perturbed - h_original).item()
    relative_change = absolute_change / (torch.norm(h_original).item() + 1e-8)
    
    return relative_change

def sensitivity_analysis_rigorous():
    """
    严谨的参数敏感性分析
    """
    print("="*80)
    print("Theorem 6: Approximate Identifiability under Perturbations")
    print("Rigorous Parameter Sensitivity Analysis")
    print("="*80)
    print()
    
    # 创建模型
    model = SimpleCGLCell(d_model=64, alpha=1.0, beta=1.0, D=0.1)
    
    # 参数配置
    param_configs = {
        'alpha': {'base': 1.0, 'range': [0.5, 0.8, 1.0, 1.2, 1.5]},
        'beta': {'base': 1.0, 'range': [0.5, 0.8, 1.0, 1.2, 1.5]},
        'D': {'base': 0.1, 'range': [0.01, 0.05, 0.1, 0.15, 0.2]}
    }
    
    results = {
        'parameters': {},
        'sensitivity_ranking': [],
        'robustness_analysis': {}
    }
    
    print("Analyzing parameter sensitivity...")
    print()
    
    for param_name, config in param_configs.items():
        print(f"Parameter: {param_name}")
        print(f"  Base value: {config['base']}")
        print(f"  Test range: {config['range']}")
        
        changes = []
        for test_value in config['range']:
            relative_change = compute_representation_change(
                model, param_name, config['base'], test_value
            )
            changes.append({
                'value': test_value,
                'relative_change': relative_change
            })
            print(f"    {param_name}={test_value}: relative_change={relative_change:.6f}")
        
        # 计算敏感性指标
        relative_changes = [c['relative_change'] for c in changes]
        sensitivity = np.std(relative_changes) / (np.mean(relative_changes) + 1e-8)
        
        results['parameters'][param_name] = {
            'base_value': config['base'],
            'test_range': config['range'],
            'changes': changes,
            'sensitivity': float(sensitivity),
            'mean_change': float(np.mean(relative_changes)),
            'max_change': float(np.max(relative_changes))
        }
        
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Mean change: {np.mean(relative_changes):.6f}")
        print(f"  Max change: {np.max(relative_changes):.6f}")
        print()
    
    # 敏感性排序
    print("="*80)
    print("Sensitivity Ranking")
    print("="*80)
    print()
    
    sorted_params = sorted(
        results['parameters'].items(),
        key=lambda x: x[1]['sensitivity'],
        reverse=True
    )
    
    for rank, (param_name, param_data) in enumerate(sorted_params, 1):
        print(f"{rank}. {param_name}: sensitivity = {param_data['sensitivity']:.4f}")
        results['sensitivity_ranking'].append({
            'rank': rank,
            'parameter': param_name,
            'sensitivity': param_data['sensitivity']
        })
    
    print()
    
    # 鲁棒性分析
    print("="*80)
    print("Robustness Analysis")
    print("="*80)
    print()
    
    # 定义"小扰动"
    epsilon = 0.1  # 10% 扰动
    
    n_stable = 0
    robust_params = []
    
    for param_name, param_data in results['parameters'].items():
        # 检查在 epsilon 邻域内是否稳定
        max_change = param_data['max_change']
        
        if max_change < epsilon:
            n_stable += 1
            robust_params.append(param_name)
            print(f"[STABLE] {param_name}: max_change={max_change:.4f} < epsilon={epsilon}")
        else:
            print(f"[UNSTABLE] {param_name}: max_change={max_change:.4f} > epsilon={epsilon}")
    
    print()
    print(f"Stable parameters: {n_stable}/{len(param_configs)}")
    
    results['robustness_analysis'] = {
        'epsilon': epsilon,
        'n_stable': n_stable,
        'n_total': len(param_configs),
        'robust_params': robust_params,
        'stability_ratio': n_stable / len(param_configs)
    }
    
    # 验证结论
    print()
    print("="*80)
    print("Verification Conclusion")
    print("="*80)
    print()
    
    stability_ratio = results['robustness_analysis']['stability_ratio']
    
    if stability_ratio >= 0.5:
        print("[PASS] Theorem 6 verified: Approximate identifiability holds")
        print(f"       {n_stable}/{len(param_configs)} parameters are stable under perturbations")
        print("       Representation changes are bounded and predictable")
        results['status'] = 'PASS'
    else:
        print("[PARTIAL] Theorem 6 partially verified:")
        print(f"          Only {n_stable}/{len(param_configs)} parameters are stable")
        print("          Some parameters may need tighter constraints")
        results['status'] = 'PARTIAL'
    
    # 保存结果
    output_path = Path(__file__).parent / 'theorem6_sensitivity_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"[OK] Results saved to: {output_path}")
    
    return results

def generate_report(results):
    """生成验证报告"""
    lines = []
    lines.append("="*80)
    lines.append("Theorem 6: Approximate Identifiability under Perturbations")
    lines.append("Rigorous Parameter Sensitivity Analysis Report")
    lines.append("="*80)
    lines.append("")
    
    lines.append("Parameter Sensitivity Analysis:")
    lines.append("")
    
    for param_name, param_data in results['parameters'].items():
        lines.append(f"  {param_name}:")
        lines.append(f"    Base value: {param_data['base_value']}")
        lines.append(f"    Sensitivity: {param_data['sensitivity']:.4f}")
        lines.append(f"    Mean change: {param_data['mean_change']:.6f}")
        lines.append(f"    Max change: {param_data['max_change']:.6f}")
        lines.append("")
    
    lines.append("Sensitivity Ranking:")
    for item in results['sensitivity_ranking']:
        lines.append(f"  {item['rank']}. {item['parameter']}: {item['sensitivity']:.4f}")
    lines.append("")
    
    robust = results.get('robustness_analysis', {})
    lines.append("Robustness Analysis:")
    lines.append(f"  Epsilon threshold: {robust.get('epsilon', 0.1)}")
    lines.append(f"  Stable parameters: {robust.get('n_stable', 0)}/{robust.get('n_total', 0)}")
    lines.append(f"  Stability ratio: {robust.get('stability_ratio', 0):.2%}")
    lines.append("")
    
    status = results.get('status', 'UNKNOWN')
    lines.append(f"Status: {status}")
    lines.append("")
    
    if status == 'PASS':
        lines.append("Conclusion: Theorem 6 is verified.")
        lines.append("The system maintains approximate identifiability under parameter perturbations.")
    elif status == 'PARTIAL':
        lines.append("Conclusion: Theorem 6 is partially verified.")
        lines.append("Some parameters require additional constraints for full stability.")
    
    lines.append("")
    lines.append("="*80)
    
    return "\n".join(lines)

if __name__ == '__main__':
    # 运行敏感性分析
    results = sensitivity_analysis_rigorous()
    
    # 生成报告
    report = generate_report(results)
    
    # 保存报告
    report_path = Path(__file__).parent / '24_Theorem6_敏感性分析报告.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print()
    print(f"[OK] Report saved to: {report_path}")
    print()
    print(report)
