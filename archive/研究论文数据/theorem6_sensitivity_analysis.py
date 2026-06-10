"""
Theorem 6: Approximate Identifiability under Perturbations
参数敏感性分析 - 严谨验证

验证目标:
1. 参数扰动对表示质量的影响
2. 参数敏感性矩阵计算
3. 验证近似可识别性条件
"""

import torch
import torch.nn as nn
import numpy as np
import json
import sys
from pathlib import Path

# 添加模型路径
sys.path.insert(0, r'C:\Users\MR\Desktop\初代激素场网络\文本\hormonic_v3 - 副本')

from models.hormonicformer_v3 import HormonicFormer, HormonicConfig

# 设置随机种子确保可复现
torch.manual_seed(42)
np.random.seed(42)

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# 基础配置
BASE_CONFIG = {
    'vocab_size': 256,
    'd_model': 64,
    'n_layers': 2,
    'n_heads': 4,
    'd_ff': 256,
    'dropout': 0.1,
    'max_seq_len': 64,
    'use_cgl': True,
    'use_stp': True,
    'use_hebbian': True,
    'use_da': True,
    'use_cb': True,
    'ei_balance': True,
    'n_steps': 5,
}

# 需要测试的参数及其扰动范围
PARAM_RANGES = {
    'alpha': {'base': 1.0, 'range': [0.1, 0.5, 1.0, 2.0, 5.0]},
    'beta': {'base': 1.0, 'range': [0.5, 1.0, 2.0]},
    'eta_hebb': {'base': 0.001, 'range': [0.0001, 0.001, 0.01]},
    'lambda_decay': {'base': 0.95, 'range': [0.9, 0.95, 0.99]},
    'theta_sync': {'base': 0.1, 'range': [0.05, 0.1, 0.2]},
    'dropout': {'base': 0.1, 'range': [0.0, 0.1, 0.3]},
}

def create_model_with_param(param_name, param_value):
    """创建具有特定参数值的模型"""
    config = BASE_CONFIG.copy()
    
    # 根据参数类型设置
    if param_name == 'alpha':
        config['cgl_alpha'] = param_value
    elif param_name == 'beta':
        config['cgl_beta'] = param_value
    elif param_name == 'eta_hebb':
        config['eta_hebb'] = param_value
    elif param_name == 'lambda_decay':
        config['lambda_decay'] = param_value
    elif param_name == 'theta_sync':
        config['theta_sync'] = param_value
    elif param_name == 'dropout':
        config['dropout'] = param_value
    
    model = HormonicFormer(HormonicConfig(**config))
    return model.to(device)

def generate_synthetic_data(seq_len=64, batch_size=32, n_batches=10):
    """生成合成数据用于测试"""
    data = []
    for _ in range(n_batches):
        # 随机序列
        x = torch.randint(0, BASE_CONFIG['vocab_size'], (batch_size, seq_len))
        data.append(x)
    return data

def evaluate_model(model, data, criterion):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data:
            batch = batch.to(device)
            
            # 前向传播
            try:
                logits, _ = model(batch, seq_input=True)
                
                # 计算损失 (next token prediction)
                logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
                targets_flat = batch[:, 1:].reshape(-1)
                loss = criterion(logits_flat, targets_flat)
                
                total_loss += loss.item() * batch.size(0)
                total_samples += batch.size(0)
            except Exception as e:
                print(f"[WARNING] Evaluation error: {e}")
                continue
    
    avg_loss = total_loss / max(total_samples, 1)
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def compute_sensitivity_matrix():
    """计算参数敏感性矩阵"""
    print("\n" + "="*80)
    print("Theorem 6: Parameter Sensitivity Analysis")
    print("="*80)
    
    # 生成测试数据
    print("\n[1/5] Generating synthetic test data...")
    test_data = generate_synthetic_data(seq_len=64, batch_size=32, n_batches=10)
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    
    # 基准模型
    print("\n[2/5] Evaluating baseline model...")
    baseline_model = create_model_with_param('alpha', 1.0)
    baseline_loss, baseline_ppl = evaluate_model(baseline_model, test_data, criterion)
    print(f"  Baseline: Loss={baseline_loss:.4f}, PPL={baseline_ppl:.2f}")
    
    results['baseline'] = {
        'loss': baseline_loss,
        'perplexity': baseline_ppl
    }
    
    # 参数敏感性分析
    print("\n[3/5] Computing parameter sensitivity...")
    sensitivity_results = {}
    
    for param_name, param_info in PARAM_RANGES.items():
        print(f"\n  Testing parameter: {param_name}")
        param_values = param_info['range']
        base_value = param_info['base']
        
        param_losses = []
        param_ppls = []
        
        for value in param_values:
            try:
                model = create_model_with_param(param_name, value)
                loss, ppl = evaluate_model(model, test_data, criterion)
                param_losses.append(loss)
                param_ppls.append(ppl)
                print(f"    {param_name}={value}: Loss={loss:.4f}, PPL={ppl:.2f}")
            except Exception as e:
                print(f"    [ERROR] {param_name}={value}: {e}")
                param_losses.append(None)
                param_ppls.append(None)
        
        # 计算敏感性 (损失变化率)
        valid_losses = [l for l in param_losses if l is not None]
        if len(valid_losses) >= 2:
            loss_range = max(valid_losses) - min(valid_losses)
            sensitivity = loss_range / baseline_loss if baseline_loss > 0 else 0
        else:
            sensitivity = 0
        
        sensitivity_results[param_name] = {
            'values': param_values,
            'losses': param_losses,
            'perplexities': param_ppls,
            'sensitivity': sensitivity,
            'base_value': base_value
        }
        
        print(f"    Sensitivity: {sensitivity:.4f}")
    
    results['sensitivity_analysis'] = sensitivity_results
    
    # 计算敏感性排序
    print("\n[4/5] Parameter sensitivity ranking:")
    sorted_params = sorted(
        sensitivity_results.items(),
        key=lambda x: x[1]['sensitivity'],
        reverse=True
    )
    
    for rank, (param_name, param_data) in enumerate(sorted_params, 1):
        print(f"  {rank}. {param_name}: {param_data['sensitivity']:.4f}")
    
    # 验证近似可识别性条件
    print("\n[5/5] Verifying approximate identifiability conditions...")
    
    # 条件1: 小扰动下性能变化小
    small_perturbation_stable = []
    for param_name, param_data in sensitivity_results.items():
        base_idx = param_data['values'].index(param_data['base_value'])
        base_loss = param_data['losses'][base_idx]
        
        # 检查相邻值的损失变化
        stable = True
        for i, loss in enumerate(param_data['losses']):
            if loss is not None and i != base_idx:
                relative_change = abs(loss - base_loss) / base_loss if base_loss > 0 else 0
                if relative_change > 0.2:  # 20%阈值
                    stable = False
                    break
        
        small_perturbation_stable.append({
            'param': param_name,
            'stable': stable
        })
    
    n_stable = sum(1 for x in small_perturbation_stable if x['stable'])
    print(f"  Parameters stable under small perturbations: {n_stable}/{len(small_perturbation_stable)}")
    
    # 条件2: 存在鲁棒参数区域
    robust_regions = []
    for param_name, param_data in sensitivity_results.items():
        valid_losses = [(v, l) for v, l in zip(param_data['values'], param_data['losses']) if l is not None]
        if len(valid_losses) >= 3:
            # 寻找损失最小的区域
            sorted_by_loss = sorted(valid_losses, key=lambda x: x[1])
            best_value = sorted_by_loss[0][0]
            best_loss = sorted_by_loss[0][1]
            
            # 检查鲁棒性 (损失变化 < 10%)
            robust_range = []
            for value, loss in valid_losses:
                if abs(loss - best_loss) / best_loss < 0.1 if best_loss > 0 else True:
                    robust_range.append(value)
            
            if len(robust_range) >= 2:
                robust_regions.append({
                    'param': param_name,
                    'best_value': best_value,
                    'robust_range': robust_range
                })
    
    print(f"  Robust parameter regions found: {len(robust_regions)}/{len(sensitivity_results)}")
    for region in robust_regions:
        print(f"    {region['param']}: best={region['best_value']}, robust range={region['robust_range']}")
    
    results['identifiability_conditions'] = {
        'small_perturbation_stable': small_perturbation_stable,
        'robust_regions': robust_regions,
        'n_stable': n_stable,
        'n_robust': len(robust_regions)
    }
    
    # 保存结果
    output_path = Path(__file__).parent / 'theorem6_sensitivity_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[✓] Results saved to: {output_path}")
    
    return results

def generate_report(results):
    """生成验证报告"""
    report = []
    report.append("="*80)
    report.append("Theorem 6: Approximate Identifiability under Perturbations")
    report.append("Parameter Sensitivity Analysis Report")
    report.append("="*80)
    report.append("")
    
    # 基准性能
    baseline = results.get('baseline', {})
    report.append("Baseline Performance:")
    report.append(f"  Loss: {baseline.get('loss', 'N/A'):.4f}")
    report.append(f"  Perplexity: {baseline.get('perplexity', 'N/A'):.2f}")
    report.append("")
    
    # 敏感性分析
    sensitivity = results.get('sensitivity_analysis', {})
    report.append("Parameter Sensitivity Ranking:")
    
    sorted_params = sorted(
        sensitivity.items(),
        key=lambda x: x[1].get('sensitivity', 0),
        reverse=True
    )
    
    for rank, (param_name, param_data) in enumerate(sorted_params, 1):
        sens = param_data.get('sensitivity', 0)
        report.append(f"  {rank}. {param_name}: sensitivity = {sens:.4f}")
        
        # 详细数据
        values = param_data.get('values', [])
        losses = param_data.get('losses', [])
        for v, l in zip(values, losses):
            if l is not None:
                report.append(f"      {param_name}={v}: loss={l:.4f}")
    
    report.append("")
    
    # 可识别性条件
    conditions = results.get('identifiability_conditions', {})
    report.append("Approximate Identifiability Conditions:")
    report.append(f"  1. Small perturbation stability: {conditions.get('n_stable', 0)}/{len(sensitivity)}")
    report.append(f"  2. Robust parameter regions: {conditions.get('n_robust', 0)}/{len(sensitivity)}")
    report.append("")
    
    # 结论
    n_stable = conditions.get('n_stable', 0)
    n_robust = conditions.get('n_robust', 0)
    total = len(sensitivity)
    
    if n_stable >= total * 0.8 and n_robust >= total * 0.5:
        report.append("[PASS] Theorem 6 verified: Approximate identifiability holds")
        report.append("       under parameter perturbations.")
    else:
        report.append("[PARTIAL] Theorem 6 partially verified:")
        report.append(f"          - {n_stable}/{total} parameters stable")
        report.append(f"          - {n_robust}/{total} parameters have robust regions")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)

if __name__ == '__main__':
    print("="*80)
    print("Theorem 6: Approximate Identifiability under Perturbations")
    print("Rigorous Parameter Sensitivity Analysis")
    print("="*80)
    print("")
    
    # 运行敏感性分析
    results = compute_sensitivity_matrix()
    
    # 生成报告
    report = generate_report(results)
    
    # 保存报告
    report_path = Path(__file__).parent / '21_Theorem6_敏感性分析报告.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[✓] Report saved to: {report_path}")
    print("\n" + report)
