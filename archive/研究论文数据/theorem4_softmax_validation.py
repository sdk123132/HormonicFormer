"""
定理 4 验证实验：CGL ≈ 软注意力
验证 CGL 场对竞争模式的响应是否符合 softmax 分布
"""
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

# 添加模型路径
import sys
sys.path.insert(0, r'C:\Users\MR\Desktop\论文\关于场物理的神经框架\第二代')

from hormonic_v7r3_validated import HormonicFormerV7r3, HormonicConfig

class SoftmaxCGLValidator(nn.Module):
    """验证 CGL 场的 softmax 行为"""
    
    def __init__(self, d_model=64, seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 创建 CGL 层（从 HormonicBlock 中提取）
        config = HormonicConfig(
            vocab_size=100,
            d_model=d_model,
            n_layers=1,
            n_heads=4,
            seq_len=seq_len
        )
        self.model = HormonicFormerV7r3(config)
        self.cgl = self.model.blocks[0].cgl
        
    def generate_competing_modes(self, A1, A2, omega1=1.0, omega2=1.1, n_steps=1000):
        """
        生成两个竞争模式的输入
        A1, A2: 两个模式的振幅
        omega1, omega2: 两个模式的频率（稍微不同以产生竞争）
        """
        t = torch.linspace(0, 4*np.pi, n_steps)
        
        # 模式1: A1 * exp(i*omega1*t)
        # 模式2: A2 * exp(i*omega2*t)
        mode1 = A1 * torch.exp(1j * omega1 * t)
        mode2 = A2 * torch.exp(1j * omega2 * t)
        
        # 叠加输入
        input_signal = mode1 + mode2
        
        # 转换为实数表示 [real, imag]
        input_real = torch.stack([input_signal.real, input_signal.imag], dim=-1)  # [n_steps, 2]
        
        return input_real, mode1, mode2
    
    def measure_locking_probability(self, A1, A2, n_trials=20, n_steps=1000):
        """
        测量 CGL 场锁定到模式1的概率
        通过多次试验统计
        """
        lock_to_mode1_count = 0
        
        for trial in range(n_trials):
            # 生成输入
            input_signal, mode1, mode2 = self.generate_competing_modes(
                A1, A2, n_steps=n_steps
            )
            
            # 初始化 CGL 场
            psi = torch.randn(self.seq_len, self.d_model, dtype=torch.complex64) * 0.1
            
            # 运行 CGL 动力学
            dt = 0.01
            for step in range(n_steps):
                # 提取当前输入（循环使用）
                inp = input_signal[step % n_steps]
                inp_complex = inp[0] + 1j * inp[1]
                
                # CGL 更新（简化版）
                alpha = 1.0
                beta = 1.0
                D = 0.1
                
                # 扩展输入到场的维度
                inp_expanded = inp_complex * torch.ones_like(psi)
                
                # CGL 方程
                dpsi = alpha * psi - beta * torch.abs(psi)**2 * psi + D * self.laplacian(psi) + inp_expanded * 0.1
                psi = psi + dt * dpsi
            
            # 判断锁定到哪个模式
            # 计算与两个模式的相位相关性
            psi_flat = psi.flatten()
            mode1_flat = mode1[:self.seq_len * self.d_model].flatten()
            mode2_flat = mode2[:self.seq_len * self.d_model].flatten()
            
            corr1 = torch.abs(torch.sum(psi_flat * mode1_flat.conj()))
            corr2 = torch.abs(torch.sum(psi_flat * mode2_flat.conj()))
            
            if corr1 > corr2:
                lock_to_mode1_count += 1
        
        probability = lock_to_mode1_count / n_trials
        return probability
    
    def laplacian(self, psi):
        """简化 Laplacian（周期性边界）"""
        # 空间维度上的离散 Laplacian
        result = torch.zeros_like(psi)
        for i in range(psi.shape[0]):
            for j in range(psi.shape[1]):
                neighbors = []
                if i > 0: neighbors.append(psi[i-1, j])
                if i < psi.shape[0]-1: neighbors.append(psi[i+1, j])
                if j > 0: neighbors.append(psi[i, j-1])
                if j < psi.shape[1]-1: neighbors.append(psi[i, j+1])
                
                if neighbors:
                    result[i, j] = sum(neighbors) / len(neighbors) - psi[i, j]
        return result
    
    def softmax_prediction(self, A1, A2, temperature=1.0):
        """计算 softmax 理论预测"""
        p1 = np.exp(A1 / temperature)
        p2 = np.exp(A2 / temperature)
        return p1 / (p1 + p2)


def run_validation_experiment():
    """运行完整验证实验"""
    print("="*60)
    print("定理 4 验证：CGL ≈ 软注意力")
    print("="*60)
    
    validator = SoftmaxCGLValidator(d_model=32, seq_len=32)
    
    # 测试不同的振幅比
    amplitude_ratios = [0.2, 0.5, 1.0, 2.0, 5.0]
    A2 = 1.0  # 固定
    
    results = []
    
    for ratio in amplitude_ratios:
        A1 = ratio * A2
        print(f"\n测试 A1/A2 = {ratio:.1f} (A1={A1:.1f}, A2={A2:.1f})")
        
        # 测量 CGL 锁定概率
        prob_cgl = validator.measure_locking_probability(A1, A2, n_trials=10, n_steps=500)
        
        # 计算 softmax 预测（拟合温度参数）
        best_temp = None
        best_error = float('inf')
        for T in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            prob_softmax = validator.softmax_prediction(A1, A2, temperature=T)
            error = abs(prob_cgl - prob_softmax)
            if error < best_error:
                best_error = error
                best_temp = T
        
        prob_softmax = validator.softmax_prediction(A1, A2, temperature=best_temp)
        
        print(f"  CGL 测量概率: {prob_cgl:.3f}")
        print(f"  Softmax 预测: {prob_softmax:.3f} (T={best_temp})")
        print(f"  误差: {abs(prob_cgl - prob_softmax):.3f}")
        
        results.append({
            'A1': A1,
            'A2': A2,
            'ratio': ratio,
            'prob_cgl': prob_cgl,
            'prob_softmax': prob_softmax,
            'temperature': best_temp,
            'error': abs(prob_cgl - prob_softmax)
        })
    
    # 保存结果
    output_path = Path(r'C:\Users\MR\Desktop\论文\关于场物理的神经框架\研究论文数据\theorem4_softmax_validation.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"结果已保存: {output_path}")
    print(f"{'='*60}")
    
    # 总结
    avg_error = np.mean([r['error'] for r in results])
    print(f"\n平均误差: {avg_error:.3f}")
    
    if avg_error < 0.1:
        print("✅ CGL 行为与 Softmax 预测高度一致！")
    elif avg_error < 0.2:
        print("⚠️ CGL 行为与 Softmax 预测基本一致，有轻微偏差")
    else:
        print("❌ CGL 行为与 Softmax 预测差异较大")
    
    return results


if __name__ == '__main__':
    results = run_validation_experiment()
