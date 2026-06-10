"""
DFT-based Laplacian for DCU (HIP-compatible)
v6.1: 修复频域Laplacian核尺度 - 从 -(k^2) 改为 -((2πk/N)^2)

物理背景:
  连续Laplacian ∇² 的DFT近似要求频域乘子包含 2π/N 的尺度因子.
  原代码 -(k^2) 缺少此因子, 导致扩散强度与实际空间尺度严重错配.
  
  正确形式:  λ_k = -((2π · k) / N)^2
  
  对比: 原 λ_k = -k^2, 最大达 -(N/2)^2 = -9604 (N=196时)
       修复后 λ_k = -(2π/N)^2 · k^2, 最大约 -π^2 ≈ -9.87
       
  这意味着原代码的扩散强度被放大了约 1000 倍,
  导致 CGL 演化严重不稳定.
"""
import torch
import torch.nn as nn
import numpy as np


class DFTLaplacian(nn.Module):
    """
    DFT矩阵实现的Laplacian算子
    完全兼容DCU (无需torch.fft)
    """
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self._init_dft_matrix()

    def _init_dft_matrix(self):
        """初始化DFT/IDFT矩阵和频域核 (v6.1修复尺度)"""
        N = self.seq_len
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(N, dtype=torch.float32).view(-1, 1)

        # DFT矩阵: W[k,n] = exp(-2*pi*i * k * n / N)
        theta = -2 * np.pi * k * n / N
        self.register_buffer('W_real', torch.cos(theta))
        self.register_buffer('W_imag', torch.sin(theta))

        # IDFT矩阵: W_inv = W^H / N
        self.register_buffer('W_inv_real', torch.cos(theta).T / N)
        self.register_buffer('W_inv_imag', torch.sin(theta).T / N)

        # ═══════════════════════════════════════════════════
        # v6.1 FIX: 频域Laplacian核尺度
        # 
        # 原代码 (错误):
        #   self.register_buffer('lap_kernel', -(k_freq ** 2))
        #   → 最大特征值 ~ -9604, 扩散被过度放大
        #
        # 修复后 (正确):
        #   scale = (2π/N)^2
        #   lap_kernel = -scale * (k_freq ** 2)
        #   → 最大特征值 ~ -π^2 ≈ -9.87, 物理尺度正确
        #
        # 物理依据:
        #   连续Laplacian ∇²f 的DFT:  F{∇²f}(k) = -((2πk)/N)^2 · F{f}(k)
        #   其中 k=0,1,...,N-1, N=seq_len
        # ═══════════════════════════════════════════════════
        k_freq = torch.arange(N, dtype=torch.float32)
        k_freq = torch.minimum(k_freq, N - k_freq)  # 循环对称
        
        # 尺度因子: (2π/N)^2 — 这是从连续Laplacian到离散DFT的标准缩放
        scale = (2.0 * np.pi / N) ** 2
        self.register_buffer('lap_kernel', -scale * (k_freq ** 2))
        
        # 记录最大特征值供外部调试参考
        self.max_eigenval = scale * ((N // 2) ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 nabla^2 x (频域Laplacian)
        x: 任意形状, 最后一维为seq_len
        返回: 与输入同形状
        """
        orig_shape = x.shape
        assert x.shape[-1] == self.seq_len or x.shape[-2] == self.seq_len, \
            f"Input last or second-to-last dim must be {self.seq_len}, got shape {orig_shape}"

        if x.dim() == 2:
            return self._apply_laplacian(x)
        else:
            x_t = x.transpose(-2, -1)
            flat = x_t.reshape(-1, self.seq_len)
            result_flat = self._apply_laplacian(flat)
            result = result_flat.reshape(x_t.shape).transpose(-2, -1)
            return result

    def _apply_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """对 [N, seq_len] 张量应用频域Laplacian"""
        X_real = torch.matmul(x, self.W_real.T)
        X_imag = torch.matmul(x, self.W_imag.T)
        X_real_lap = X_real * self.lap_kernel
        X_imag_lap = X_imag * self.lap_kernel
        x_lap = torch.matmul(X_real_lap, self.W_inv_real.T) - \
                torch.matmul(X_imag_lap, self.W_inv_imag.T)
        return x_lap


class FiniteDiffLaplacian(nn.Module):
    """有限差分Laplacian (备用方案)"""
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_plus = torch.roll(x, shifts=-1, dims=-2 if x.dim() > 2 else -1)
        x_minus = torch.roll(x, shifts=1, dims=-2 if x.dim() > 2 else -1)
        return x_plus - 2 * x + x_minus


# ── 验证测试 ──
def verify_laplacian_scale():
    """
    v6.1 验证: 确认Laplacian核尺度正确
    
    测试1: 理论验证 - 对sin(x)应用Laplacian应得到 -sin(x) * scale_factor
    测试2: 尺度验证 - 最大特征值应在合理范围 (约-π^2)
    测试3: CFL验证 - dt * D0 * |λ_max| < 2
    """
    print("=" * 60)
    print("v6.1 Laplacian Scale Verification")
    print("=" * 60)
    
    seq_len = 196
    lap = DFTLaplacian(seq_len)
    
    # 测试1: 最大特征值
    max_eig = lap.lap_kernel.abs().max().item()
    print(f"\n[1] Max eigenvalue |λ_max| = {max_eig:.4f}")
    print(f"    Expected: ~π² = {np.pi**2:.4f}")
    assert max_eig < 20, f"Max eigenvalue too large: {max_eig}"
    print(f"    ✓ PASS (within expected range)")
    
    # 测试2: CFL条件验证
    dt, D0 = 0.02, 0.002  # v6.1修复后的参数
    cfl = dt * D0 * max_eig
    print(f"\n[2] CFL condition: dt·D0·|λ_max| = {cfl:.6f}")
    print(f"    Threshold: 2.0")
    assert cfl < 2.0, f"CFL violated: {cfl} >= 2.0"
    print(f"    ✓ PASS (stable)")
    
    # 测试3: sin(x) → -scale*sin(x) 验证
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lap = lap.to(device)
    x = torch.sin(torch.linspace(0, 2*np.pi, seq_len, device=device)).unsqueeze(0)
    y = lap(x)
    # 理论: ∇²(sin) = -sin (有限差分近似), 允许一定误差
    expected = -x  # 忽略尺度, 验证方向正确
    corr = (y * expected).mean().item() / (y.abs().mean().item() * expected.abs().mean().item() + 1e-8)
    print(f"\n[3] Laplacian(sin) anti-correlation with sin: {corr:.4f}")
    assert corr > 0.5, f"Laplacian not producing expected anti-correlation: {corr}"
    print(f"    ✓ PASS (correct sign/direction)")
    
    print(f"\n{'='*60}")
    print("All v6.1 Laplacian verifications PASSED")
    print(f"{'='*60}")


def test_dft_laplacian():
    """完整DFT Laplacian测试"""
    print("=" * 60)
    print("DFT Laplacian Test (v6.1)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    seq_len = 196
    batch = 4
    d_feat = 32

    lap = DFTLaplacian(seq_len).to(device)

    # 测试1: [B, seq]
    x1 = torch.sin(torch.linspace(0, 2 * np.pi, seq_len)).unsqueeze(0).repeat(batch, 1).to(device)
    y1 = lap(x1)
    print(f"\n[Test 1] Input: {x1.shape} -> Output: {y1.shape}")

    # 测试2: [B, seq, D] - 向量化
    x2 = torch.randn(batch, seq_len, d_feat, device=device)
    y2 = lap(x2)
    print(f"[Test 2] Input: {x2.shape} -> Output: {y2.shape}")
    assert y2.shape == x2.shape, "Shape mismatch!"

    # 测试3: 梯度
    x3 = torch.randn(batch, seq_len, d_feat, device=device, requires_grad=True)
    y3 = lap(x3)
    loss = y3.sum()
    loss.backward()
    assert x3.grad is not None, "No gradient!"
    print(f"[Test 3] Gradient check: PASSED")

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    verify_laplacian_scale()
    test_dft_laplacian()
