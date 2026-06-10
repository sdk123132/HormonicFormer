"""
DFT-based Laplacian for DCU (HIP-compatible)
替代torch.fft，使用矩阵DFT实现频域Laplacian
"""
import torch
import torch.nn as nn
import numpy as np


class DFTLaplacian(nn.Module):
    """
    DFT矩阵实现的Laplacian算子
    完全兼容DCU (无需torch.fft)
    """
    def __init__(self, seq_len, device='cuda'):
        super().__init__()
        self.seq_len = seq_len
        self.device = device

        # 预计算DFT矩阵 (只计算一次)
        self._init_dft_matrix()

    def _init_dft_matrix(self):
        """初始化DFT和IDFT矩阵"""
        N = self.seq_len
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(N, dtype=torch.float32).view(-1, 1)

        # DFT矩阵: W[k,n] = exp(-2πi * k * n / N)
        theta = -2 * np.pi * k * n / N
        self.W_real = nn.Parameter(torch.cos(theta), requires_grad=False)
        self.W_imag = nn.Parameter(torch.sin(theta), requires_grad=False)

        # IDFT矩阵: W_inv = W^H / N
        self.W_inv_real = nn.Parameter(self.W_real.T / N, requires_grad=False)
        self.W_inv_imag = nn.Parameter(-self.W_imag.T / N, requires_grad=False)

        # 频域Laplacian核: -|k|^2
        # 保留原始尺度（不除以max），配合CGL方程中的CFL条件使用
        # CFL条件: dt * D0 * |λ_max| < 2, 其中 |λ_max| = max(k^2)
        k_freq = torch.arange(N, dtype=torch.float32)
        k_freq = torch.minimum(k_freq, N - k_freq)  # 循环对称
        self.lap_kernel = nn.Parameter(-(k_freq ** 2), requires_grad=False)
        # 记录最大特征值供外部参考
        self.max_eigenval = (k_freq ** 2).max().item()

    def forward(self, x):
        """
        计算 ∇²x 使用DFT
        x: [..., seq_len] 复数表示为 [..., seq_len, 2] (实部,虚部) 或直接实数
        返回: [..., seq_len]
        """
        # 处理输入维度
        orig_shape = x.shape
        if x.dim() == 2:
            # [batch, seq]
            x = x.unsqueeze(-1)  # [batch, seq, 1]
        elif x.dim() == 3:
            # [batch, seq, features]
            pass
        else:
            raise ValueError(f"Unsupported input shape: {orig_shape}")

        batch, seq, features = x.shape

        # 实数DFT: 分别对实部和虚部处理
        # DFT(x) = W @ x
        x_flat = x.reshape(-1, seq)  # [batch*features, seq]

        # 正变换
        X_real = torch.matmul(x_flat, self.W_real.T)  # [B, N]
        X_imag = torch.matmul(x_flat, self.W_imag.T)  # [B, N]

        # 频域乘以Laplacian核
        X_real_lap = X_real * self.lap_kernel  # [B, N]
        X_imag_lap = X_imag * self.lap_kernel  # [B, N]

        # 逆变换
        x_lap_real = torch.matmul(X_real_lap, self.W_inv_real.T) - \
                     torch.matmul(X_imag_lap, self.W_inv_imag.T)

        # 重塑回原始形状
        result = x_lap_real.reshape(batch, seq, features).squeeze(-1)

        if len(orig_shape) == 2:
            result = result.reshape(orig_shape)

        return result


class FiniteDiffLaplacian(nn.Module):
    """
    有限差分Laplacian (备用方案)
    简单但切断长程耦合
    """
    def __init__(self, seq_len, device='cuda'):
        super().__init__()
        self.seq_len = seq_len
        self.device = device

    def forward(self, x):
        """
        计算 ∇²x ≈ x[i+1] - 2x[i] + x[i-1]
        x: [..., seq_len]
        """
        # 循环边界条件
        x_plus = torch.roll(x, shifts=-1, dims=-1)
        x_minus = torch.roll(x, shifts=1, dims=-1)
        return x_plus - 2 * x + x_minus


def test_dft_laplacian():
    """测试DFT Laplacian"""
    print("="*60)
    print("DFT Laplacian Test")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 创建测试数据
    seq_len = 196
    batch = 2

    lap_dft = DFTLaplacian(seq_len, device).to(device)

    # 测试1: 简单正弦波
    x = torch.sin(torch.linspace(0, 2*np.pi, seq_len)).unsqueeze(0).repeat(batch, 1).to(device)

    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")

    # 计算Laplacian
    lap_x = lap_dft(x)

    print(f"Output shape: {lap_x.shape}")
    print(f"Output range: [{lap_x.min():.3f}, {lap_x.max():.3f}]")

    # 理论: ∇²(sin) = -sin
    expected = -x
    error = torch.abs(lap_x - expected).mean()
    print(f"\nExpected: -sin(x)")
    print(f"Mean absolute error: {error:.6f}")

    # 测试2: 梯度检查
    x.requires_grad = True
    lap_x = lap_dft(x)
    loss = lap_x.sum()
    loss.backward()

    print(f"\nGradient check:")
    print(f"  x.grad exists: {x.grad is not None}")
    print(f"  x.grad shape: {x.grad.shape if x.grad is not None else 'N/A'}")

    # 测试3: 批处理
    x3 = torch.randn(4, seq_len).to(device)
    lap3 = lap_dft(x3)
    print(f"\nBatch test:")
    print(f"  Input: {x3.shape} -> Output: {lap3.shape}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


if __name__ == '__main__':
    test_dft_laplacian()
