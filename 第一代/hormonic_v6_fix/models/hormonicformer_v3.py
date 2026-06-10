"""
HormonicFormer v6.1 - 完整生物神经机制 (修复版)
兼容 PyTorch 2.7.1 + ROCm/HIP (DCU Ready)

v6.1 修复:
  1. Laplacian核尺度: -(k^2) → -((2πk/N)^2) [field/laplacian_dft.py]
  2. CGL参数: D0=0.002, dt=0.02, n_steps=10 (CFL稳定)
  3. DA初始值: da_init=2.5, ema_alpha=0.9 (避免初始surprise爆炸)
  4. STP跨batch重置: 每epoch开始时调用reset_all()
  5. 反馈迭代中可塑性屏蔽: update_plasticity=False
  6. G掩码缓存: 只在BWO时重新计算

v6 新增 (保留):
  6. 树突计算 (DendriticCompartment) - apical/basal双通道
  7. 短时突触可塑性 STP (Tsodyks-Markram)
  8. 稳态可塑性 (Homeostatic Scaling)
  9. Top-Down反馈回路

时间尺度谱:
  τ~0.2  STP促进/抑制
  τ~1    CGL场演化 + E/I平衡 + 感觉反馈 + 跨频耦合 + 能量门控
  τ~5    Top-Down反馈迭代
  τ~10   Hebbian可塑性
  τ~30   钙波扩散
  τ~50   稳态可塑性
  τ~100  BWO剪枝+重生
  τ~1000 DA/CB神经调质
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'field'))
from laplacian_dft import DFTLaplacian, FiniteDiffLaplacian


# ═══════════════════════════════════════════════════════════════════════════
# 位置编码
# ═══════════════════════════════════════════════════════════════════════════
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ═══════════════════════════════════════════════════════════════════════════
# [v5-1] 局部 E/I 平衡器
# ═══════════════════════════════════════════════════════════════════════════
class LocalEIBalance(nn.Module):
    """局部兴奋/抑制平衡 (Lateral Inhibition)"""
    def __init__(self, d_model: int, seq_len: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.gamma_E = nn.Parameter(torch.ones(n_heads) * 1.0)
        self.gamma_I = nn.Parameter(torch.ones(n_heads) * 0.5)
        self.kappa = nn.Parameter(torch.ones(n_heads) * 0.3)
        self.lateral_pool = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)

    def forward(self, re, im):
        B, S, D = re.shape
        H, Dh = self.n_heads, self.d_head
        re_h = re.view(B, S, H, Dh)
        im_h = im.view(B, S, H, Dh)
        amp_sq_h = (re_h.pow(2) + im_h.pow(2)).mean(dim=-1)
        self_inh = self.gamma_I.view(1, 1, H) * amp_sq_h
        amp_for_pool = amp_sq_h.permute(0, 2, 1).reshape(B * H, 1, S)
        neighbor_amp = self.lateral_pool(amp_for_pool).reshape(B, H, S).permute(0, 2, 1)
        lateral_inh = self.kappa.view(1, 1, H) * neighbor_amp
        excitation = self.gamma_E.view(1, 1, H)
        net_mod = (excitation - self_inh - lateral_inh).unsqueeze(-1)
        d_re = (net_mod * re_h).reshape(B, S, D)
        d_im = (net_mod * im_h).reshape(B, S, D)
        return d_re, d_im


# ═══════════════════════════════════════════════════════════════════════════
# [v5-2] Hebbian/STDP 突触可塑性
# ═══════════════════════════════════════════════════════════════════════════
class HebbianPlasticity(nn.Module):
    def __init__(self, seq_len: int, config: dict):
        super().__init__()
        hebb_cfg = config.get('hebbian', {})
        self.eta_potentiate = hebb_cfg.get('eta_potentiate', 0.001)
        self.eta_depress = hebb_cfg.get('eta_depress', 0.0005)
        self.sync_threshold = hebb_cfg.get('sync_threshold', 0.3)
        self.decay = hebb_cfg.get('decay', 0.999)

    @torch.no_grad()
    def update(self, G, alive_mask, phase):
        theta = phase.mean(dim=0)
        phase_diff = theta.unsqueeze(1) - theta.unsqueeze(0)
        sync = torch.cos(phase_diff)
        potentiation = F.relu(sync - self.sync_threshold) * self.eta_potentiate
        depression = F.relu(self.sync_threshold - sync) * self.eta_depress
        delta_G = (potentiation - depression) * alive_mask
        G_new = G * self.decay + delta_G
        return torch.clamp(G_new, -1.0, 1.0)

    def compute_sync_metric(self, phase):
        theta = phase.mean(dim=0)
        z = torch.complex(torch.cos(theta), torch.sin(theta))
        return z.mean().abs().item()


# ═══════════════════════════════════════════════════════════════════════════
# [v5-5] 胶质细胞系统
# ═══════════════════════════════════════════════════════════════════════════
class GlialSystem(nn.Module):
    def __init__(self, seq_len: int, d_model: int, config: dict):
        super().__init__()
        g = config.get('glial', {})
        self.energy_budget = g.get('energy_budget', 1.0)
        self.energy_penalty = g.get('energy_penalty', 0.1)
        self.register_buffer('energy_ema', torch.ones(seq_len) * 0.5)
        self.energy_alpha = g.get('energy_alpha', 0.9)
        self.calcium_diffusion = g.get('calcium_diffusion', 0.01)
        self.calcium_interval = g.get('calcium_interval', 3)

    @torch.no_grad()
    def compute_energy_gate(self, amp_sq):
        current_energy = amp_sq.mean(dim=0)
        self.energy_ema.copy_(
            self.energy_alpha * self.energy_ema + (1 - self.energy_alpha) * current_energy
        )
        return torch.sigmoid(
            (self.energy_budget - self.energy_ema) / (self.energy_penalty + 1e-8)
        )

    @torch.no_grad()
    def calcium_wave(self, G):
        d = self.calcium_diffusion
        G_row = (1 - 2 * d) * G + d * torch.roll(G, 1, 0) + d * torch.roll(G, -1, 0)
        return (1 - 2 * d) * G_row + d * torch.roll(G_row, 1, 1) + d * torch.roll(G_row, -1, 1)

    def energy_loss(self, amp_sq):
        return F.relu(amp_sq - self.energy_budget).mean() * self.energy_penalty


# ═══════════════════════════════════════════════════════════════════════════
# [v5-4] 跨频耦合
# ═══════════════════════════════════════════════════════════════════════════
class CrossFrequencyCoupling(nn.Module):
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.coupling_alpha = nn.Parameter(torch.ones(max(n_layers - 1, 1)) * 0.1)

    def apply_coupling(self, re, im, prev_phase, layer_idx):
        if prev_phase is None or layer_idx == 0:
            return re, im
        alpha = self.coupling_alpha[min(layer_idx - 1, len(self.coupling_alpha) - 1)]
        curr_phase = torch.atan2(im.mean(dim=-1), re.mean(dim=-1))
        modulation = (1.0 + alpha * torch.cos(prev_phase - curr_phase)).unsqueeze(-1)
        return re * modulation, im * modulation


# ═══════════════════════════════════════════════════════════════════════════
# [v6-6] 树突计算
# ═══════════════════════════════════════════════════════════════════════════
class DendriticCompartment(nn.Module):
    """树突双通道计算: basal(前馈) + apical(反馈) + soma(乘性门控)"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.W_basal = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.W_apical = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.soma_lambda = nn.Parameter(torch.tensor(0.5))
        self.W_soma = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_ff: torch.Tensor, x_td: torch.Tensor = None) -> torch.Tensor:
        basal = self.W_basal(x_ff)
        if x_td is not None:
            apical = self.W_apical(x_td)
            soma = basal * (1.0 + self.soma_lambda * apical)
        else:
            soma = basal
        return self.drop(self.W_soma(soma))


# ═══════════════════════════════════════════════════════════════════════════
# [v6-7] 短时突触可塑性 (STP)
# ═══════════════════════════════════════════════════════════════════════════
class ShortTermPlasticity(nn.Module):
    """Tsodyks-Markram STP: 促进(u↑) + 抑制(r↓) = efficacy(u*r)"""
    def __init__(self, seq_len: int, d_model: int, config: dict):
        super().__init__()
        stp_cfg = config.get('stp', {})
        self.U = stp_cfg.get('U', 0.2)
        self.tau_f = stp_cfg.get('tau_f', 1.0)
        self.tau_d = stp_cfg.get('tau_d', 3.0)
        self.dt = stp_cfg.get('dt', 0.05)
        self.register_buffer('u', torch.ones(seq_len) * self.U)
        self.register_buffer('r', torch.ones(seq_len))

    @torch.no_grad()
    def step(self, activity: torch.Tensor):
        """activity: [B, S] 每个token的活动量"""
        spike = activity.mean(dim=0)
        spike = torch.clamp(spike / (spike.max() + 1e-8), 0, 1)
        du = self.dt / self.tau_f * (self.U - self.u) + self.U * (1 - self.u) * spike
        self.u.add_(du)
        self.u.clamp_(0.01, 0.99)
        dr = self.dt / self.tau_d * (1.0 - self.r) - self.u * self.r * spike
        self.r.add_(dr)
        self.r.clamp_(0.01, 1.0)

    def get_efficacy(self) -> torch.Tensor:
        """v6.1 FIX: .detach().clone() 避免 inplace 操作破坏梯度图"""
        return (self.u * self.r).detach().clone()

    @torch.no_grad()
    def reset(self):
        """v6.1: 重置到基线状态 (每个epoch开始时调用)"""
        self.u.fill_(self.U)
        self.r.fill_(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# [v6-8] 稳态可塑性
# ═══════════════════════════════════════════════════════════════════════════
class HomeostaticPlasticity(nn.Module):
    """稳态可塑性: 目标活动率 → 慢速增益调节"""
    def __init__(self, seq_len: int, config: dict):
        super().__init__()
        h_cfg = config.get('homeostatic', {})
        self.target_rate = h_cfg.get('target_rate', 0.5)
        self.eta = h_cfg.get('eta', 0.001)
        self.ema_alpha = h_cfg.get('ema_alpha', 0.99)
        self.register_buffer('activity_ema', torch.ones(seq_len) * self.target_rate)
        self.register_buffer('gain', torch.ones(seq_len))

    @torch.no_grad()
    def update(self, activity: torch.Tensor):
        """activity: [B, S]"""
        current = activity.mean(dim=0)
        self.activity_ema.copy_(
            self.ema_alpha * self.activity_ema + (1 - self.ema_alpha) * current
        )
        error = self.target_rate - self.activity_ema
        self.gain.add_(self.eta * error)
        self.gain.clamp_(0.1, 3.0)

    def apply_gain(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, D]
        v6.1 FIX: .detach().clone() 避免 inplace 操作破坏梯度图
        问题: gain 参与前向计算后被 update() 的 .add_() 修改
        """
        return x * self.gain.detach().clone().unsqueeze(0).unsqueeze(-1)

    def get_activity_mean(self) -> float:
        """v6.1: 获取当前平均活动率 (用于诊断和校准target_rate)"""
        return self.activity_ema.mean().item()


# ═══════════════════════════════════════════════════════════════════════════
# 多头场注意力
# ═══════════════════════════════════════════════════════════════════════════
class FieldAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x, field_mod=None, stp_efficacy=None):
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, S, H, Dh).transpose(1, 2)
        k = self.W_k(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.W_v(x).view(B, S, H, Dh).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if field_mod is not None:
            attn = attn + field_mod.unsqueeze(1).unsqueeze(2) * 0.5
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        if stp_efficacy is not None:
            v = v * stp_efficacy.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out)


# ═══════════════════════════════════════════════════════════════════════════
# CGL 场动力学块 (v6.1: 修复CGL参数)
# ═══════════════════════════════════════════════════════════════════════════
class HormonicBlock(nn.Module):
    """
    HormonicFormer核心块
    v6.1: CGL参数改为 D0=0.002, dt=0.02, n_steps=10 (CFL稳定)
    """
    def __init__(self, d_model: int, n_heads: int, seq_len: int,
                 n_steps: int = 10, D0_amp: float = 0.002, D0_phase: float = 0.002,
                 dt: float = 0.02, noise_scale: float = 0.005,
                 dropout: float = 0.1, use_dft: bool = True,
                 feedback_strength: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.n_steps = n_steps
        self.D0_amp = D0_amp
        self.D0_phase = D0_phase
        self.dt = dt
        self.noise_scale = noise_scale
        self.feedback_strength = feedback_strength

        self.laplacian = DFTLaplacian(seq_len) if use_dft else FiniteDiffLaplacian(seq_len)
        self.W_in = nn.Linear(d_model, d_model * 2)
        self.W_out = nn.Linear(d_model * 2, d_model)
        self.W_feedback = nn.Linear(d_model, d_model * 2)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.ei_balance = LocalEIBalance(d_model, seq_len, n_heads)
        self.dendrite = DendriticCompartment(d_model, dropout)
        self.field_attn = FieldAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, x_input, DA=0.5, CB=0.0, G_mask=None,
                energy_gate=None, prev_phase=None,
                cfc=None, layer_idx=0,
                stp_efficacy=None, top_down=None):
        B, S, D = x.shape
        psi = self.W_in(self.norm1(x)).reshape(B, S, self.d_model, 2)
        feedback = self.W_feedback(x_input).reshape(B, S, self.d_model, 2) * self.feedback_strength

        for step in range(self.n_steps):
            psi = self._cgl_step(psi, feedback, DA, CB, G_mask,
                                 energy_gate, prev_phase, cfc, layer_idx)

        re, im = psi[..., 0], psi[..., 1]
        field_amp = torch.sqrt(re.pow(2).mean(-1) + im.pow(2).mean(-1) + 1e-8)
        phase = torch.atan2(im.mean(-1), re.mean(-1))
        cgl_out = self.W_out(psi.reshape(B, S, self.d_model * 2))
        cgl_out = self.dendrite(cgl_out, x_td=top_down)
        x = x + cgl_out
        attn_out = self.field_attn(self.norm2(x), field_mod=field_amp,
                                   stp_efficacy=stp_efficacy)
        x = x + attn_out
        x = x + self.ffn(self.norm3(x))
        return x, field_amp, phase

    def _cgl_step(self, psi, feedback, DA, CB, G_mask,
                  energy_gate, prev_phase, cfc, layer_idx):
        re, im = psi[..., 0], psi[..., 1]
        amp_sq = re.pow(2) + im.pow(2)
        # v6.1 FIX: DA 已经是 update_DA() 返回的 sigmoid-clamped 值 (范围 [0.1, 0.9])
        # 直接使用, 不再二次 sigmoid (避免动态范围被压缩到 [0.525, 0.711])
        DA_t = torch.tensor(DA, device=re.device)
        alpha_eff = self.alpha * (0.5 + DA_t)
        d_re = (alpha_eff * amp_sq - amp_sq.pow(2)) * re - self.beta * amp_sq * im
        d_im = (alpha_eff * amp_sq - amp_sq.pow(2)) * im + self.beta * amp_sq * re
        ei_re, ei_im = self.ei_balance(re, im)
        d_re, d_im = d_re + ei_re, d_im + ei_im
        re_half = re + 0.5 * self.dt * d_re
        im_half = im + 0.5 * self.dt * d_im

        # v6.1: Laplacian核尺度已修复, 扩散自然稳定
        lap_re = self.laplacian(re_half)
        lap_im = self.laplacian(im_half)
        D_amp_eff = self.D0_amp * (1.0 - CB)
        D_phase_eff = self.D0_phase * (1.0 - CB)
        re_full = re_half + self.dt * D_amp_eff * lap_re
        im_full = im_half + self.dt * D_phase_eff * lap_im

        if G_mask is not None:
            g_diag = G_mask.diag().unsqueeze(0).unsqueeze(-1)
            re_full = re_full * (1.0 + 0.1 * g_diag)
            im_full = im_full * (1.0 + 0.1 * g_diag)

        re_full = re_full + feedback[..., 0]
        im_full = im_full + feedback[..., 1]

        if energy_gate is not None:
            gate = energy_gate.unsqueeze(0).unsqueeze(-1)
            re_full, im_full = re_full * gate, im_full * gate

        if cfc is not None and prev_phase is not None:
            re_full, im_full = cfc.apply_coupling(re_full, im_full, prev_phase, layer_idx)

        # v6.1: 噪声 + soft-clip防止极端值
        if self.training and self.noise_scale > 0:
            ns = self.noise_scale * math.sqrt(self.dt)
            re_full = re_full + torch.randn_like(re_full) * ns
            im_full = im_full + torch.randn_like(im_full) * ns
            re_full = torch.tanh(re_full)
            im_full = torch.tanh(im_full)

        amp_sq2 = re_full.pow(2) + im_full.pow(2)
        d_re2 = (alpha_eff * amp_sq2 - amp_sq2.pow(2)) * re_full - self.beta * amp_sq2 * im_full
        d_im2 = (alpha_eff * amp_sq2 - amp_sq2.pow(2)) * im_full + self.beta * amp_sq2 * re_full
        ei_re2, ei_im2 = self.ei_balance(re_full, im_full)
        d_re2, d_im2 = d_re2 + ei_re2, d_im2 + ei_im2
        re_next = re_full + 0.5 * self.dt * d_re2
        im_next = im_full + 0.5 * self.dt * d_im2
        return torch.stack([re_next, im_next], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# [v6-9] Top-Down 反馈投影
# ═══════════════════════════════════════════════════════════════════════════
class TopDownProjection(nn.Module):
    def __init__(self, d_model: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                nn.Tanh(), nn.Dropout(dropout),
            )
            for _ in range(n_layers - 1)
        ])

    def project(self, h_upper: torch.Tensor, target_layer: int) -> torch.Tensor:
        idx = min(target_layer, len(self.projections) - 1)
        return self.projections[idx](h_upper)


# ═══════════════════════════════════════════════════════════════════════════
# 神经调质系统 (v6.1: +STP重置 + G掩码缓存)
# ═══════════════════════════════════════════════════════════════════════════
class Neuromodulator(nn.Module):
    """
    v6.1修复:
      - da_init=2.5 (避免初始surprise爆炸)
      - ema_alpha=0.9 (更快跟踪loss)
      - G掩码缓存 (只在BWO时重新计算)
      - STP/稳态/胶质 reset_all() 接口 (每个epoch重置)
    """
    def __init__(self, seq_len: int, d_model: int, config: dict):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        nm = config.get('neuromod', {})

        # v6.1 FIX: da_init=2.5 (接近初始CE loss)
        # da_ema 存 loss EMA (用于计算 surprise), da_value 存 sigmoid-DA (用于 CGL)
        self.register_buffer('da_ema', torch.tensor(nm.get('da_init', 2.5)))
        self.register_buffer('da_var', torch.tensor(0.5))
        # v6.1 FIX: da_value 存 update_DA() 返回的 sigmoid-DA (不是 loss EMA)
        self.register_buffer('da_value', torch.tensor(0.5))  # 初始中性探索
        self.da_min = nm.get('da_min', 0.1)
        self.da_max = nm.get('da_max', 0.9)
        # v6.1 FIX: 存储配置值确保 update_DA 使用正确的 alpha
        self.da_ema_alpha = nm.get('da_ema_alpha', 0.9)
        self.da_var_alpha = nm.get('da_var_alpha', 0.9)

        self.cb_threshold = nm.get('cb_threshold', 0.25)
        self.use_cb = nm.get('use_cb', False)

        # G矩阵
        init_density = nm.get('init_density', 0.3)
        self.register_buffer('G', torch.rand(seq_len, seq_len) * init_density)
        self.register_buffer('alive_mask', torch.ones(seq_len, seq_len))
        self.target_sparsity = nm.get('target_sparsity', 0.70)

        # v6.1 FIX: G掩码缓存
        self.register_buffer('_cached_G_mask', torch.zeros(seq_len, seq_len))
        self._G_mask_dirty = True

        # 子系统
        self.hebbian = HebbianPlasticity(seq_len, config)
        self.glial = GlialSystem(seq_len, d_model, config)
        self.stp = ShortTermPlasticity(seq_len, d_model, config)
        self.homeostatic = HomeostaticPlasticity(seq_len, config)

        self.register_buffer('step_counter', torch.tensor(0, dtype=torch.long))

    def update_DA(self, pred_loss):
        """
        v6.1 FIX: 
          - 使用 self.da_ema_alpha (配置值真正生效)
          - 更新 self.da_value (sigmoid-DA, 供 forward 使用)
          - 返回的是 sigmoid-DA (已 clamp 到 [da_min, da_max])
        """
        pred_loss_t = torch.tensor(pred_loss, device=self.da_ema.device)
        if torch.isnan(pred_loss_t) or torch.isinf(pred_loss_t):
            return self.da_value.item()
        # 使用实例变量中的配置值, 而非函数默认参数
        ema_alpha = self.da_ema_alpha
        var_alpha = self.da_var_alpha
        self.da_ema.copy_(ema_alpha * self.da_ema + (1 - ema_alpha) * pred_loss_t)
        self.da_var.copy_(var_alpha * self.da_var + (1 - var_alpha) * (pred_loss_t - self.da_ema).pow(2))
        surprise = (pred_loss_t - self.da_ema) / (self.da_var.sqrt() + 1e-8)
        da = torch.clamp(torch.sigmoid(surprise), self.da_min, self.da_max)
        self.da_value.copy_(da)  # 更新 da_value 供 forward 使用
        return da.item()

    def update_CB(self, sync_metric):
        if not self.use_cb or sync_metric is None:
            return 0.0
        if sync_metric > self.cb_threshold:
            return min((sync_metric - self.cb_threshold) / (1.0 - self.cb_threshold + 1e-8), 1.0)
        return 0.0

    # ═══════════════════════════════════════════════════
    # v6.1 FIX: G掩码缓存机制
    # 只在BWO(prune_G)时标记dirty, 其余时间返回缓存值
    # ═══════════════════════════════════════════════════
    def get_G_mask(self):
        """获取G掩码 (带缓存)"""
        if not self._G_mask_dirty:
            return self._cached_G_mask

        G_sparse = self.G * self.alive_mask
        G_abs = G_sparse.abs()
        G_max = G_abs.max()
        if G_max > 0:
            G_sparse = G_sparse / G_max
            flat = G_abs.flatten()
            k = int(self.target_sparsity * flat.numel())
            if 0 < k < flat.numel():
                thr = torch.topk(flat, k, largest=False)[0][-1]
                G_sparse = G_sparse * (G_abs >= thr).float()

        self._cached_G_mask.copy_(G_sparse)
        self._G_mask_dirty = False
        return self._cached_G_mask

    def _invalidate_G_cache(self):
        """标记G掩码缓存为dirty (BWO操作时调用)"""
        self._G_mask_dirty = True

    @torch.no_grad()
    def hebbian_update(self, phase):
        self.G.copy_(self.hebbian.update(self.G, self.alive_mask, phase))

    @torch.no_grad()
    def glial_update(self, amp_sq):
        self.step_counter += 1
        if self.step_counter % self.glial.calcium_interval == 0:
            self.G.copy_(self.glial.calcium_wave(self.G) * self.alive_mask)

    def get_energy_gate(self, amp_sq):
        return self.glial.compute_energy_gate(amp_sq).detach().clone()

    def get_energy_loss(self, amp_sq):
        return self.glial.energy_loss(amp_sq)

    def compute_sync_metric(self, phase):
        return self.hebbian.compute_sync_metric(phase)

    @torch.no_grad()
    def stp_step(self, activity):
        self.stp.step(activity)

    def get_stp_efficacy(self):
        return self.stp.get_efficacy()

    @torch.no_grad()
    def homeostatic_update(self, activity):
        self.homeostatic.update(activity)

    def apply_homeostatic_gain(self, x):
        return self.homeostatic.apply_gain(x)

    # ═══════════════════════════════════════════════════
    # v6.1 FIX: STP/稳态/胶质的epoch重置接口
    # 训练循环应在每个epoch开始时调用
    # ═══════════════════════════════════════════════════
    @torch.no_grad()
    def reset_all(self):
        """
        重置所有可塑性状态 (每个epoch开始时调用)
        防止STP资源耗尽、稳态增益漂移等问题
        """
        self.stp.reset()
        # 稳态可塑性增益重置为1.0 (中性)
        self.homeostatic.gain.fill_(1.0)
        # 活动EMA重置为目标值
        self.homeostatic.activity_ema.fill_(self.homeostatic.target_rate)
        # 胶质能量EMA重置
        self.glial.energy_ema.fill_(0.5)

    def prune_G(self, prune_ratio=0.05):
        """v6.1: 剪枝后标记G缓存为dirty"""
        flat_G = self.G.abs().flatten()
        k = int(prune_ratio * flat_G.numel())
        if k > 0:
            thr = torch.topk(flat_G, k, largest=False)[0][-1]
            self.alive_mask.copy_((self.G.abs() > thr).float())
        self._invalidate_G_cache()

    def enforce_mask(self):
        self.G.mul_(self.alive_mask)


# ═══════════════════════════════════════════════════════════════════════════
# 完整模型 (v6.1: 修复反馈迭代可塑性 + epoch重置)
# ═══════════════════════════════════════════════════════════════════════════
class HormonicFormer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        mc = config['model']

        d_model = mc['d_model']
        n_heads = mc['n_heads']
        n_layers = mc['n_layers']
        seq_len = mc['seq_len']
        n_steps = mc['n_steps']
        patch_size = mc['patch_size']
        dropout = mc.get('dropout', 0.1)
        D0_amp = mc.get('D0_amp', 0.002)
        D0_phase = mc.get('D0_phase', 0.002)
        dt = mc.get('dt', 0.02)
        noise_scale = mc.get('noise_scale', 0.005)
        feedback_strength = mc.get('feedback_strength', 0.1)

        self.n_layers = n_layers
        self.n_feedback_iters = config.get('topdown', {}).get('n_feedback_iters', 1)

        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_enc = LearnablePositionalEncoding(d_model, max_len=seq_len + 1)
        self.embed_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            HormonicBlock(
                d_model, n_heads, seq_len, n_steps,
                D0_amp, D0_phase, dt, noise_scale,
                dropout, True, feedback_strength,
            )
            for _ in range(n_layers)
        ])

        self.neuromod = Neuromodulator(seq_len, d_model, config)
        self.cfc = CrossFrequencyCoupling(d_model, n_layers)
        self.topdown = TopDownProjection(d_model, n_layers, dropout)

        pc_cfg = config.get('pc', {})
        self.use_pc = pc_cfg.get('use_pc', False)
        self.aux_weight = pc_cfg.get('aux_weight', 0.01)
        if self.use_pc:
            pred_hidden = d_model * pc_cfg.get('pred_hidden_mult', 4)
            self.predictors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, pred_hidden), nn.GELU(),
                    nn.Linear(pred_hidden, d_model),
                )
                for _ in range(n_layers - 1)
            ])
        else:
            self.predictors = None

        n_classes = mc['n_classes']
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()
        self._log_params()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[HormonicFormer v6.1] Total: {total:,} | Trainable: {trainable:,}")

    # ═══════════════════════════════════════════════════
    # v6.1 FIX: _run_layers 添加 update_plasticity 参数
    # 反馈迭代时设为 False, 避免可塑性更新两次
    # ═══════════════════════════════════════════════════
    def _run_layers(self, x, x_input, DA, CB, G_mask,
                    top_down_signals=None, update_plasticity=True):
        """
        单次前向: 所有层依次处理
        
        update_plasticity: bool
            True  = 主前向传播: 更新STP/Hebbian/Glial/Homeostatic
            False = 反馈迭代:   只计算输出, 不更新可塑性状态
        """
        aux_losses = []
        energy_losses = []
        layer_outputs = []
        prev_phase = None
        prev_amp_sq = None

        for i, block in enumerate(self.blocks):
            x_prev = x

            if i > 0 and prev_amp_sq is not None:
                energy_gate = self.neuromod.get_energy_gate(prev_amp_sq)
            else:
                energy_gate = None

            stp_eff = self.neuromod.get_stp_efficacy()

            td = None
            if top_down_signals is not None and i < len(top_down_signals):
                td = top_down_signals[i]

            x, field_amp, phase = block(
                x, x_input, DA=DA, CB=CB, G_mask=G_mask,
                energy_gate=energy_gate, prev_phase=prev_phase,
                cfc=self.cfc, layer_idx=i,
                stp_efficacy=stp_eff, top_down=td,
            )

            x = self.neuromod.apply_homeostatic_gain(x)
            prev_phase = phase.detach()
            prev_amp_sq = field_amp.pow(2).detach()
            layer_outputs.append(x)

            # v6.1 FIX: 只在主前向时更新可塑性, 反馈迭代时跳过
            if self.training and update_plasticity:
                self.neuromod.stp_step(prev_amp_sq)
                self.neuromod.hebbian_update(phase.detach())
                self.neuromod.glial_update(prev_amp_sq)
                self.neuromod.homeostatic_update(prev_amp_sq)
                energy_losses.append(self.neuromod.get_energy_loss(prev_amp_sq))

                if self.neuromod.use_cb:
                    sync_R = self.neuromod.compute_sync_metric(phase.detach())
                    CB = self.neuromod.update_CB(sync_R)

            if self.use_pc and self.predictors is not None and i > 0:
                pred = self.predictors[i - 1](x_prev)
                aux_losses.append(F.mse_loss(pred, x.detach()))

        return x, layer_outputs, aux_losses, energy_losses

    def forward(self, images, targets=None):
        B = images.shape[0]
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = self.pos_enc(x)
        x = self.embed_drop(x)
        x_input = x.detach()

        G_mask = self.neuromod.get_G_mask()
        # v6.1 FIX: 用 da_value (sigmoid-DA) 代替 da_ema (loss EMA)
        # da_value 由 update_DA() 每次更新, 是 [da_min, da_max] 范围内的有效 DA
        DA = self.neuromod.da_value.item()
        CB = 0.0

        # Pass 1: 纯前馈 (update_plasticity=True: 更新所有可塑性)
        x_ff, layer_outputs, aux_losses, energy_losses = self._run_layers(
            x, x_input, DA, CB, G_mask, top_down_signals=None,
            update_plasticity=True,
        )

        # Pass 2..K: 反馈迭代 (update_plasticity=False: 不重复更新)
        for fb_iter in range(self.n_feedback_iters):
            td_signals = [None] * self.n_layers
            for i in range(self.n_layers - 2, -1, -1):
                td_signals[i] = self.topdown.project(
                    layer_outputs[i + 1].detach() if fb_iter == 0
                    else layer_outputs[i + 1],
                    target_layer=i,
                )

            # v6.1 FIX: 反馈迭代不更新可塑性状态
            x_fb, layer_outputs, aux_fb, energy_fb = self._run_layers(
                x, x_input, DA, CB, G_mask,
                top_down_signals=td_signals,
                update_plasticity=False,  # ← 关键修复
            )
            aux_losses.extend(aux_fb)
            energy_losses.extend(energy_fb)
            x_ff = x_fb

        x_pool = x_ff.mean(dim=1)
        logits = self.classifier(x_pool)

        if targets is not None:
            ce_loss = F.cross_entropy(logits, targets)
            total_aux = torch.stack(aux_losses).mean() if aux_losses else torch.zeros(1, device=logits.device)
            total_energy = torch.stack(energy_losses).mean() if energy_losses else torch.zeros(1, device=logits.device)
            energy_w = self.config.get('glial', {}).get('energy_loss_weight', 0.01)
            total_loss = ce_loss + self.aux_weight * total_aux + energy_w * total_energy
            return logits, total_loss

        return logits

    def update_neuromod(self, pred_loss, sync_metric=None):
        DA = self.neuromod.update_DA(pred_loss)
        CB = self.neuromod.update_CB(sync_metric) if sync_metric else 0.0
        return DA, CB

    # ═══════════════════════════════════════════════════
    # v6.1 FIX: 每个epoch开始时重置可塑性状态
    # ═══════════════════════════════════════════════════
    def reset_neuromod_for_epoch(self):
        """每个epoch开始时调用: 重置STP/稳态/胶质状态"""
        self.neuromod.reset_all()

    def prune_and_regrow(self, epoch, interval=5, regrow_ratio=0.02):
        if epoch % interval == 0 and epoch > 0:
            self.neuromod.prune_G(prune_ratio=0.05)
            dead_mask = (self.neuromod.alive_mask == 0)
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                n_regrow = max(1, int(regrow_ratio * n_dead))
                dead_idx = torch.where(dead_mask.flatten())[0]
                perm = torch.randperm(len(dead_idx), device=dead_idx.device)[:n_regrow]
                sel = dead_idx[perm]
                self.neuromod.alive_mask.flatten()[sel] = 1.0
                self.neuromod.G.flatten()[sel] = torch.randn(
                    n_regrow, device=self.neuromod.G.device) * 0.01
            self.neuromod.enforce_mask()
            # v6.1 FIX: 重生后显式失效 G 掩码缓存
            # prune_G() 已标记 dirty, 但后续直接修改 G 值需要再次标记
            self.neuromod._invalidate_G_cache()

    def get_diagnostics(self):
        """v6.1: 增强诊断, 包含活动率"""
        G_sparse = self.neuromod.get_G_mask()
        stp_eff = self.neuromod.get_stp_efficacy()
        return {
            'DA': self.neuromod.da_ema.item(),
            'da_var': self.neuromod.da_var.item(),
            'G_sparsity': (G_sparse == 0).float().mean().item(),
            'alive_ratio': self.neuromod.alive_mask.mean().item(),
            'energy_mean': self.neuromod.glial.energy_ema.mean().item(),
            'stp_u_mean': self.neuromod.stp.u.mean().item(),
            'stp_r_mean': self.neuromod.stp.r.mean().item(),
            'stp_efficacy_mean': stp_eff.mean().item(),
            'homeo_gain_mean': self.neuromod.homeostatic.gain.mean().item(),
            'homeo_gain_std': self.neuromod.homeostatic.gain.std().item(),
            'homeo_activity_mean': self.neuromod.homeostatic.get_activity_mean(),
            'step_counter': self.neuromod.step_counter.item(),
        }
