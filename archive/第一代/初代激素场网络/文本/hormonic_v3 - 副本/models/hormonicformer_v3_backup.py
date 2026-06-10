"""
HormonicFormer v3 - DCU Ready Version
完整模型定义，兼容PyTorch 2.7.1 + ROCm/HIP

新增强机制 (Priority 1-3):
  1. 局部 E/I 平衡 (EIBalance)     - 侧抑制防止强token主导
  2. 感觉反馈/外部驱动 (SensoryFeedback) - 演化中持续锚定输入
  3. Hebbian 实时突触可塑性 (HebbianPlasticity) - G矩阵中速经验学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

# 导入DFT Laplacian
sys.path.insert(0, str(Path(__file__).parent.parent / 'field'))
from laplacian_dft import DFTLaplacian, FiniteDiffLaplacian


# =============================================================================
# Priority 1: 局部 E/I 平衡 (侧抑制机制)
# =============================================================================
class EIBalance(nn.Module):
    """
    局部兴奋/抑制平衡器

    生物原理:
      - 每个 token 维护局部兴奋 E 和抑制 I 状态
      - 强活动的 token 通过侧抑制 (lateral inhibition) 压制邻近 token
      - 防止单个 token 或 head 的场振幅过大而主导全局

    数学:
      dE/dt = -E/tau_e + gamma_e * ReLU(psi_amplitude)
      dI/dt = -I/tau_i + gamma_i * local_avg_inhibition(psi_amplitude)
      modulation = E - I

    其中 local_avg_inhibition 是邻域内其他 token 活动的加权平均。
    """
    def __init__(self, seq_len, d_model, tau_e=2.0, tau_i=1.0,
                 gamma_e=1.0, gamma_i=0.8, w_inh=0.3, inh_radius=3):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.gamma_e = gamma_e
        self.gamma_i = gamma_i
        self.w_inh = w_inh
        self.inh_radius = inh_radius

        # 可学习的兴奋/抑制时间常数 (每个维度独立)
        self.tau_e_param = nn.Parameter(torch.ones(d_model) * tau_e)
        self.tau_i_param = nn.Parameter(torch.ones(d_model) * tau_i)

        # 局部侧抑制权重: 基于距离的高斯权重
        positions = torch.arange(seq_len).float().unsqueeze(0)  # [1, S]
        distances = torch.abs(positions - positions.T)  # [S, S]
        # 循环距离 (环形拓扑)
        distances = torch.minimum(distances, seq_len - distances)
        # 高斯权重，距离越近抑制越强
        inh_weights = torch.exp(-(distances ** 2) / (2 * inh_radius ** 2))
        # 自身抑制为0 (对角线)
        inh_weights.fill_diagonal_(0)
        # 归一化每行的权重
        inh_weights = inh_weights / (inh_weights.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer('inh_weights', inh_weights)  # [S, S]

        # 状态初始化
        self.E = None
        self.I = None

    def reset_state(self, batch_size, device):
        """为新 batch 重置 E/I 状态"""
        self.E = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        self.I = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)

    def forward(self, psi_amp):
        """
        psi_amp: [B, S, D] 场的振幅
        返回: modulation [B, S, D], 正值=净兴奋, 负值=净抑制
        """
        B, S, D = psi_amp.shape

        if self.E is None or self.E.shape[0] != B:
            self.reset_state(B, psi_amp.device)

        # 兴奋更新: 快速响应局部活动
        # dE = (-E + gamma_e * ReLU(amp)) / tau_e
        exc_input = self.gamma_e * F.relu(psi_amp)
        dE = (-self.E + exc_input) / (self.tau_e_param.view(1, 1, D) + 0.1)
        self.E = self.E + dE

        # 抑制更新: 局部侧抑制
        # 对每个维度，计算邻域内其他token的加权平均活动
        # inh_weights: [S, S] -> [1, S, S, 1]
        inh_weights = self.inh_weights.unsqueeze(0).unsqueeze(-1)  # [1, S, S, 1]
        psi_amp_expanded = psi_amp.unsqueeze(1)  # [B, 1, S, D]
        # 加权平均: [B, S, S, D] * [1, S, S, 1] -> sum over dim=2
        local_activity = (inh_weights * psi_amp_expanded).sum(dim=2)  # [B, S, D]
        inh_input = self.gamma_i * local_activity

        dI = (-self.I + inh_input) / (self.tau_i_param.view(1, 1, D) + 0.1)
        self.I = self.I + dI

        # E/I 平衡调制
        modulation = self.E - self.I  # [B, S, D]
        # 软限幅，防止极端值
        modulation = torch.tanh(modulation)
        return modulation


# =============================================================================
# Priority 2: 感觉反馈 / 外部驱动
# =============================================================================
class SensoryFeedback(nn.Module):
    """
    感觉反馈模块: 在CGL演化过程中持续注入外部输入

    生物原理:
      - 生物脑不是一次性接收输入后就自组织演化
      - 感觉皮层持续接收来自丘脑的外部驱动
      - 防止场因过度扩散而"遗忘"输入信息

    数学:
      I_ext(t) = feedback_strength * embed_input
      每步演化都加入: psi += dt * I_ext
    """
    def __init__(self, d_model, feedback_strength=0.3, feedback_freq=1):
        super().__init__()
        self.d_model = d_model
        self.feedback_strength = feedback_strength
        self.feedback_freq = feedback_freq

        # 可学习的反馈投影: 将输入嵌入投影到复数场空间
        self.W_feedback = nn.Linear(d_model, d_model * 2)

        self.stored_input = None
        self.step_count = 0

    def set_input(self, x_embed):
        """
        设置外部输入
        x_embed: [B, S, D] 输入嵌入
        """
        self.stored_input = x_embed.detach().clone()
        self.step_count = 0

    def get_feedback(self, batch_size, seq_len):
        """
        获取当前时刻的外部驱动
        返回: [B, S, 2*D] 或 None (如果不该注入)
        """
        if self.stored_input is None:
            return None

        self.step_count += 1
        if (self.step_count - 1) % self.feedback_freq != 0:
            return None

        # 投影到复数场空间 [B, S, 2*D]
        feedback = self.W_feedback(self.stored_input)
        feedback = feedback * self.feedback_strength
        return feedback

    def clear(self):
        self.stored_input = None
        self.step_count = 0


# =============================================================================
# Priority 3: Hebbian 实时突触可塑性
# =============================================================================
class HebbianPlasticity(nn.Module):
    """
    Hebbian 学习 + 反 Hebbian: G矩阵的中速经验学习

    生物原理 (STDP/Hebbian):
      - "一起激发的神经元连在一起" (Fire together, wire together)
      - 如果两个 token 的相位同步度高，强化它们之间的连接 G_ij
      - 如果同步度过低，削弱连接 (反 Hebbian)

    数学:
      同步度: sync_ij = cos(θ_i - θ_j)
      Hebbian:  ΔG_ij = η_hebb * ReLU(sync_ij - threshold)
      反Hebbian: ΔG_ij = -η_anti * ReLU(threshold - sync_ij)
      衰减:      G = G * decay (防止无限增长)

    时间尺度: tau_hebb ~ 10 steps (介于快速场演化和慢速BWO之间)
    """
    def __init__(self, seq_len, eta_hebb=0.001, eta_anti=0.0005,
                 sync_threshold=0.3, tau_hebb=10.0, decay=0.999):
        super().__init__()
        self.seq_len = seq_len
        self.eta_hebb = eta_hebb
        self.eta_anti = eta_anti
        self.sync_threshold = sync_threshold
        self.tau_hebb = int(tau_hebb)
        self.decay = decay

        # G 矩阵作为可学习的参数，但主要通过 Hebbian 规则更新
        self.G = nn.Parameter(torch.randn(seq_len, seq_len) * 0.01, requires_grad=True)
        # alive_mask 控制哪些连接存在
        self.alive_mask = nn.Parameter(torch.ones(seq_len, seq_len), requires_grad=False)

        self.step_counter = 0

    def compute_phase_sync(self, psi):
        """
        计算 token 间的相位同步度
        psi: [B, S, D, 2] (实部, 虚部)
        返回: sync_matrix [S, S] (跨batch平均)
        """
        B, S, D, _ = psi.shape
        re, im = psi[..., 0], psi[..., 1]  # [B, S, D]

        # 计算每个token的相位: [B, S, D]
        phase = torch.atan2(im, re + 1e-8)

        # 跨 batch 和维度平均，得到每个token的代表性相位
        mean_phase = phase.mean(dim=(0, 2))  # [S]

        # 计算相位差的余弦: sync_ij = cos(θ_i - θ_j)
        phase_diff = mean_phase.unsqueeze(0) - mean_phase.unsqueeze(1)  # [S, S]
        sync_matrix = torch.cos(phase_diff)  # [S, S]

        return sync_matrix

    def hebbian_update(self, psi):
        """
        基于相位同步度更新 G 矩阵
        psi: [B, S, D, 2]
        """
        self.step_counter += 1
        if self.step_counter % self.tau_hebb != 0:
            return

        with torch.no_grad():
            sync = self.compute_phase_sync(psi.detach())  # [S, S]
            self._last_sync = sync.detach()  # 保存同步矩阵供BWO重生使用

            # Hebbian: 同步度高 -> 增强
            hebb_term = self.eta_hebb * F.relu(sync - self.sync_threshold)
            # 反 Hebbian: 同步度低 -> 削弱
            anti_hebb_term = -self.eta_anti * F.relu(self.sync_threshold - sync)

            delta_G = hebb_term + anti_hebb_term

            # 应用更新 (只对存活的连接)
            self.G.data = (self.G.data + delta_G) * self.alive_mask
            # 缓慢衰减
            self.G.data = self.G.data * self.decay

            # 保持对角线为0 (不自连)
            self.G.data.fill_diagonal_(0)

    def get_G_matrix(self):
        """获取当前 G 矩阵 (应用 alive_mask)"""
        G_sparse = self.G * self.alive_mask
        # 归一化
        G_max = G_sparse.abs().max()
        if G_max > 1e-6:
            G_sparse = G_sparse / G_max
        # 对角线为0
        G_sparse = G_sparse.clone()
        G_sparse.fill_diagonal_(0)
        return G_sparse

    def prune_and_regrow(self, prune_fraction=0.5, regrow_fraction=0.1):
        """BWO: 强力剪枝 + 高同步重生
        
        prune_fraction: 每次剪掉当前存活连接的 50%（最弱的）
        regrow_fraction: 从死连接中重生 10%（仅高同步的）
        """
        with torch.no_grad():
            # DEBUG
            import logging
            logger = logging.getLogger('hormonic')
            logger.info(f"[BWO DEBUG] prune_fraction={prune_fraction}, regrow_fraction={regrow_fraction}")
            initial_alive = (self.alive_mask > 0).sum().item()
            logger.info(f"[BWO DEBUG] Initial alive: {initial_alive}/{self.alive_mask.numel()}")
            # === 1. 强力剪枝：使用 enforce_target_sparsity 直接达到目标 ===
            # 先尝试按比例剪枝，如果无效则强制达到目标稀疏度
            alive = (self.alive_mask > 0)
            n_alive = alive.sum().item()
            pruned = False
            if n_alive > 0:
                alive_weights = self.G[alive].abs()
                k = max(1, int(prune_fraction * n_alive))
                if k < len(alive_weights):
                    threshold = torch.topk(alive_weights, k, largest=False)[0][-1]
                    # 仅保留权重高于阈值的连接
                    self.alive_mask.data = (self.G.abs() > threshold).float()
                    # DEBUG
                    after_prune_alive = (self.alive_mask > 0).sum().item()
                    pruned_count = initial_alive - after_prune_alive
                    print(f"[BWO DEBUG] After prune: {after_prune_alive} alive (pruned {pruned_count})")
                    if pruned_count > 0:
                        pruned = True
            
            # 如果按比例剪枝无效，强制达到目标稀疏度
            if not pruned:
                print(f"[BWO DEBUG] Proportional prune failed, using enforce_target_sparsity")
                self.enforce_target_sparsity(target_sparsity=0.5)

            # === 2. 重生：只复活"高同步"的死连接 ===
            dead_mask = (self.alive_mask == 0)
            n_dead = dead_mask.sum().item()
            if n_dead > 0 and hasattr(self, '_last_sync'):
                sync = self._last_sync  # [seq_len, seq_len]
                should_regrow = dead_mask & (sync > 0.5)  # 同步度阈值
                candidates = torch.where(should_regrow.flatten())[0]
                n_regrow = max(1, int(regrow_fraction * n_dead))
                if len(candidates) > 0:
                    n_regrow = min(n_regrow, len(candidates))
                    sel = candidates[torch.randperm(len(candidates), device=candidates.device)[:n_regrow]]
                    self.alive_mask.data.flatten()[sel] = 1.0
                    self.G.data.flatten()[sel] = torch.randn(n_regrow, device=self.G.device) * 0.01
            elif n_dead > 0:
                # fallback: 如果没有同步信息，少量随机重生
                n_regrow = max(1, int(0.02 * n_dead))
                dead_indices = torch.where(dead_mask.flatten())[0]
                perm = torch.randperm(n_dead, device=dead_indices.device)[:n_regrow]
                regrow_indices = dead_indices[perm]
                self.alive_mask.data.flatten()[regrow_indices] = 1.0
                self.G.data.flatten()[regrow_indices] = torch.randn(
                    n_regrow, device=self.G.device) * 0.01

            # 强制掩码
            self.G.data = self.G.data * self.alive_mask
            self.G.data.fill_diagonal_(0)
            # DEBUG
            final_alive = (self.alive_mask > 0).sum().item()
            logger.info(f"[BWO DEBUG] Final: {final_alive} alive, sparsity={(1-final_alive/self.alive_mask.numel())*100:.2f}%")
class ComplexField(nn.Module):
    """复数Ginzburg-Landau场工具"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def amplitude(self, psi):
        """|ψ| = sqrt(Re² + Im²)"""
        re, im = psi[..., 0], psi[..., 1]
        return torch.sqrt(re**2 + im**2 + 1e-8)

    def phase(self, psi):
        """arg(ψ) = atan2(Im, Re)"""
        re, im = psi[..., 0], psi[..., 1]
        return torch.atan2(im, re)


# =============================================================================
# 核心 Hormonic Block V2 (整合 Priority 1-3)
# =============================================================================
class HormonicBlockV2(nn.Module):
    """
    HormonicFormer核心块 V2
    CGL场动力学 + 神经调质调制 + E/I平衡 + 感觉反馈 + Hebbian可塑性
    """
    def __init__(self, d_model, n_heads, seq_len, n_steps=3,
                 D0_amp=0.1, D0_phase=0.1, dt=0.05, use_dft=True,
                 # E/I 平衡参数
                 use_ei=True, tau_e=2.0, tau_i=1.0,
                 gamma_e=1.0, gamma_i=0.8, w_inh=0.3, inh_radius=3,
                 # 感觉反馈参数
                 use_feedback=True, feedback_strength=0.3, feedback_freq=1,
                 # Hebbian 参数
                 use_hebbian=True, eta_hebb=0.001, eta_anti=0.0005,
                 sync_threshold=0.3, tau_hebb=10.0, hebb_decay=0.999):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.n_steps = n_steps
        self.D0_amp = D0_amp
        self.D0_phase = D0_phase
        self.dt = dt

        self.use_ei = use_ei
        self.use_feedback = use_feedback
        self.use_hebbian = use_hebbian

        # 复数场工具
        self.cfield = ComplexField(d_model)

        # Laplacian算子
        if use_dft:
            self.laplacian = DFTLaplacian(seq_len)
        else:
            self.laplacian = FiniteDiffLaplacian(seq_len)

        # 线性投影
        self.W_in = nn.Linear(d_model, d_model * 2)
        self.W_out = nn.Linear(d_model * 2, d_model)

        # CGL参数 (可学习)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # ===== Priority 1: E/I 平衡 =====
        if self.use_ei:
            self.ei_balance = EIBalance(
                seq_len, d_model,
                tau_e=tau_e, tau_i=tau_i,
                gamma_e=gamma_e, gamma_i=gamma_i,
                w_inh=w_inh, inh_radius=inh_radius
            )

        # ===== Priority 2: 感觉反馈 =====
        if self.use_feedback:
            self.sensory_feedback = SensoryFeedback(
                d_model, feedback_strength, feedback_freq
            )

        # ===== Priority 3: Hebbian 可塑性 =====
        if self.use_hebbian:
            self.hebbian = HebbianPlasticity(
                seq_len, eta_hebb, eta_anti,
                sync_threshold, tau_hebb, hebb_decay
            )

        # LayerNorm (稳定训练)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, x_embed=None):
        """
        x: [batch, seq, d_model] 输入
        x_embed: [batch, seq, d_model] 原始输入嵌入 (用于感觉反馈)
        返回: [batch, seq, d_model]
        """
        batch, seq, _ = x.shape
        x = self.norm_in(x)

        # 消融模式：n_steps=0 时，跳过场演化，只做线性投影
        if self.n_steps == 0:
            # 纯嵌入基线模式
            output = self.W_out(self.W_in(x))
            return self.norm_out(output)

        # 投影到复数场
        psi = self.W_in(x)  # [B, S, 2*D]
        psi = psi.reshape(batch, seq, self.d_model, 2)  # [B, S, D, 2]

        # 设置感觉反馈输入
        if self.use_feedback and x_embed is not None:
            self.sensory_feedback.set_input(x_embed)

        # 重置 E/I 状态
        if self.use_ei:
            self.ei_balance.reset_state(batch, psi.device)

        # CGL场演化 (n_steps步)
        for step in range(self.n_steps):
            psi = self._cgl_step(psi, step)

        # 清除感觉反馈
        if self.use_feedback:
            self.sensory_feedback.clear()

        # 投影回实数
        psi_flat = psi.reshape(batch, seq, self.d_model * 2)
        output = self.W_out(psi_flat)
        output = self.norm_out(output)

        return output

    def prune_and_regrow(self, prune_fraction=0.5, regrow_fraction=0.1):
        """转发 BWO 剪枝请求到 Hebbian 可塑性模块"""
        if self.use_hebbian and self.hebbian is not None:
            self.hebbian.prune_and_regrow(prune_fraction, regrow_fraction)

    def _cgl_step(self, psi, step_idx):
        """
        单步CGL演化: Strang splitting + E/I平衡 + 感觉反馈 + Hebbian
        支持消融模式:
          - D0_amp=0: 跳过扩散
          - dt=0: 跳过反应（只做扩散）
        psi: [B, S, D, 2]
        """
        batch, seq, d, _ = psi.shape
        re, im = psi[..., 0], psi[..., 1]

        # === 消融模式: dt=0 时跳过所有反应，只做扩散 ===
        if self.dt == 0:
            # 只做扩散 (如果 D0_amp 也=0，则不做任何事)
            if self.D0_amp > 0:
                lap_re = torch.zeros_like(re)
                lap_im = torch.zeros_like(im)
                for i in range(d):
                    lap_re[:, :, i] = self.laplacian(re[:, :, i])
                    lap_im[:, :, i] = self.laplacian(im[:, :, i])
                re = re + self.D0_amp * lap_re
                im = im + self.D0_amp * lap_im
            return torch.stack([re, im], dim=-1)

        # === 计算 E/I 调制 (基于当前振幅) ===
        ei_mod = None
        if self.use_ei:
            amp = self.cfield.amplitude(psi)  # [B, S, D]
            ei_mod = self.ei_balance(amp)  # [B, S, D], 正值=净兴奋, 负值=净抑制

        # === 计算 Hebbian 更新 G (基于当前相位) ===
        if self.use_hebbian and step_idx == 0:
            self.hebbian.hebbian_update(psi)

        # === 获取 G 矩阵耦合 ===
        G_matrix = None
        if self.use_hebbian:
            G_matrix = self.hebbian.get_G_matrix()  # [S, S]

        # === 半步: 非线性反应 (local) + E/I调制 ===
        amp_sq = re**2 + im**2  # [B, S, D]

        # Soft-clip amp_sq 防止数值爆炸 (CGL饱和效应)
        amp_sq = torch.tanh(amp_sq)  # 限制在 [0, 1)

        # 有效非线性强度: alpha * (1 + E/I调制)
        # Clamp E/I调制范围，防止极端值
        if ei_mod is not None:
            alpha_eff = self.alpha * (1.0 + torch.clamp(ei_mod, -0.5, 1.0))  # [B, S, D]
        else:
            alpha_eff = self.alpha

        # CGL反应项 (带E/I调制，amp_sq已clipped)
        d_re = (alpha_eff * amp_sq - amp_sq**2) * re - self.beta * amp_sq * im
        d_im = (alpha_eff * amp_sq - amp_sq**2) * im + self.beta * amp_sq * re

        re_half = re + 0.5 * self.dt * d_re
        im_half = im + 0.5 * self.dt * d_im

        # === 整步: 扩散 (Laplacian) [可消融] ===
        if self.D0_amp > 0:
            lap_re = torch.zeros_like(re)
            lap_im = torch.zeros_like(im)

            for i in range(d):
                lap_re[:, :, i] = self.laplacian(re_half[:, :, i])
                lap_im[:, :, i] = self.laplacian(im_half[:, :, i])

            # 尺度归一化: laplacian输出除以seq_len使扩散步稳定
            # 这保证了 CFL 条件: dt * D0 * |laplacian| / seq_len < 2/seq_len
            # 同时保持物理正确性 (∇²sin = -sin)
            seq_len = lap_re.shape[1]
            lap_re = lap_re / seq_len
            lap_im = lap_im / seq_len

            D_eff = self.D0_amp
            re_full = re_half + self.dt * D_eff * lap_re
            im_full = im_half + self.dt * D_eff * lap_im
        else:
            # 消融: 无扩散
            re_full = re_half
            im_full = im_half

        # === 半步: 非线性反应 ===
        amp_sq_full = re_full**2 + im_full**2
        amp_sq_full = torch.tanh(amp_sq_full)  # Soft-clip 防止爆炸

        d_re2 = (alpha_eff * amp_sq_full - amp_sq_full**2) * re_full - self.beta * amp_sq_full * im_full
        d_im2 = (alpha_eff * amp_sq_full - amp_sq_full**2) * im_full + self.beta * amp_sq_full * re_full

        re_next = re_full + 0.5 * self.dt * d_re2
        im_next = im_full + 0.5 * self.dt * d_im2

        # === G矩阵耦合 (Hebbian结构连接) ===
        if G_matrix is not None:
            # G: [S, S] -> 对每个batch和维度做线性耦合
            # re_next: [B, S, D], G: [S, S]
            # 耦合: sum_j G_ij * re_j -> [B, S, D]
            G_coupling_re = torch.einsum('ij,bjd->bid', G_matrix, re_next)
            G_coupling_im = torch.einsum('ij,bjd->bid', G_matrix, im_next)
            coupling_strength = 0.1  # 耦合强度
            re_next = re_next + coupling_strength * G_coupling_re
            im_next = im_next + coupling_strength * G_coupling_im

        # === 感觉反馈: 外部驱动注入 ===
        if self.use_feedback:
            feedback = self.sensory_feedback.get_feedback(batch, seq)
            if feedback is not None:
                # feedback: [B, S, 2*D] -> [B, S, D, 2]
                fb = feedback.reshape(batch, seq, d, 2)
                re_next = re_next + self.dt * fb[..., 0]
                im_next = im_next + self.dt * fb[..., 1]

        return torch.stack([re_next, im_next], dim=-1)


# =============================================================================
# 神经调质系统 (DA/CB/G)
# =============================================================================
class Neuromodulator(nn.Module):
    """神经调质系统: DA + CB + G
    
    DA初始化改进:
      - da_ema初始值设为2.5 (接近初始CE loss for 10类分类)
      - ema_alpha=0.9 (更快跟踪loss变化，避免初始surprise过大)
      - 这样DA初始值~0.5 (中性探索)，而非~0.9 (过度兴奋)
    """
    def __init__(self, seq_len, d_model, da_init=2.5, ema_alpha=0.9):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # DA (多巴胺) - 预测误差/惊奇
        # 初始值设为接近初始loss，避免初始surprise爆炸
        self.da_ema = da_init
        self.da_var = 0.5  # 增大初始方差
        self.ema_alpha = ema_alpha

        # CB (大麻素) - 同步抑制
        self.cb_threshold = 0.25

        # G (胶质) - 结构可塑性
        self.G_global = nn.Parameter(torch.tensor(1.0))

    def update_DA(self, pred_loss, ema_alpha=None, var_alpha=0.95):
        """更新DA: 相对惊奇度 (改进版)"""
        alpha = ema_alpha if ema_alpha is not None else self.ema_alpha
        
        # EMA更新跟踪loss
        self.da_ema = alpha * self.da_ema + (1 - alpha) * pred_loss
        
        # 方差更新 (使用Welford算法更稳定)
        delta = pred_loss - self.da_ema
        self.da_var = var_alpha * self.da_var + (1 - var_alpha) * (delta ** 2)
        
        # 相对惊奇度 (标准化)
        std = math.sqrt(self.da_var) + 1e-8
        surprise = (pred_loss - self.da_ema) / std
        
        # sigmoid映射到[0,1]，然后clamp到[0.1, 0.9]
        DA = torch.sigmoid(torch.tensor(surprise))
        return torch.clamp(DA, 0.1, 0.9).item()

    def update_CB(self, sync_metric):
        """更新CB: 同步抑制"""
        if sync_metric > self.cb_threshold:
            CB = (sync_metric - self.cb_threshold) / (1 - self.cb_threshold)
            return min(CB, 1.0)
        return 0.0


# =============================================================================
# 完整 HormonicFormer V3 模型
# =============================================================================
class HormonicFormer(nn.Module):
    """HormonicFormer V3 - 整合生物启发的核心机制"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        d_model = config['model']['d_model']
        n_heads = config['model']['n_heads']
        n_layers = config['model']['n_layers']
        seq_len = config['model']['seq_len']
        n_steps = config['model']['n_steps']
        patch_size = config['model']['patch_size']
        n_classes = config['model']['n_classes']

        # 读取新机制配置
        ei_cfg = config['model'].get('ei_balance', {})
        fb_cfg = config['model'].get('sensory_feedback', {})
        hebb_cfg = config['model'].get('hebbian', {})

        # 输入嵌入
        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)

        # Hormonic Blocks V2
        self.blocks = nn.ModuleList([
            HormonicBlockV2(
                d_model=d_model,
                n_heads=n_heads,
                seq_len=seq_len,
                n_steps=n_steps,
                D0_amp=config['model']['D0_amp'],
                D0_phase=config['model']['D0_phase'],
                dt=config['model']['dt'],
                use_dft=True,
                # E/I 平衡
                use_ei=ei_cfg.get('enabled', True),
                tau_e=ei_cfg.get('tau_e', 2.0),
                tau_i=ei_cfg.get('tau_i', 1.0),
                gamma_e=ei_cfg.get('gamma_e', 1.0),
                gamma_i=ei_cfg.get('gamma_i', 0.8),
                w_inh=ei_cfg.get('w_inh', 0.3),
                inh_radius=ei_cfg.get('inh_radius', 3),
                # 感觉反馈
                use_feedback=fb_cfg.get('enabled', True),
                feedback_strength=fb_cfg.get('feedback_strength', 0.3),
                feedback_freq=fb_cfg.get('feedback_freq', 1),
                # Hebbian
                use_hebbian=hebb_cfg.get('enabled', True),
                eta_hebb=hebb_cfg.get('eta_hebb', 0.001),
                eta_anti=hebb_cfg.get('eta_anti', 0.0005),
                sync_threshold=hebb_cfg.get('sync_threshold', 0.3),
                tau_hebb=hebb_cfg.get('tau_hebb', 10.0),
                hebb_decay=hebb_cfg.get('decay', 0.999),
            )
            for _ in range(n_layers)
        ])

        # 神经调质
        self.neuromod = Neuromodulator(seq_len, d_model)

        # 预测编码
        pc_cfg = config.get('pc', {})
        if pc_cfg.get('use_pc', True):
            pred_hidden = d_model * pc_cfg.get('pred_hidden_mult', 4)
            self.predictor = nn.Sequential(
                nn.Linear(d_model, pred_hidden),
                nn.ReLU(),
                nn.Linear(pred_hidden, d_model)
            )
            self.aux_weight = pc_cfg.get('aux_weight', 0.01)
        else:
            self.predictor = None
            self.aux_weight = 0

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, images, targets=None):
        """
        images: [B, 1, 28, 28] (Fashion-MNIST) or [B, seq_len, d_model] (sequence)
        targets: [B] (可选)
        """
        B = images.shape[0]
        
        # Check if sequence input [B, S, d_model] or image [B, C, H, W]
        if images.dim() == 3:
            # Sequence input - skip patch_embed
            x = images  # [B, seq_len, d_model]
        else:
            # Image input - use patch_embed
            x = self.patch_embed(images)  # [B, d_model, H', W']
            x = x.flatten(2).transpose(1, 2)  # [B, seq_len, d_model]

        # 保存输入嵌入用于感觉反馈
        x_embed = x.detach().clone()

        # 通过Hormonic Blocks
        aux_loss = 0
        for i, block in enumerate(self.blocks):
            x_out = block(x, x_embed=x_embed if block.use_feedback else None)

            # 预测编码辅助损失
            if self.predictor is not None and i > 0:
                pred = self.predictor(x)
                aux_loss = aux_loss + F.mse_loss(pred, x_out.detach())

            x = x_out

        # Check if we need per-position output (sequence task) or pooled output (image task)
        if images.dim() == 3:
            # Sequence input: per-position classification
            logits = self.classifier(x)  # [B, seq_len, n_classes]
        else:
            # Image input: global pooling then classification
            x = x.mean(dim=1)  # [B, d_model]
            logits = self.classifier(x)  # [B, n_classes]

        if targets is not None:
            ce_loss = F.cross_entropy(logits, targets)
            total_loss = ce_loss + self.aux_weight * aux_loss
            return logits, total_loss

        return logits

    def update_neuromod(self, pred_loss, sync_metric=None):
        """更新神经调质状态"""
        DA = self.neuromod.update_DA(pred_loss)
        CB = self.neuromod.update_CB(sync_metric) if sync_metric else 0.0
        return DA, CB

    def prune_and_regrow(self, epoch, interval=5, prune_fraction=0.5, regrow_fraction=0.1):
        """BWO: 强力剪枝+高同步重生 (调用HormonicBlockV2)
        
        interval: BWO执行间隔 (epochs)
        prune_fraction: 每次剪掉存活连接的比例
        regrow_fraction: 从死连接中重生的比例
        """
        import logging
        logger = logging.getLogger('hormonic')
        logger.info(f"[HF BWO] Called at epoch {epoch}, interval={interval}")
        if epoch % interval == 0 and epoch > 0:
            logger.info(f"[HF BWO] Condition met, processing {len(self.blocks)} blocks")
            for i, block in enumerate(self.blocks):
                logger.info(f"[HF BWO] Block {i}: calling block.prune_and_regrow")
                block.prune_and_regrow(prune_fraction, regrow_fraction)
                logger.info(f"[HF BWO] Block {i}: done")

    def get_hebbian_stats(self):
        """获取Hebbian学习的统计信息"""
        stats = []
        for i, block in enumerate(self.blocks):
            if block.use_hebbian:
                G = block.hebbian.get_G_matrix()
                stats.append({
                    'layer': i,
                    'G_mean': G.mean().item(),
                    'G_std': G.std().item(),
                    'G_sparsity': (G.abs() < 0.01).float().mean().item(),
                    'alive_ratio': block.hebbian.alive_mask.mean().item(),
                })
        return stats


# =============================================================================
# 测试
# =============================================================================
def test_model():
    """测试 HormonicFormer V3"""
    print("="*70)
    print("HormonicFormer V3 - DCU Ready Test")
    print("新机制: E/I平衡 + 感觉反馈 + Hebbian可塑性")
    print("="*70)

    config = {
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'seq_len': 196,
            'n_steps': 3,
            'patch_size': 2,
            'n_classes': 10,
            'D0_amp': 0.1,
            'D0_phase': 0.1,
            'dt': 0.05,
            'ei_balance': {
                'enabled': True,
                'tau_e': 2.0,
                'tau_i': 1.0,
                'gamma_e': 1.0,
                'gamma_i': 0.8,
                'w_inh': 0.3,
                'inh_radius': 3,
            },
            'sensory_feedback': {
                'enabled': True,
                'feedback_strength': 0.3,
                'feedback_freq': 1,
            },
            'hebbian': {
                'enabled': True,
                'eta_hebb': 0.001,
                'eta_anti': 0.0005,
                'sync_threshold': 0.3,
                'tau_hebb': 10.0,
                'decay': 0.999,
            },
        },
        'pc': {
            'use_pc': True,
            'pred_hidden_mult': 4,
            'aux_weight': 0.01,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    model = HormonicFormer(config).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total params: {total_params/1e6:.2f}M")
    print(f"  Trainable: {trainable_params/1e6:.2f}M")

    # 测试前向
    dummy_input = torch.randn(2, 1, 28, 28).to(device)
    dummy_target = torch.randint(0, 10, (2,)).to(device)

    print(f"\nForward test:")
    print(f"  Input: {dummy_input.shape}")

    model.train()
    logits, loss = model(dummy_input, dummy_target)

    print(f"  Logits: {logits.shape}")
    print(f"  CE Loss: {loss.item():.4f}")

    # 测试反向
    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"  Gradients computed: {has_grad}")

    # 测试推理模式
    model.eval()
    with torch.no_grad():
        logits_eval = model(dummy_input)
        pred = logits_eval.argmax(dim=-1)
        print(f"  Eval logits: {logits_eval.shape}")
        print(f"  Predictions: {pred.tolist()}")

    # Hebbian 统计
    print(f"\nHebbian Stats:")
    for stat in model.get_hebbian_stats():
        print(f"  Layer {stat['layer']}: G_mean={stat['G_mean']:.4f}, "
              f"sparsity={stat['G_sparsity']:.2%}, alive={stat['alive_ratio']:.2%}")

    print("\n" + "="*70)
    print("Test passed! Model ready for DCU training.")
    print("="*70)


if __name__ == '__main__':
    test_model()
