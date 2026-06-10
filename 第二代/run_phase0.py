"""
HormonicFormer v7r3 - 阶段 0：模块单元测试
确认修复生效，无报错后进入阶段 1
"""
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("HormonicFormer v7r3 - 阶段 0：模块单元测试")
print("=" * 70)

# ================================================================================
# 模块定义（从 hormonic_v7r3_validated.py 精简）
# ================================================================================

class DFTLaplacian(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self._init_dft_matrix()

    def _init_dft_matrix(self):
        N = self.seq_len
        n = torch.arange(N, dtype=torch.float64)
        k = torch.arange(N, dtype=torch.float64).view(-1, 1)
        theta = -2 * np.pi * k * n / N
        self.register_buffer('W_real', torch.cos(theta).float())
        self.register_buffer('W_imag', torch.sin(theta).float())
        self.register_buffer('W_inv_real', torch.cos(theta).T.float() / N)
        self.register_buffer('W_inv_imag', torch.sin(theta).T.float() / N)
        k_freq = torch.arange(N, dtype=torch.float64)
        k_freq = torch.minimum(k_freq, N - k_freq)
        scale = (2.0 * np.pi / N) ** 2
        self.register_buffer('lap_kernel', (-scale * (k_freq ** 2)).float())
        self.max_eigenval = scale * ((N // 2) ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.seq_len:
            return self._apply_laplacian(x)
        elif x.shape[-2] == self.seq_len:
            *leading, S, D = x.shape
            x_flat = x.reshape(-1, S, D)
            x_flat = x_flat.permute(0, 2, 1).reshape(-1, S)
            lap_flat = self._apply_laplacian(x_flat)
            lap_flat = lap_flat.reshape(-1, D, S).permute(0, 2, 1)
            return lap_flat.reshape(*leading, S, D)
        else:
            raise ValueError(f"Last two dims must contain seq_len={self.seq_len}, got {x.shape}")

    def _apply_laplacian(self, x):
        X_real = torch.matmul(x, self.W_real.T)
        X_imag = torch.matmul(x, self.W_imag.T)
        X_real_lap = X_real * self.lap_kernel
        X_imag_lap = X_imag * self.lap_kernel
        return torch.matmul(X_real_lap, self.W_inv_real.T) + torch.matmul(X_imag_lap, self.W_inv_imag.T)


class ExponentialSTP(nn.Module):
    def __init__(self, n_heads: int, d_model: int, config: dict):
        super().__init__()
        stp_cfg = config.get('stp', {})
        self.U = stp_cfg.get('U', 0.2)
        self.tau_f = stp_cfg.get('tau_f', 1.0)
        self.tau_d = stp_cfg.get('tau_d', 3.0)
        self.dt = stp_cfg.get('dt', 0.05)
        n_groups = max(1, d_model // n_heads)
        self.register_buffer('u', torch.ones(1, n_groups) * self.U)
        self.register_buffer('r', torch.ones(1, n_groups))
        self.register_buffer('exp_f', torch.tensor(math.exp(-self.dt / self.tau_f)))
        self.register_buffer('exp_d', torch.tensor(math.exp(-self.dt / self.tau_d)))

    def step(self, spike_intensity: torch.Tensor):
        if not isinstance(spike_intensity, torch.Tensor):
            spike_intensity = torch.tensor([[float(spike_intensity)]], device=self.u.device)
        if spike_intensity.dim() == 0:
            spike_intensity = spike_intensity.view(1, 1)
        if spike_intensity.dim() == 1:
            spike_intensity = spike_intensity.unsqueeze(0)
        spike_avg = spike_intensity.mean(dim=0, keepdim=True)
        assert spike_avg.shape[1] == self.u.shape[1], f"STP group mismatch"
        u_relax = self.U + (self.u - self.U) * self.exp_f
        r_relax = 1.0 + (self.r - 1.0) * self.exp_d
        u_new = u_relax + self.U * (1.0 - u_relax) * spike_avg
        r_new = r_relax - u_relax * r_relax * spike_avg
        self.u.copy_(torch.clamp(u_new, 0.01, 1.0))
        self.r.copy_(torch.clamp(r_new, 0.01, 1.0))

    def get_efficacy(self):
        return self.u * self.r

    def get_modulation_factor(self):
        eff = self.get_efficacy()
        return 0.5 + eff

    def reset(self):
        self.u.fill_(self.U)
        self.r.fill_(1.0)


class DivisiveEIBalance(nn.Module):
    def __init__(self, seq_len, d_model, gamma_e=1.0, gamma_i=0.8, kappa=0.3, inh_radius=3):
        super().__init__()
        self.gamma_e = gamma_e
        self.gamma_i = gamma_i
        self.kappa = kappa
        positions = torch.arange(seq_len).float().unsqueeze(0)
        distances = torch.abs(positions - positions.T)
        distances = torch.minimum(distances, seq_len - distances)
        inh_weights = torch.exp(-(distances**2)/(2*inh_radius**2))
        inh_weights.fill_diagonal_(0)
        inh_weights = inh_weights / (inh_weights.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer('inh_weights', inh_weights)

    def forward(self, psi_amp):
        B, S, D = psi_amp.shape
        if S != self.inh_weights.shape[0]:
            positions = torch.arange(S, device=psi_amp.device).float().unsqueeze(0)
            distances = torch.abs(positions - positions.T)
            distances = torch.minimum(distances, S - distances)
            inh_weights = torch.exp(-(distances**2)/(2*3**2))
            inh_weights.fill_diagonal_(0)
            inh_weights = inh_weights / (inh_weights.sum(dim=1, keepdim=True) + 1e-8)
        else:
            inh_weights = self.inh_weights.to(psi_amp.device)
        excitation = self.gamma_e * F.relu(psi_amp)
        amp_sq = psi_amp ** 2
        self_inh = self.gamma_i * amp_sq
        neighbor = torch.einsum('ij,bjd->bid', inh_weights, amp_sq)
        lateral = self.kappa * neighbor
        return excitation / (1.0 + self_inh + lateral)


class CGLFieldEvolution(nn.Module):
    def __init__(self, d_model, seq_len, n_steps=10, D0_amp=0.002, D0_phase=0.002,
                 dt=0.02, noise_scale=0.001):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_steps = n_steps
        self.D0_amp = D0_amp
        self.D0_phase = D0_phase
        self.dt = dt
        self.noise_scale = noise_scale
        self.laplacian = DFTLaplacian(seq_len)
        self.alpha_raw = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.ei_balance = DivisiveEIBalance(seq_len, d_model)

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw)

    def _cgl_nonlinear(self, re, im, alpha, beta):
        amp_sq = re**2 + im**2
        coeff = alpha - amp_sq
        return coeff*re - beta*im, coeff*im + beta*re

    def _diffusion_step(self, re, im, D_amp, D_phase, dt):
        B, S, D = re.shape
        re_r = re.permute(0,2,1).reshape(B*D, S)
        im_r = im.permute(0,2,1).reshape(B*D, S)
        lr = self.laplacian(re_r).reshape(B,D,S).permute(0,2,1)
        li = self.laplacian(im_r).reshape(B,D,S).permute(0,2,1)
        return re + dt*D_amp*lr, im + dt*D_phase*li

    def _heun_halfstep(self, re, im, alpha, beta, dt_half):
        k1r, k1i = self._cgl_nonlinear(re, im, alpha, beta)
        rp, ip = re + dt_half*k1r, im + dt_half*k1i
        k2r, k2i = self._cgl_nonlinear(rp, ip, alpha, beta)
        return re + dt_half*0.5*(k1r+k2r), im + dt_half*0.5*(k1i+k2i)

    def forward(self, psi_in, alpha_eff=None, beta_eff=None,
                D_amp_override=None, D_phase_override=None):
        B, S, D, _ = psi_in.shape
        re, im = psi_in[...,0], psi_in[...,1]
        if alpha_eff is None: alpha_eff = self.alpha
        if beta_eff is None: beta_eff = self.beta
        Da = D_amp_override if D_amp_override is not None else self.D0_amp
        Dp = D_phase_override if D_phase_override is not None else self.D0_phase
        with torch.no_grad():
            mod = self.ei_balance(torch.sqrt(re**2 + im**2 + 1e-8))
        re, im = re*mod, im*mod
        h = 0.5 * self.dt
        for _ in range(self.n_steps):
            rh, ih = self._heun_halfstep(re, im, alpha_eff, beta_eff, h)
            rf, if_ = self._diffusion_step(rh, ih, Da, Dp, self.dt)
            re, im = self._heun_halfstep(rf, if_, alpha_eff, beta_eff, h)
            if self.noise_scale > 0:
                re += self.noise_scale * torch.randn_like(re) * math.sqrt(self.dt)
                im += self.noise_scale * torch.randn_like(im) * math.sqrt(self.dt)
        return torch.stack([re, im], dim=-1)


class MetaplasticHebbian(nn.Module):
    def __init__(self, seq_len, config):
        super().__init__()
        hc = config.get('hebbian', {})
        self.eta_pot = hc.get('eta_potentiate', 0.001)
        self.eta_dep = hc.get('eta_depress', 0.0005)
        self.thresh = hc.get('sync_threshold', 0.3)
        self.decay = hc.get('decay', 0.999)
        self.G = nn.Parameter(torch.randn(seq_len, seq_len) * 0.01)
        self.register_buffer('alive_mask', torch.ones(seq_len, seq_len))

    def get_sync_metric(self, phase):
        pd = phase.unsqueeze(2) - phase.unsqueeze(1)
        R = torch.complex(torch.cos(pd), torch.sin(pd)).mean(dim=0).abs()
        return R.mean().item()


class NeuromodulatorV7r3(nn.Module):
    def __init__(self, seq_len, d_model, config):
        super().__init__()
        nc = config.get('neuromod', {})
        self.register_buffer('da_ema', torch.tensor(nc.get('da_init', 2.5)))
        self.register_buffer('da_var', torch.tensor(0.1))
        self.da_ema_alpha = nc.get('da_ema_alpha', 0.9)
        self.da_var_alpha = nc.get('da_var_alpha', 0.9)
        self.da_min = nc.get('da_min', 0.1)
        self.da_max = nc.get('da_max', 0.9)
        self.use_cb = nc.get('use_cb', True)
        self.cb_gain = nc.get('cb_gain', 2.0)
        self.cb_thresh = nc.get('cb_threshold', 0.25)
        self.tau_cb = nc.get('tau_cb', 10.0)
        self.cb_dt = nc.get('cb_dt', 0.05)
        self.register_buffer('cb_state', torch.zeros(1))
        self.stp = ExponentialSTP(config.get('model', {}).get('n_heads', 4), d_model, config)
        self.prev_amp = None

    def compute_stp_step(self, amp):
        am = amp.mean().item()
        if self.prev_amp is not None:
            spike = min(abs(am - self.prev_amp) / (am + 1e-8), 5.0)
            # 扩展为 [1, n_groups] 匹配 STP 期望
            n_groups = self.stp.u.shape[1]
            spike_tensor = torch.full((1, n_groups), spike, device=self.stp.u.device)
            self.stp.step(spike_tensor)
        self.prev_amp = am

    def get_stp_modulation(self):
        return self.stp.get_modulation_factor()

    def reset(self):
        self.da_ema.fill_(2.5)
        self.da_var.fill_(0.1)
        self.cb_state.zero_()
        self.prev_amp = None
        self.stp.reset()


class HormonicBlockV7r3(nn.Module):
    def __init__(self, d_model, seq_len, config):
        super().__init__()
        mc = config.get('model', {})
        self.d_model = d_model
        self.seq_len = seq_len
        self.use_nm = config.get('use_neuromod', True)
        self.g_strength = config.get('g_coupling_strength', 0.1)

        self.cgl = CGLFieldEvolution(d_model, seq_len,
            n_steps=mc.get('n_cgl_steps', 10),
            D0_amp=mc.get('D0_amp', 0.002),
            D0_phase=mc.get('D0_phase', 0.002),
            dt=mc.get('cgl_dt', 0.02),
            noise_scale=mc.get('noise_scale', 0.001))
        self.hebbian = MetaplasticHebbian(seq_len, config)
        self.neuromod = NeuromodulatorV7r3(seq_len, d_model, config) if self.use_nm else None
        self.norm_re = nn.LayerNorm(d_model)
        self.norm_im = nn.LayerNorm(d_model)

    def forward(self, psi, surprise_score=None):
        B, S, D, _ = psi.shape
        stp_mod = 1.0

        if self.use_nm and self.neuromod is not None:
            amp = torch.sqrt(psi[...,0]**2 + psi[...,1]**2 + 1e-8)
            self.neuromod.compute_stp_step(amp)
            stp_raw = self.neuromod.get_stp_modulation()
            G_groups = stp_raw.shape[1]
            repeat = D // G_groups
            assert D % G_groups == 0
            stp_mod = stp_raw.view(1, 1, G_groups, 1).repeat_interleave(repeat, dim=2)

        psi = self.cgl(psi)

        re, im = psi[...,0], psi[...,1]
        gre = torch.einsum('ij,bjd->bid', self.hebbian.G.detach(), re)
        gim = torch.einsum('ij,bjd->bid', self.hebbian.G.detach(), im)
        psi = torch.stack([re + self.g_strength * gre, im + self.g_strength * gim], dim=-1)

        if isinstance(stp_mod, torch.Tensor):
            psi = psi * stp_mod.to(psi.device)

        re_out = self.norm_re(psi[..., 0])
        im_out = self.norm_im(psi[..., 1])
        psi = torch.stack([re_out, im_out], dim=-1)
        return psi


# ================================================================================
# 阶段 0 测试
# ================================================================================

def test_laplacian_multidim():
    """0.1 Laplacian 通用输入测试"""
    print("\n" + "-" * 60)
    print("[0.1] Laplacian 通用输入测试")
    print("-" * 60)
    
    lap = DFTLaplacian(128)
    results = []
    
    # 2D: (B, S)
    x2d = torch.randn(16, 128)
    y2d = lap(x2d)
    ok = y2d.shape == (16, 128)
    results.append(ok)
    status = "OK" if ok else "FAIL"
    print(f"  2D (16, 128) -> {y2d.shape} {status}")
    
    # 3D: (B, S, D)
    x3d = torch.randn(16, 128, 64)
    y3d = lap(x3d)
    ok = y3d.shape == (16, 128, 64)
    results.append(ok)
    status = "OK" if ok else "FAIL"
    print(f"  3D (16, 128, 64) -> {y3d.shape} {status}")
    
    # 4D: (B, H, S, D)
    x4d = torch.randn(2, 8, 128, 64)
    y4d = lap(x4d)
    ok = y4d.shape == (2, 8, 128, 64)
    results.append(ok)
    print(f"  4D (2, 8, 128, 64) -> {y4d.shape} {'OK' if ok else 'FAIL'}")
    
    # 5D: (B, H1, H2, S, D)
    x5d = torch.randn(2, 4, 8, 128, 64)
    y5d = lap(x5d)
    ok = y5d.shape == (2, 4, 8, 128, 64)
    results.append(ok)
    print(f"  5D (2, 4, 8, 128, 64) -> {y5d.shape} {'OK' if ok else 'FAIL'}")
    
    all_pass = all(results)
    print(f"\n  结果: {sum(results)}/4 {'OK 通过' if all_pass else 'FAIL 失败'}")
    return all_pass


def test_stp_batch():
    """0.2 STP batch 聚合测试"""
    print("\n" + "-" * 60)
    print("[0.2] STP batch 聚合测试")
    print("-" * 60)
    
    cfg = {'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05}}
    stp = ExponentialSTP(n_heads=4, d_model=64, config=cfg)  # 64/4=16 groups
    results = []
    
    # 初始状态检查
    ok = stp.u.shape == (1, 16) and stp.r.shape == (1, 16)
    results.append(ok)
    print(f"  初始 u.shape={stp.u.shape}, r.shape={stp.r.shape} {'OK' if ok else 'FAIL'}")
    
    # batch step
    stp.step(torch.randn(4, 16).abs())
    ok = stp.u.shape == (1, 16) and stp.r.shape == (1, 16)
    results.append(ok)
    print(f"  step后 u.shape={stp.u.shape}, r.shape={stp.r.shape} {'OK' if ok else 'FAIL'}")
    
    # efficacy 范围
    eff = stp.get_efficacy()
    ok = (eff > 0).all() and (eff <= 1).all()
    results.append(ok)
    print(f"  efficacy范围 (0,1]: min={eff.min():.4f}, max={eff.max():.4f} {'OK' if ok else 'FAIL'}")
    
    # 多次 step
    for i in range(10):
        stp.step(torch.randn(8, 16).abs())
    eff_final = stp.get_efficacy()
    ok = (eff_final > 0).all() and (eff_final <= 1).all()
    results.append(ok)
    print(f"  10次step后 efficacy: min={eff_final.min():.4f}, max={eff_final.max():.4f} {'OK' if ok else 'FAIL'}")
    
    all_pass = all(results)
    print(f"\n  结果: {sum(results)}/4 {'OK 通过' if all_pass else 'FAIL 失败'}")
    return all_pass


def test_block_forward():
    """0.3 Block 前向形状一致性测试"""
    print("\n" + "-" * 60)
    print("[0.3] Block 前向形状一致性测试")
    print("-" * 60)
    
    block_cfg = {
        'model': {'d_model': 64, 'seq_len': 128, 'n_layers': 2, 'n_heads': 4,
                  'n_cgl_steps': 3, 'D0_amp': 0.002, 'D0_phase': 0.002,
                  'cgl_dt': 0.02, 'noise_scale': 0.0, 'dropout': 0.0},
        'use_neuromod': False, 'use_pac': False, 'use_pc': False,
        'g_coupling_strength': 0.1, 'hebbian': {}
    }
    
    block = HormonicBlockV7r3(d_model=64, seq_len=128, config=block_cfg)
    psi = torch.randn(2, 128, 64, 2)  # [B=2, S=128, D=64, 2]
    
    results = []
    
    # 形状检查
    out = block(psi)
    ok = out.shape == (2, 128, 64, 2)
    results.append(ok)
    print(f"  输出形状 {out.shape} == (2, 128, 64, 2) {'OK' if ok else 'FAIL'}")
    
    # 无 NaN
    ok = not torch.isnan(out).any()
    results.append(ok)
    print(f"  无 NaN: {ok} {'OK' if ok else 'FAIL'}")
    
    # 无 Inf
    ok = not torch.isinf(out).any()
    results.append(ok)
    print(f"  无 Inf: {ok} {'OK' if ok else 'FAIL'}")
    
    # eval 模式也正常
    block.eval()
    out_eval = block(psi)
    ok = out_eval.shape == (2, 128, 64, 2) and not torch.isnan(out_eval).any()
    results.append(ok)
    print(f"  eval模式正常: {ok} {'OK' if ok else 'FAIL'}")
    
    all_pass = all(results)
    print(f"\n  结果: {sum(results)}/4 {'OK 通过' if all_pass else 'FAIL 失败'}")
    return all_pass


def test_stp_broadcast():
    """0.4 STP 调制不阻塞测试"""
    print("\n" + "-" * 60)
    print("[0.4] STP 调制广播测试")
    print("-" * 60)
    
    block_cfg = {
        'model': {'d_model': 64, 'seq_len': 128, 'n_layers': 2, 'n_heads': 4,
                  'n_cgl_steps': 3, 'D0_amp': 0.002, 'D0_phase': 0.002,
                  'cgl_dt': 0.02, 'noise_scale': 0.0, 'dropout': 0.0},
        'use_neuromod': True, 'use_pac': False, 'use_pc': False,
        'g_coupling_strength': 0.1,
        'neuromod': {'da_init': 2.5, 'da_ema_alpha': 0.9, 'use_cb': True},
        'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05},
        'hebbian': {'eta_potentiate': 0.001, 'eta_depress': 0.0005}
    }
    
    block = HormonicBlockV7r3(d_model=64, seq_len=128, config=block_cfg)
    psi = torch.randn(2, 128, 64, 2)
    
    results = []
    
    # 前向不报错
    try:
        out = block(psi)
        ok = out.shape == (2, 128, 64, 2)
        results.append(ok)
        print(f"  前向传播成功，输出形状 {out.shape} {'OK' if ok else 'FAIL'}")
    except Exception as e:
        print(f"  前向传播失败: {e} FAIL")
        results.append(False)
    
    # STP modulation 形状正确
    stp_mod = block.neuromod.get_stp_modulation()
    ok = stp_mod.shape == (1, 16)  # n_groups = 64/4 = 16
    results.append(ok)
    print(f"  STP modulation形状 {stp_mod.shape} == (1, 16) {'OK' if ok else 'FAIL'}")
    
    # modulation 值范围
    ok = (stp_mod > 0).all() and (stp_mod < 2).all()
    results.append(ok)
    print(f"  modulation范围 (0,2): min={stp_mod.min():.4f}, max={stp_mod.max():.4f} {'OK' if ok else 'FAIL'}")
    
    # 多次前向稳定
    for _ in range(5):
        out = block(torch.randn(2, 128, 64, 2))
    ok = not torch.isnan(out).any() and not torch.isinf(out).any()
    results.append(ok)
    print(f"  5次前向后稳定: {ok} {'OK' if ok else 'FAIL'}")
    
    all_pass = all(results)
    print(f"\n  结果: {sum(results)}/4 {'OK 通过' if all_pass else 'FAIL 失败'}")
    return all_pass


# ================================================================================
# 主程序
# ================================================================================

def run_all_tests():
    """运行阶段 0 全部测试"""
    print("\n" + "=" * 70)
    print("开始阶段 0 测试")
    print("=" * 70)
    
    all_results = []
    all_results.append(test_laplacian_multidim())
    all_results.append(test_stp_batch())
    all_results.append(test_block_forward())
    all_results.append(test_stp_broadcast())
    
    print("\n" + "=" * 70)
    print("阶段 0 测试总结")
    print("=" * 70)
    print(f"\n总结果: {sum(all_results)}/{len(all_results)} 项通过")
    
    if all(all_results):
        print("\nSUCCESS 阶段 0 全部通过！可以进入阶段 1（视觉任务回归）")
        print("\n下一步:")
        print("  python run_phase1_cifar10.py")
        return True
    else:
        print("\nFAIL 阶段 0 有失败项，请修复后再进入阶段 1")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
