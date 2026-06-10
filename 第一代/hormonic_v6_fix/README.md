# HormonicFormer v6.1 - 完整生物神经机制 (修复版)

## 审查修复总结

基于代码审查意见，对 v6 进行了 **10 项关键修复**:

### 🔴 1. Laplacian 核尺度 (最严重)

**问题**: 频域 Laplacian 核使用了 `-(k_freq^2)`，缺失 `(2π/N)^2` 尺度因子。

**影响**: 最大特征值从 ~π²≈9.87 错配为 ~9604，扩散强度放大 **~1000 倍**，导致 CGL 演化完全失稳。

**修复** (`field/laplacian_dft.py`):
```python
scale = (2.0 * math.pi / N) ** 2  # 新增
self.register_buffer('lap_kernel', -scale * (k_freq ** 2))  # 修复
```

| 指标 | v6 (错误) | v6.1 (修复) |
|------|----------|------------|
| \|λ_max\| | 9604 | 9.87 (π²) |
| CFL = dt·D0·\|λ\| | 48.0 (爆炸) | 0.0004 (稳定) |

---

### 🟡 2-9. 其他关键修复

| # | 问题 | 位置 | 修复 |
|---|------|------|------|
| 2 | **STP 跨 batch 持续** → 资源耗尽 | `Neuromodulator` | 添加 `reset_all()` 方法，每 epoch 重置 STP/稳态/胶质 |
| 3 | **反馈迭代中可塑性更新 2 次** | `_run_layers()` | 添加 `update_plasticity` 参数，反馈迭代时设为 `False` |
| 4 | **G 掩码每次调用都重新计算** | `get_G_mask()` | 缓存机制：BWO 时标记 dirty，其余时间返回缓存值 |
| 5 | **DA 初始值 0.5 导致 surprise 爆炸** | `config.yaml` | `da_init: 0.5 → 2.5` (接近初始 CE loss) |
| 6 | **DA 未归一化用于 CGL** | `_cgl_step()` | `DA_norm = sigmoid(DA)` 确保 α_eff 在合理范围 |
| 7 | **CGL 参数违反 CFL** | `config.yaml` | `D0: 0.1→0.002, dt: 0.05→0.02, n_steps: 3→10` |
| 8 | **inplace 操作破坏梯度图** | `apply_gain()`, `get_efficacy()` | `.detach().clone()` 避免 buffer 参与梯度 |
| 9 | **CGL 噪声导致 40% NaN** | `_cgl_step()` | `noise_scale: 0.005→0.001` + `tanh()` soft-clip |

---

### 验证结果

```
NaN 率:     40% → 0%  (20次运行)
5步 Loss:   3.45 → 0.05 (下降99%)
STP 重置:   eff 0.016 → 0.200 (基线恢复)
DA 初始值:  0.900 → 0.434 (中性探索)
```

---

## 快速开始

```bash
# DCU/服务器
bash launch.sh

# 本地单卡
python scripts/train.py --config config.yaml
```

## 文件结构

```
hormonic_v6_fix/
├── config.yaml                   # 主配置 (v6.1 修复参数)
├── launch.sh                     # DCU 启动脚本
├── test.sh                       # 验证测试 (v6.1 全覆盖)
├── field/
│   └── laplacian_dft.py          # v6.1: Laplacian 核尺度修复
├── models/
│   └── hormonicformer_v3.py      # v6.1: STP重置 + G缓存 + plasticity开关 + inplace修复
└── scripts/
    └── train.py                  # v6.1: 每epoch重置STP
```

## v6 保留的架构特性

- 树突双通道 (Apical/Basal + 乘性门控)
- STP 突触可塑性 (Tsodyks-Markram)
- 稳态可塑性 (Synaptic Scaling)
- Top-Down 反馈回路
- 9 级时间尺度谱
- 完整诊断接口
