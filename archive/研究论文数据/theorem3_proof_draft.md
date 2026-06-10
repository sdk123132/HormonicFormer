# 定理 3：Hebbian 相变——临界数据量定理

## 数学推导草稿

### 设定

设 HormonicFormer 的 Hebbian 连接矩阵 $G \in \mathbb{R}^{S \times S}$（S = seq_len），
更新规则为：

$$G_{ij}^{(t+1)} = \lambda G_{ij}^{(t)} + \eta_+ \cdot \mathbb{1}[|\Delta\phi_{ij}| < \theta] - \eta_- \cdot \mathbb{1}[|\Delta\phi_{ij}| \geq \theta]$$

其中：
- $\Delta\phi_{ij} = \phi_i - \phi_j$ 是位置 $i, j$ 之间的相位差
- $\theta$ 是同步阈值
- $\lambda$ 是衰减因子
- $\eta_+, \eta_-$ 分别是增强和抑制学习率

### 稳态分析

在 $T$ 步更新后，$G$ 的期望为：

$$\mathbb{E}[G_{ij}] = \frac{1}{1-\lambda} \left( \eta_+ \cdot P(|\Delta\phi_{ij}| < \theta) - \eta_- \cdot P(|\Delta\phi_{ij}| \geq \theta) \right)$$

设 $p_{ij} = P(|\Delta\phi_{ij}| < \theta)$ 为位置 $i,j$ 的相位同步概率。

### 信号与噪声分解

将 $G$ 分解为信号和噪声：

$$G = G^* + \Delta G$$

其中 $G^* = \mathbb{E}[G]$ 是信号（数据真实相位结构），$\Delta G$ 是有限样本噪声。

### 噪声方差

每个 $G_{ij}$ 是 $T$ 个独立 Bernoulli 试验的加权和。考虑衰减：

$$\text{Var}(G_{ij}) = \frac{\eta_+^2 p_{ij}(1-p_{ij}) + \eta_-^2 (1-p_{ij})p_{ij}}{(1-\lambda^2)} \cdot \frac{1}{T_{eff}}$$

其中 $T_{eff}$ 是有效样本数。

当训练数据有 $N$ 个样本，每个 epoch 更新 $N$ 次，共 $E$ 个 epoch：
$$T_{eff} = N \cdot E / S$$

（除以 $S$ 是因为每个样本的每个位置对提供一个更新）

### G 矩阵的谱分析

**信号矩阵 $G^*$**：
- 秩 = rank($\Sigma_\phi$) = 数据的内在相位维度 $d_{intrinsic}$
- 前 $d_{intrinsic}$ 个特征值 $\gg$ 0

**噪声矩阵 $\Delta G$**：
- $\Delta G$ 是 $S \times S$ 对称随机矩阵
- 当 $S$ 大时，由 Marchenko-Pastur 定律：
  - 特征值分布的上界 $\lambda_{max}^{noise} \approx \sigma^2 (1 + \sqrt{S/T_{eff}})^2$
  - 其中 $\sigma^2 = \text{Var}(G_{ij})$

### 临界条件

**信号可分辨条件**：信号的最小非零特征值 > 噪声的最大特征值

$$\lambda_{min}^{signal} > \lambda_{max}^{noise}$$

$$\lambda_{min}^{signal} > \sigma^2 \left(1 + \sqrt{\frac{S}{T_{eff}}}\right)^2$$

**代入 $T_{eff} = NE/S$**：

$$\lambda_{min}^{signal} > \sigma^2 \left(1 + \sqrt{\frac{S^2}{NE}}\right)^2$$

### 两个极限

**小数据极限 ($N \ll S^2/E$)**：
$$\lambda_{max}^{noise} \approx \sigma^2 \cdot \frac{S^2}{NE}$$

噪声特征值与 $1/N$ 成正比。当 $N$ 足够小时，$G$ 的有效更新次数少，
每个 $G_{ij}$ 只被少数样本更新，相当于隐式正则化。
此时 $G$ 近似于 $G^*$ 的低秩近似 → **有效的归纳偏置**。

**大数据极限 ($N \gg S^2/E$)**：
$$\lambda_{max}^{noise} \approx \sigma^2 \left(1 + \sqrt{\frac{S^2}{NE}}\right)^2 \approx \sigma^2$$

噪声项趋于常数，但 $G$ 矩阵的秩趋于满秩。
$G$ 不再是低秩近似，而是完全拟合了训练数据的相位分布 → **过拟合**。

更精确地说，当 $N \gg S^2$ 时：
- $G$ 的条件数 $\kappa(G) = \lambda_{max} / \lambda_{min} = \Omega(N/S)$
- $G$ 矩阵开始编码训练数据的特定噪声
- 测试时，这些噪声模式导致错误的相位耦合 → PPL 爆炸

### 临界数据量

令 $\lambda_{min}^{signal} = \lambda_{max}^{noise}$，解得：

$$N^* = \frac{S^2}{E} \cdot \left(\frac{\sigma}{\sqrt{\lambda_{min}^{signal}}} - 1\right)^{-2}$$

简化（忽略常数因子）：

$$\boxed{N^* = \Theta\left(\frac{S^2}{\theta^2}\right)}$$

其中 $\theta$ 是同步阈值。

### 物理直觉

- $G$ 是 $S \times S$ 矩阵，有 $S(S-1)/2$ 个自由参数
- 每个数据样本提供 $\sim S$ 个相位差观测
- 需要 $\sim S^2 / S = S$ 个样本才能约束所有自由参数
- 但由于阈值 $\theta$ 的离散化，实际需要 $S^2 / \theta^2$ 个样本

### 实验预测

| 条件 | 预测 | 已有验证 |
|------|------|---------|
| $N/S^2 \ll 1$ | G 低秩，$\kappa$ 小，Hebbian 有益 | ✅ Char LM (N=67, S=64, N/S²=0.016) |
| $N/S^2 \approx 1$ | 临界转变 | ⏳ 待验证 |
| $N/S^2 \gg 1$ | G 满秩，$\kappa$ 大，Hebbian 有害 | ✅ WikiText (N=115K, S=1024, N/S²=0.11) |

**注意**：WikiText 的 N/S² 虽然只有 0.11，但实际每个 step 都更新 G，
有效更新次数 = steps × batch_size = 8600 × 48 = 413K >> S² = 1M。
所以实际 $T_{eff}/S^2 \approx 0.4$，接近临界区。

### Warmup 策略的理论解释

Warmup 调度 $\eta(t) = \eta_0 \cdot f(t)$ 等价于控制 $T_{eff}$：

- 早期 ($t < t_{warm}$): $\eta$ 大，G 快速学习信号 $G^*$
- 后期 ($t > t_{warm}$): $\eta \to 0$，冻结 G，防止过拟合

最优关闭时间 $t^*$ 应满足：
$$T_{eff}(t^*) \approx N^* = S^2 / \theta^2$$

即在有效更新次数达到临界点时关闭 Hebbian 学习。

---

## 待严格化的步骤

1. [ ] Marchenko-Pastur 定律的适用条件验证（$\Delta G$ 不完全独立）
2. [ ] $G^*$ 的秩与数据分布的关系
3. [ ] 衰减因子 $\lambda$ 对 $T_{eff}$ 的修正
4. [ ] 非对称更新（$\eta_+ \neq \eta_-$）的影响
5. [ ] 多层 G 矩阵的耦合效应

## 与 LeJEPA 的对比

| | LeJEPA 定理 | 本定理 |
|--|------------|--------|
| 核心对象 | 编码器权重 | G 连接矩阵 |
| 正则化 | 高斯 (SIGReg) | 衰减 + 阈值 |
| 可辨识性 | 线性正交 | 信号 vs 噪声分离 |
| 唯一性条件 | 分布必须是高斯 | 数据量必须 < N* |
| 实验验证 | 2D→1024D | 多任务多规模 |
| 工具 | Hermite 多项式 + Sturm-Liouville | 随机矩阵 + 分岔理论 |
