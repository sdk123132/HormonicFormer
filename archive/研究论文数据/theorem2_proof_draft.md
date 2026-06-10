# 定理 2：Hebbian G 矩阵的谱收敛

## 数学推导草稿

### 2.1 设定与符号

设数据分布为 $\mathcal{D}$，序列长度 $S$，相位向量 $\phi \in [0, 2\pi)^S$。

定义**数据相位相关矩阵**：
$$\Sigma_{ij} = \mathbb{E}_{x \sim \mathcal{D}}[\cos(\phi_i - \phi_j)]$$

Hebbian 更新规则（简化版）：
$$G_{ij}^{(t+1)} = \lambda G_{ij}^{(t)} + \eta \cdot \mathbb{1}[|\Delta\phi_{ij}| < \theta]$$

其中 $\Delta\phi_{ij} = \phi_i - \phi_j$，$\theta$ 为同步阈值。

### 2.2 稳态分析

设稳态时 $G^* = \mathbb{E}[G]$，则：
$$G^*_{ij} = \frac{\eta}{1-\lambda} \cdot P(|\Delta\phi_{ij}| < \theta)$$

对于高斯分布的相位差，$\Delta\phi_{ij} \sim \mathcal{N}(0, \sigma_{ij}^2)$：
$$P(|\Delta\phi_{ij}| < \theta) = \text{erf}\left(\frac{\theta}{\sqrt{2}\sigma_{ij}}\right)$$

### 2.3 与数据相关矩阵的联系

关键观察：当阈值 $\theta$ 适当时，
$$\mathbb{1}[|\Delta\phi_{ij}| < \theta] \approx \cos(\Delta\phi_{ij})$$

这是因为：
- $\cos(\Delta\phi) = 1$ 当 $\Delta\phi = 0$
- $\cos(\Delta\phi) \approx 1 - \frac{(\Delta\phi)^2}{2}$ 当 $\Delta\phi$ 小
- 阈值 $\theta$ 截断了大的 $\Delta\phi$

因此：
$$G^*_{ij} \approx \frac{\eta}{1-\lambda} \cdot \mathbb{E}[\cos(\phi_i - \phi_j)] = \frac{\eta}{1-\lambda} \cdot \Sigma_{ij}$$

### 2.4 谱收敛定理

**定理 2** (Spectral Convergence of G). 

设数据相位相关矩阵 $\Sigma$ 的特征值为 $\lambda_1(\Sigma) \geq \lambda_2(\Sigma) \geq ... \geq \lambda_S(\Sigma)$，
Hebbian G 矩阵的特征值为 $\lambda_k(G)$。

在足够多的更新步后，以概率至少 $1-\delta$：
$$|\lambda_k(G) - c \cdot \lambda_k(\Sigma)| \leq \epsilon$$

对所有 $k \leq r$ 成立，其中：
- $c = \frac{\eta}{1-\lambda}$ 是缩放因子
- $r = \text{rank}(\Sigma)$ 是数据内在维度
- $\epsilon = O\left(\sqrt{\frac{\log(S/\delta)}{T_{eff}}}\right)$ 是收敛误差
- $T_{eff}$ 是有效更新次数

**证明思路**：
1. G 的更新是在线 PCA 的变体
2. Oja's rule 的收敛性保证（引用 Oja 1982, Sanger 1989）
3. 阈值化引入的偏差分析
4. 矩阵 Bernstein 不等式控制方差

### 2.5 收敛速率

误差界：
$$\epsilon(T) = \frac{C}{\sqrt{T_{eff}}} + O\left(\frac{1}{T_{eff}}\right)$$

其中 $C$ 依赖于：
- 数据维度 $S$
- 特征值间隙 $\Delta\lambda = \lambda_r - \lambda_{r+1}$
- 学习率 $\eta$
- 衰减因子 $\lambda$

### 2.6 与定理 3 的联系

定理 3 的相变可以重新解释：
- 小数据时 ($T_{eff} < T^*$)：$\epsilon$ 大，G 是 $\Sigma$ 的粗糙近似 → 正则化效应
- 大数据时 ($T_{eff} > T^*$)：$\epsilon$ 小，G 精确拟合 $\Sigma$ → 过拟合训练数据的特定结构

临界点：
$$T^* = \frac{S^2 \cdot \log S}{\Delta\lambda^2}$$

### 2.7 实验验证设计

需要验证的预测：

1. **特征值对应**：G 的前 $k$ 个特征向量与 $\Sigma$ 的前 $k$ 个特征向量对齐
2. **缩放关系**：$\lambda_k(G) \approx c \cdot \lambda_k(\Sigma)$
3. **收敛速率**：误差随 $1/\sqrt{T}$ 下降

实验方法：
- 生成已知 $\Sigma$ 的合成数据
- 训练过程中定期提取 G
- 计算 G 和 $\Sigma$ 的特征分解
- 比较子空间对齐度（用 canonical angles）

---

## 待严格化的步骤

1. [ ] 证明阈值化 $\mathbb{1}[|\Delta\phi| < \theta]$ 与 $\cos(\Delta\phi)$ 的近似误差界
2. [ ] 用矩阵 Bernstein 不等式推导方差上界
3. [ ] 分析特征值间隙对收敛速率的影响
4. [ ] 扩展到多层 G 矩阵的耦合

## 关键引用

- Oja, E. (1982). Simplified neuron model as a principal component analyzer
- Sanger, T. D. (1989). Optimal unsupervised learning in a single-layer linear feedforward neural network
- Jain, P., et al. (2016). Streaming PCA: Matching matrix Bernstein and near-optimal finite sample guarantees for Oja's algorithm
