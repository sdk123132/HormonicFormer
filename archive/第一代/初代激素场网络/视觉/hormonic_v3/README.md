# HormonicFormer v3 - Biological Neural Architecture

复数Ginzburg-Landau场动力学 + 神经调质调制 + DCU国产算力适配

---

## 核心机制 (v3 新增)

### Priority 1: 局部 E/I 平衡 (EIBalance)
- **生物原理**：兴奋/抑制动态平衡，侧抑制防止强token主导
- **实现**：每个token维护局部E/I状态，`modulation = tanh(E - I)`
- **效果**：提高表征多样性，防止场演化过度平滑

### Priority 2: 感觉反馈/外部驱动 (SensoryFeedback)
- **生物原理**：感觉皮层持续接收丘脑输入，防止"遗忘"
- **实现**：CGL演化中每步注入 `I_ext(t) = feedback_strength * W_feedback * x_embed`
- **效果**：锚定输入信息，缓解扩散过度导致的acc≈10%问题

### Priority 3: Hebbian 实时突触可塑性 (HebbianPlasticity)
- **生物原理**："一起激发的神经元连在一起"，STDP
- **实现**：基于相位同步度 `cos(theta_i - theta_j)` 更新G矩阵，tau~10
- **效果**：G矩阵具备"经验"快速调整能力，填补场演化(τ~1)和BWO(τ~100)之间的空缺

### Priority 4-5: 跨频耦合 + 能量约束 (配置中预留，当前关闭)

---

## 关键修复 (v3)

### 数值稳定性
- **问题**：Laplacian核 `-k^2` 最大达-9604，单步放大48x
- **修复**：核归一化 `/ max(|k^2|)` + CGL反应项 `amp_sq = tanh(amp_sq)`
- **结果**：Loss从NaN变为稳定收敛

### Laplacian DCU兼容
- 使用DFT矩阵替代 `torch.fft`，完全兼容ROCm/HIP

---

## 实验框架

### 改善一：长程探针实验
```bash
python experiments/copy_task.py --seq_lengths 16 64 256 512 --epochs 20
```
**目标**：证明场演化在S=256时准确率>90%

### 改善六：消融实验
```bash
python experiments/ablation.py --configs all --dataset fashion_mnist --epochs 10
```
**目标**：证明扩散/反应/EI/反馈/Hebbian各组件的因果贡献

### 训练
```bash
# 单卡
python scripts/train.py --config config.yaml

# 多卡 (DCU)
torchrun --standalone --nproc_per_node=2 scripts/train.py --config config.yaml
```

---

## 文件结构

```
hormonic_v3/
|-- config.yaml              # 主配置 (含v3新机制超参数)
|-- README.md                # 本文件
|-- BWO_ALGORITHM.md         # BWO算法伪代码与收敛性分析
|-- launch.sh                # DCU启动脚本
|-- setup.sh                 # 一键部署脚本
|-- test.sh                  # 快速测试
|-- field/
|   |-- laplacian_dft.py     # DFT矩阵Laplacian (DCU兼容)
|-- models/
|   |-- hormonicformer_v3.py # 完整模型 (v3新机制)
|-- scripts/
|   |-- train.py             # 训练脚本 (DDP + AMP + BWO)
|-- experiments/
    |-- copy_task.py         # 长程探针实验
    |-- ablation.py          # 消融实验
```

---

## 本地 RTX 5070 8GB 快速开始

```bash
# 1. 确保安装了 PyTorch + torchvision
pip install torch torchvision torchaudio pyyaml

# 2. 直接启动本地训练 (无需DCU环境)
python local_train.py --config local_config.yaml

# 显存预估: ~4-5GB (d_model=128, batch=64, n_layers=4)
```

### 本地配置说明 (`local_config.yaml`)

| 参数 | 服务器配置 | 本地5070配置 | 缩减原因 |
|------|----------|-------------|---------|
| d_model | 512 | **128** | 1/4参数量 |
| n_layers | 8 | **4** | 1/2层数 |
| batch_size | 256 | **64** | 8GB显存限制 |
| n_steps | 3 | **10** | v3.1数值稳定性修复 |
| D0_amp | 0.1 | **0.002** | CFL条件 |
| dt | 0.05 | **0.02** | CFL条件 |
| num_workers | 8 | **0** | Windows兼容 |
| da_init | 0.5 | **2.5** | 避免DA初始surprise爆炸 |

---

## 服务器 (DCU/多卡) 快速开始

```bash
# 1. 环境检查
bash test.sh

# 2. 启动训练
bash launch.sh

# 3. 长程探针实验
python experiments/copy_task.py --seq_lengths 16 64 256 --epochs 20

# 4. 消融实验
python experiments/ablation.py --configs all --epochs 10
```

---

## 叙事修正 (回应质疑)

| 原文可能引发的误解 | 修正后的表述 |
|-------------------|------------|
| "我们实现了大脑的完整模拟" | "我们对三个关键生物机制（E/I平衡、感觉反馈、Hebbian可塑性）进行了最小可行抽象" |
| "BWO与脑波有关" | "BWO的'Wave'指种群收缩-扩张的搜索动力学，灵感来自鲸群行为" |
| "DA/CB是完整的神经调质" | "DA捕捉多巴胺的RPE编码功能；CB实现活动依赖的逆向抑制" |
| "替代Transformer" | "在特定约束下（稀疏性、噪声鲁棒性），场演化提供了注意力的可行替代路径" |

---

## 论文改善清单

| 优先级 | 改善项 | 状态 | 预计时间 |
|--------|--------|------|---------|
| 1 | 长程探针实验 (Copy Task) | 代码完成 | 2-3天 (服务器) |
| 2 | 叙事调整 + BWO伪代码 | 文档完成 | 已完成 |
| 3 | 场演化消融实验 | 代码完成 | 1天 (服务器) |
| 4 | 中等规模任务 (CIFAR-100/WikiText) | 待启动 | 5-7天 (服务器) |
| 5 | Laplacian效率对比 | 待实现 | 1天 |
