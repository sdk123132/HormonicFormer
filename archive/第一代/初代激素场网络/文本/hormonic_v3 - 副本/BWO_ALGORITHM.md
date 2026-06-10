# BWO (Bi-Phase Wave Optimization) 算法

## 1. 算法伪代码

```
Algorithm 1: BWO for Sparse Neural Architecture Search
=======================================================
Input:
  G(0)          : 初始连接矩阵 [S x S]
  A(0)          : 初始存活掩码 {0,1}^(SxS)
  P             : 种群大小 (default: 20)
  T             : 最大进化迭代次数
  sparsity_min  : 最小稀疏度 (default: 0.65)
  sparsity_max  : 最大稀疏度 (default: 0.75)
  flip_ratio    : 重生比例 (default: 0.02)
  elite_ratio   : 精英保留比例 (default: 0.1)
  spiral_rate   : 螺旋收缩率 (default: 0.9)

Output:
  A*            : 优化后的二值存活掩码

1:  function BWO(G, A, P, T):
2:    // 初始化种群
3:    Population <- {A}  // 包含当前掩码
4:    for p = 2 to P:
5:      A_p <- RandomMask(sparsity_min, sparsity_max)
6:      Population <- Population U {A_p}
7:
8:    // 主循环
9:    for t = 1 to T:
10:     // 评估适应度 (使用验证集准确率)
11:     Fitness <- Evaluate(Population, eval_samples=2000)
12:     A_best <- argmax(Fitness)  // 当前最优
13:
14:     // 精英保留 (Elitism)
15:     n_elite <- ceil(elite_ratio * P)
16:     NextGen <- TopK(Population, n_elite)  // 前k直接进入下一代
17:
18:     // 螺旋收缩 (Spiral Contraction)
19:     for p = n_elite+1 to P:
20:       A_parent <- TournamentSelect(Population, Fitness)
21:       // 向最优解螺旋移动: 概率性地将A_parent的比特翻转为A_best
22:       A_child <- A_parent
23:       for each bit (i,j):
24:         if A_best[i,j] != A_parent[i,j]:
25:           p_flip <- spiral_rate * (1 - t/T)  // 随时间衰减
26:           with probability p_flip:
27:             A_child[i,j] <- A_best[i,j]
28:       NextGen <- NextGen U {A_child}
29:
30:     // 鲸落突变 (Whale Fall Mutation)
31:     n_fall <- max(1, floor(0.1 * P))
32:     for p = 1 to n_fall:
33:       A_dead <- RandomSelect(NextGen)
34:       // 随机重置部分连接
35:       n_reset <- floor(flip_ratio * S^2)
36:       Randomly reset n_reset bits of A_dead
37:
38:     // 硬约束投影 (Hard Constraint Projection)
39:     for each A in NextGen:
40:       current_sparsity <- 1 - sum(A) / S^2
41:       if current_sparsity < sparsity_min:
42:         // 随机激活额外连接直到满足最小稀疏度
43:         Randomly activate (sparsity_min - current_sparsity) * S^2 bits
44:       elif current_sparsity > sparsity_max:
45:         // 随机剪枝连接直到满足最大稀疏度
46:         Randomly prune (current_sparsity - sparsity_max) * S^2 bits
47:
48:     Population <- NextGen
49:
50:   return A_best
```

## 2. 收敛性分析

### 2.1 基本性质

**BWO属于群体智能随机搜索算法家族**，其搜索空间为离散的 {0,1}^(SxS) 二值矩阵空间，状态总数为 2^(S^2)。

**性质1 (遍历性)**：鲸落突变算子以非零概率随机重置任意比特，因此在极限情况下 (T -> inf)，算法能以概率1访问搜索空间中的所有点，**保证了全局最优的可达性**。

**性质2 (收敛趋势)**：螺旋收缩算子的翻转概率 p_flip = spiral_rate * (1 - t/T) 随时间递减。这意味着：
- 早期 (t << T)：p_flip ~ spiral_rate，种群保持高多样性，广泛探索
- 后期 (t -> T)：p_flip -> 0，种群收敛到当前最优，精细利用

**性质3 (精英保证)**：精英保留机制确保最优解的适应度序列单调不减：
```
Fitness(A_best(t+1)) >= Fitness(A_best(t))  for all t
```

### 2.2 与Lottery Ticket Hypothesis的类比

**BWO可视为在训练过程中动态搜索"中奖彩票"**（即最优稀疏子网络）。

| 特性 | LTH (Frankle & Carbin) | BWO (本文) |
|------|----------------------|-----------|
| 搜索方式 | 迭代剪枝-重置-重训 | 种群进化并行搜索 |
| 稀疏度控制 | 固定剪枝比例 | 硬约束区间 [65%, 75%] |
| 多样性维持 | 低（单一网络） | 高（P个个体同时探索） |
| 计算开销 | O(k x full_training) | O(P x eval_samples) |
| 动态适应 | 无 | 随训练进展调整结构 |

**关键区别**：BWO在整个训练过程中维持种群的多样性，从而避免了过早收敛到次优的稀疏结构。DA（多巴胺）信号作为适应度函数的调制因子，使搜索优先关注当前训练阶段最重要的连接。

## 3. 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|----------|----------|
| 种群初始化 | O(P * S^2) | O(P * S^2) |
| 适应度评估 | O(P * eval_samples) | O(S^2) |
| 螺旋收缩 | O(P * S^2) | O(P * S^2) |
| 鲸落突变 | O(n_fall * S^2) | O(S^2) |
| 硬约束投影 | O(P * S^2) | O(S^2) |
| **总计每代** | **O(P * (S^2 + eval_samples))** | **O(P * S^2)** |

在典型配置下 (P=20, S=196, eval_samples=2000)：
- 时间 ~ 20 * (196^2 + 2000) = 20 * 40,416 = 808K 操作/代
- 实际运行时间 < 1分钟/代（GPU）

## 4. 生物学灵感说明

**"Wave"的含义**：BWO中的"Wave"指代种群在搜索空间中的**收缩-扩张动力学**：
- **收缩阶段**：螺旋算子使种群向当前最优解收敛（类似鲸群包围猎物）
- **扩张阶段**：鲸落突变引入随机扰动，防止过早收敛（类似鲸鱼死亡后滋养新生态系统）

这与脑电波（EEG wave）是不同层面的概念，不应混淆。BWO的生物学灵感主要来自**座头鲸的bubble-net捕食行为**和**海洋生态系统的营养循环**。
