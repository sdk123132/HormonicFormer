# HormonicFormer: A Field-Theoretic Neural Framework (Abandoned)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Abandoned](https://img.shields.io/badge/Status-Abandoned-red.svg)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Status: 🚫 ABANDONED**— This project explored a novel neural architecture derived from physics but failed to achieve competitive performance on standard benchmarks. Published here for transparency and as a learning resource.

**🔥 为什么这个失败项目值得关注？**
- 6个可证明的数学定理，全部验证通过
- 从第一性原理（场论）推导神经网络架构
- 透明的失败分析 — 比成功的 PR 更稀缺
- 证明了"优美的数学 ≠ 实用的工程"

---

## What Is This?

**HormonicFormer** is an experimental neural sequence model derived from the **Complex Ginzburg-Landau (CGL) field equation** — a reaction-diffusion PDE from non-equilibrium physics. Instead of empirically designing attention mechanisms, we attempted to derive a neural architecture from first principles of field theory.

The core idea: replace self-attention with **oscillatory field dynamics**, where:
- **Phase** encodes positional/relational information
- **Amplitude** encodes feature magnitude  
- **Diffusion (∇²)** provides O(S) local connectivity
- **Hebbian plasticity** provides adaptive long-range connectivity
- **Neuromodulation** provides global state-dependent modulation

## Why It Failed

Despite proving 6 theoretical properties and achieving elegant mathematical results, the architecture **could not compete on real-world tasks**:

| Task | HormonicFormer | Baseline | Verdict |
|------|---------------|----------|---------|
| WikiText-103 LM | PPL 805 | GPT-2: PPL 18 | ❌ 45× worse |
| CIFAR-10 | 35.08% | ResNet: 95%+ | ❌ Unusable |
| Adding Problem (LRA) | ~50% | Transformer: 99%+ | ❌ Random chance |
| Copy Task (S=64) | 100% ✓ | Transformer: 100% | ⚠️ Parity, 3× slower |
| Char LM (tiny) | PPL 5.68 | — | ✓ Works on toy scale |

**Root causes of failure:**
1. **Architecture-task mismatch** — CGL dynamics suit oscillatory/continuous data, not discrete token prediction
2. **Hebbian update too slow** — η=10⁻³ can't keep up with gradient descent
3. **Hyperparameter explosion** — CGL (α,β,D) × Hebbian (θ_c, η+, η-) × Neuromod (da, cb) = nightmare to tune
4. **O(S²) Hebbian matrix** — The G matrix that was supposed to replace attention has the same memory cost
5. **Complexity overhead** — 1.5× faster than Transformer in theory, but 3× slower in practice due to implementation overhead

## What Worked (Theoretically)

Six provable properties, all validated via SymPy + numerical simulation:

1. **Theorem 1**: CGL limit-cycle attractor exists with capacity controlled by r* = √(α/β)
2. **Theorem 2**: Hebbian G matrix converges spectrally to data covariance
3. **Theorem 3**: Phase transition in Hebbian learning (regularization ↔ overfitting)
4. **Theorem 4**: CGL steady-state ≡ temperature-scaled softmax attention
5. **Theorem 5**: Rotation invariance of learned representations
6. **Theorem 6**: Approximate identifiability under parameter perturbation

The theory is beautiful. The practice is not.

## Project Structure

```
├── 论文草稿/
│   ├── HormonicFormer_论文草稿.md          # Full paper draft (English)
│   ├── hormonic_v7r3_complete.txt          # Complete theory document
│   ├── theory_v2_content.txt               # Theoretical framework v2
│   └── 方案B_HormonicFormer_论文规划.txt    # Paper plan (Chinese)
│
├── 第一版/                                  # Phase 1: Initial implementation
│   ├── hormonic_v6_fix/                    # v6 codebase (DCU cluster)
│   │   ├── models/hormonicformer_v3.py     # Core model (40KB)
│   │   ├── field/laplacian_dft.py          # DFT-based Laplacian
│   │   └── scripts/train.py               # Training script
│   │
│   ├── 超算运行产出/文本/                    # HPC outputs (logs & reports)
│   └── 超算运行产出/视觉/                    # Visual experiments (CIFAR-10)
│
├── 第二版/                                  # Phase 2: CIFAR-10 & MLM attempts
│   ├── run_phase0.py                       # Sanity check experiments
│   ├── run_phase1_cifar10.py               # CIFAR-10 training (failed)
│   ├── run_phase2_mlm.py                   # Masked LM (marginal)
│   └── data/                               # CIFAR-10, WikiText-2
│
├── 研究论文数据/                             # Phase 3: Theorem validation
│   ├── theorem1_alpha_sweep.py             # Limit cycle validation
│   ├── theorem2_spectral_alignment.py      # Hebbian convergence
│   ├── theorem3_spectral.py                # Phase transition
│   ├── theorem4_softmax_validation.py      # Attention equivalence
│   ├── theorem5c_rotation_invariance.py    # Rotation invariance
│   ├── theorem6_sensitivity_analysis.py    # Identifiability
│   ├── experiment1_wikitext103.py          # WikiText-103 (failed)
│   ├── adding_problem.py                   # LRA Adding Problem (failed)
│   └── *.json                              # All experimental results
│
└── sympy_output.txt                        # Symbolic verification results
```

## Key Experimental Results

### Ablation Study (Copy Task, S=64, 3 epochs)

| Component Removed | Accuracy | Δ vs Full | Conclusion |
|-------------------|----------|-----------|------------|
| Full Model | 61.1% | baseline | — |
| **No CGL Reaction (dt=0)** | **25.9%** | **-35.2%** | **Critical component** |
| No Diffusion | 62.2% | +1.1% | Dispensable |
| No Hebbian | 75.2% | +14.1% | Harmful in short training |
| No E/I Balance | 71.3% | +10.2% | Dispensable |
| No Cross-Frequency | 78.6% | +17.5% | Dispensable |

**Key finding:** Only the CGL reaction term (α·ψ − β|ψ|²ψ) is essential. All bio-inspired add-ons (Hebbian, neuromodulation, STP) actually hurt performance on standard tasks.

### Hebbian Warmup Discovery

The one genuinely interesting finding: a "Hebbian warmup" schedule (full → decay → off) improves Char LM by 65% PPL reduction compared to always-on Hebbian:

| Strategy | Final PPL |
|----------|-----------|
| No Hebbian | 5.68 |
| Always-on Hebbian | 6.23 |
| **Hebbian Warmup** | **3.42** |

This mirrors the "critical period" in neurodevelopment — plasticity helps early but must be reduced later for stability.

### Complexity Scaling

| Model | S=128 | S=256 | S=512 | S=1024 | Exponent |
|-------|-------|-------|-------|--------|----------|
| HormonicFormer | 14.4ms | 29.1ms | 61.4ms | 135.8ms | p=1.08 |
| Transformer++ | 21.6ms | 43.3ms | 91.6ms | 197.7ms | p=1.07 |
| **Speedup** | **1.50×** | **1.49×** | **1.49×** | **1.46×** | — |

The O(S) diffusion step does give a real 1.5× throughput advantage — but this is meaningless when the model produces garbage outputs.

## Lessons Learned

1. **Beautiful math ≠ good engineering.** All 6 theorems are correct. The model still sucks at language.

2. **Beware the toy task trap.** 100% on Copy Task means nothing when WikiText-103 PPL is 805.

3. **Bio-inspiration is seductive but dangerous.** Hebbian learning, neuromodulation, STP — each sounds compelling in isolation. Together they create an untunable mess.

4. **Complexity budget matters.** If your "O(S) alternative to attention" needs an S² Hebbian matrix anyway, you've gained nothing.

5. **Don't mistake theoretical beauty for practical utility.** The CGL ↔ softmax equivalence (Theorem 4) is mathematically elegant, but it only holds at steady state — a condition never reached in real sequence processing.

6. **Know when to stop.** This project consumed ~3 weeks of full-time work across two phases (HPC cluster + local GPU). The sunk cost fallacy is real.

## Requirements

```
torch >= 2.0
numpy
scipy (for SymPy verification)
sympy (for theorem verification)
```

## 🚀 如何让更多人看到这个项目

### 分享渠道

1. **Reddit**
   - r/MachineLearning: [分享失败的研究经验]
   - r/LocalLLaMA: [另类架构探索]
   - r/CompSci: [从物理第一原理设计神经网络]

2. **Twitter/X**
   - 标签: #NeuralArchitecture #FailedExperiments #FieldTheory #DeepLearning
   - 示例推文: "6个定理全部验证，但模型PPL 805 vs GPT-2的18。这就是HormonicFormer——一个从CGL场方程推导出的失败神经网络。"

3. **Hacker News**
   - 标题: "Show HN: A neural architecture derived from physics that failed spectacularly"
   - 重点: 透明的失败分析比成功的 PR 更稀缺

4. **中文社区**
   - 知乎: 如何优雅地失败——一个物理启发的神经网络项目复盘
   - 小红书: 科研失败案例分享
   - V2EX: 独立研究项目分享

5. **学术社区**
   - Papers With Code: 添加为"失败案例"
   - GitHub Awesome Lists: 提议添加到 awesome-failure 或类似列表

### 引用格式

```bibtex
@misc{hormonicformer2026,
  title={HormonicFormer: A Field-Theoretic Framework for Neural Sequence Modeling (Abandoned)},
  author={[Author]},
  year={2026},
  note={Unpublished. Theoretical results valid; empirical performance insufficient for publication.},
  howpublished={\url{https://github.com/sdk123132/HormonicFormer}}
}
```

### 相关讨论

- **失败的价值**: [The Importance of Negative Results in Science](https://www.nature.com/articles/s41562-019-0569-3)
- **物理启发AI**: [Physics-informed neural networks](https://en.wikipedia.org/wiki/Physics-informed_neural_networks)
- **CGL方程**: [Complex Ginzburg-Landau equation](https://en.wikipedia.org/wiki/Complex_Ginzburg%E2%80%93Landau_equation)

## License

MIT — Use freely, but manage your expectations.

---

*"Not all who wander are lost, but this one definitely got lost." — The Author, after seeing PPL 805*
