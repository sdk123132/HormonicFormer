#!/bin/bash
# HormonicFormer v6.1 - 验证测试脚本
# 覆盖v6.1所有修复点

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "  HormonicFormer v6.1 - Verification"
echo "=========================================="

PASS=0
FAIL=0

run_test() {
    local name="$1"
    local code="$2"
    echo -e "\n[$((PASS+FAIL+1))] $name..."
    if python3 -c "$code" 2>&1; then
        echo "  PASS"
        ((PASS++))
    else
        echo "  FAIL"
        ((FAIL++))
    fi
}

# ── v6.1 新增验证 ──

# [1] Laplacian核尺度验证
run_test "Laplacian Scale Fix (v6.1)" '
import sys, torch, math
sys.path.insert(0, "field")
from laplacian_dft import DFTLaplacian
lap = DFTLaplacian(196)
max_eig = lap.lap_kernel.abs().max().item()
assert max_eig < 20, f"Scale wrong: {max_eig}"
assert abs(max_eig - math.pi**2) < 1.0, f"Not ~pi^2: {max_eig}"
print(f"  |λ_max|={max_eig:.4f} (expected ~{math.pi**2:.4f}) OK")
'

# [2] CFL稳定性验证
run_test "CFL Stability (v6.1)" '
import sys, torch, math
sys.path.insert(0, "field")
from laplacian_dft import DFTLaplacian
lap = DFTLaplacian(196)
dt, D0 = 0.02, 0.002
cfl = dt * D0 * lap.lap_kernel.abs().max().item()
assert cfl < 2.0, f"CFL violated: {cfl}"
print(f"  CFL={cfl:.6f} < 2.0 OK")
'

# [3] STP重置功能验证
run_test "STP Reset (v6.1)" '
import sys, torch, yaml
sys.path.insert(0, "models")
from hormonicformer_v3 import ShortTermPlasticity
with open("config.yaml") as f: config = yaml.safe_load(f)
stp = ShortTermPlasticity(16, 64, config)
# 耗竭资源
for _ in range(50):
    stp.step(torch.ones(4, 16) * 5.0)
eff_before = stp.get_efficacy().mean().item()
assert eff_before < 0.5, f"STP not depleted: {eff_before}"
# 重置
stp.reset()
eff_after = stp.get_efficacy().mean().item()
assert abs(eff_after - 0.2) < 0.01, f"STP reset failed: {eff_after}"
print(f"  Before reset: eff={eff_before:.3f}")
print(f"  After reset:  eff={eff_after:.3f} OK")
'

# [4] Neuromodulator.reset_all()验证
run_test "Neuromodulator Reset All (v6.1)" '
import sys, torch, yaml
sys.path.insert(0, "models")
from hormonicformer_v3 import Neuromodulator
with open("config.yaml") as f: config = yaml.safe_load(f)
nm = Neuromodulator(16, 64, config)
# 改变状态
for _ in range(20):
    nm.stp.step(torch.ones(4, 16) * 3.0)
    nm.homeostatic.update(torch.ones(4, 16) * 3.0)
u_before = nm.stp.u.mean().item()
gain_before = nm.homeostatic.gain.mean().item()
# 重置
nm.reset_all()
u_after = nm.stp.u.mean().item()
gain_after = nm.homeostatic.gain.mean().item()
assert abs(u_after - 0.2) < 0.01, f"STP u not reset"
assert abs(gain_after - 1.0) < 0.01, f"Homeo gain not reset"
print(f"  STP u: {u_before:.3f} -> {u_after:.3f} OK")
print(f"  Gain:  {gain_before:.3f} -> {gain_after:.3f} OK")
'

# [5] G掩码缓存验证
run_test "G Mask Caching (v6.1)" '
import sys, torch, yaml
sys.path.insert(0, "models")
from hormonicformer_v3 import Neuromodulator
with open("config.yaml") as f: config = yaml.safe_load(f)
nm = Neuromodulator(16, 64, config)
# 首次调用应计算
m1 = nm.get_G_mask()
assert not nm._G_mask_dirty, "Cache not marked clean"
# 再次调用应返回缓存
m2 = nm.get_G_mask()
assert torch.allclose(m1, m2), "Cache returned different values"
# prune后应重新计算
nm.prune_G(prune_ratio=0.1)
assert nm._G_mask_dirty, "Cache not marked dirty after prune"
m3 = nm.get_G_mask()
print(f"  Cache: hit OK, invalidate OK, recompute OK")
'

# ── v6 原有功能验证 ──

# [6] 树突计算
run_test "Dendritic Computation" '
import sys, torch
sys.path.insert(0, "models")
from hormonicformer_v3 import DendriticCompartment
dend = DendriticCompartment(64)
x_ff = torch.randn(2, 196, 64)
x_td = torch.randn(2, 196, 64)
out_ff = dend(x_ff, x_td=None)
out_td = dend(x_ff, x_td=x_td)
assert out_ff.shape == x_ff.shape
assert out_td.shape == x_ff.shape
assert (out_ff - out_td).abs().mean() > 0
diff = (out_ff - out_td).abs().mean().item()
print(f"  FF vs TD diff: {diff:.4f} OK")
'

# [7] 稳态可塑性
run_test "Homeostatic Plasticity" '
import sys, torch, yaml
sys.path.insert(0, "models")
from hormonicformer_v3 import HomeostaticPlasticity
with open("config.yaml") as f: config = yaml.safe_load(f)
hp = HomeostaticPlasticity(16, config)
for _ in range(100): hp.update(torch.ones(4, 16) * 5.0)
gain_high = hp.gain.mean().item()
hp2 = HomeostaticPlasticity(16, config)
for _ in range(100): hp2.update(torch.ones(4, 16) * 0.01)
gain_low = hp2.gain.mean().item()
assert gain_high < 1.0 and gain_low > 1.0
print(f"  High activity gain={gain_high:.3f} <1 OK")
print(f"  Low activity  gain={gain_low:.3f} >1 OK")
'

# [8] 完整前向
run_test "Full Model Forward" '
import sys, torch, yaml
sys.path.insert(0, "models")
from hormonicformer_v3 import HormonicFormer
with open("config.yaml") as f: config = yaml.safe_load(f)
config["model"]["n_layers"] = 2
config["model"]["d_model"] = 64
config["model"]["n_heads"] = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HormonicFormer(config).to(device)
dummy = torch.randn(2, 1, 28, 28, device=device)
with torch.no_grad():
    logits = model(dummy)
assert logits.shape == (2, 10)
print(f"  Output: {logits.shape} OK")
'

# [9] 训练步骤
run_test "Training Step" '
import sys, torch, yaml
sys.path.insert(0, "models")
from hormonicformer_v3 import HormonicFormer
with open("config.yaml") as f: config = yaml.safe_load(f)
config["model"]["n_layers"] = 2
config["model"]["d_model"] = 64
config["model"]["n_heads"] = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HormonicFormer(config).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
images = torch.randn(4, 1, 28, 28, device=device)
targets = torch.randint(0, 10, (4,), device=device)
logits, loss = model(images, targets)
loss.backward()
opt.step()
assert not torch.isnan(loss)
print(f"  Loss: {loss.item():.4f} OK")
'

# [10] 诊断接口
run_test "Diagnostics Interface" '
import sys, torch, yaml
sys.path.insert(0, "models")
from hormonicformer_v3 import HormonicFormer
with open("config.yaml") as f: config = yaml.safe_load(f)
config["model"]["n_layers"] = 2
config["model"]["d_model"] = 64
config["model"]["n_heads"] = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HormonicFormer(config).to(device)
model.train()
images = torch.randn(4, 1, 28, 28, device=device)
targets = torch.randint(0, 10, (4,), device=device)
_, loss = model(images, targets)
diag = model.get_diagnostics()
required = ["DA", "G_sparsity", "stp_u_mean", "stp_r_mean",
            "stp_efficacy_mean", "homeo_gain_mean", "homeo_gain_std",
            "homeo_activity_mean", "energy_mean", "alive_ratio"]
for k in required:
    assert k in diag, f"Missing key: {k}"
print("  All diagnostic keys present OK")
print(f"  DA={diag['DA']:.3f} (should be ~0.5 with da_init=2.5)")
'

echo -e "\n=========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "=========================================="

if [ $FAIL -gt 0 ]; then
    exit 1
fi

echo ""
echo "All v6.1 verifications PASSED!"
echo "Ready to train: bash launch.sh"
