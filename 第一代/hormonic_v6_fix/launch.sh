#!/bin/bash
# HormonicFormer v6.1 - DCU启动脚本
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  HormonicFormer v6.1 - DCU Launcher${NC}"
echo -e "${GREEN}  Fixes: Laplacian + CFL + DA + STP${NC}"
echo -e "${GREEN}========================================${NC}"

# [1] 环境检查
echo -e "\n${YELLOW}[1/5] 检查环境...${NC}"
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

if command -v rocm-smi &> /dev/null; then
    echo -e "${GREEN}DCU detected:${NC}"
    rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null || true
else
    echo -e "${YELLOW}Warning: rocm-smi not found (CUDA mode)${NC}"
fi

NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo -e "${GREEN}GPUs available: $NGPU${NC}"

# [2] 工作目录
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"
echo -e "\n${YELLOW}[2/5] Working dir: $WORKDIR${NC}"

# [3] 必要目录
mkdir -p checkpoints logs data
echo -e "\n${YELLOW}[3/5] Directories ready${NC}"

# [4] v6.1验证: Laplacian核尺度 + DFT功能
echo -e "\n${YELLOW}[4/5] v6.1 Validation...${NC}"
python3 -c "
import sys, torch, math
sys.path.insert(0, 'field')
from laplacian_dft import DFTLaplacian

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lap = DFTLaplacian(196).to(device)

# 验证1: Laplacian核尺度正确
max_eig = lap.lap_kernel.abs().max().item()
assert max_eig < 20, f'Laplacian scale WRONG: |λ_max|={max_eig} (should be ~π²≈9.87)'
print(f'  [PASS] Laplacian scale: |λ_max|={max_eig:.4f} (expected ~{math.pi**2:.4f})')

# 验证2: CFL条件满足
dt, D0 = 0.02, 0.002
cfl = dt * D0 * max_eig
assert cfl < 2.0, f'CFL violated: {cfl}'
print(f'  [PASS] CFL condition: {cfl:.6f} < 2.0')

# 验证3: DFT功能正常
x = torch.randn(2, 196, 32, device=device)
y = lap(x)
assert y.shape == x.shape
print(f'  [PASS] DFT Laplacian (vectorized): OK')
"

# [5] 训练
LOGFILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

if [ "$NGPU" -gt 1 ]; then
    echo -e "\n${YELLOW}[5/5] 启动DDP训练...${NC}"
    torchrun --standalone --nnodes=1 --nproc_per_node=$NGPU \
        scripts/train.py --config config.yaml 2>&1 | tee "$LOGFILE"
else
    echo -e "\n${YELLOW}[5/5] 启动单卡训练...${NC}"
    python3 scripts/train.py --config config.yaml 2>&1 | tee "$LOGFILE"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Training Complete!${NC}"
echo -e "${GREEN}  Log: $LOGFILE${NC}"
echo -e "${GREEN}========================================${NC}"
