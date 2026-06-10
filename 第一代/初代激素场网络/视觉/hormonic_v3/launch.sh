#!/bin/bash
# HormonicFormer v3 - DCU启动脚本
# 适配国家算力网

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  HormonicFormer v3 - DCU Launcher${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查环境
echo -e "\n${YELLOW}[1/5] 检查环境...${NC}"

# Python版本
python3 --version

# PyTorch版本
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# DCU检测
if command -v rocm-smi &> /dev/null; then
    echo -e "${GREEN}DCU detected:${NC}"
    rocm-smi --showproductname --showmeminfo vram --showuse
else
    echo -e "${YELLOW}Warning: rocm-smi not found${NC}"
fi

# GPU数量
NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo -e "${GREEN}GPUs available: $NGPU${NC}"

# 工作目录
WORKDIR="/work/home/$(whoami)/hormonic"
mkdir -p $WORKDIR
cd $WORKDIR

echo -e "\n${YELLOW}[2/5] 准备数据...${NC}"
# Fashion-MNIST会自动下载

# 检查代码
echo -e "\n${YELLOW}[3/5] 检查代码...${NC}"
if [ ! -f "models/hormonicformer_v3.py" ]; then
    echo -e "${RED}Error: Model file not found!${NC}"
    echo "Please upload hormonic_v3/ to $WORKDIR"
    exit 1
fi

# 测试DFT Laplacian
echo -e "\n${YELLOW}[4/5] 测试DFT Laplacian...${NC}"
python3 -c "
import torch
import sys
sys.path.insert(0, 'field')
from laplacian_dft import DFTLaplacian

print('Testing DFT Laplacian...')
lap = DFTLaplacian(196, 'cuda')
x = torch.randn(2, 196).cuda()
y = lap(x)
print(f'Input shape: {x.shape}')
print(f'Output shape: {y.shape}')
print(f'Output range: [{y.min():.3f}, {y.max():.3f}]')
print('DFT Laplacian test PASSED!')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}DFT Laplacian test FAILED!${NC}"
    exit 1
fi

# 运行训练
echo -e "\n${YELLOW}[5/5] 启动训练...${NC}"

if [ $NGPU -gt 1 ]; then
    echo -e "${GREEN}Using DDP with $NGPU GPUs${NC}"
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NGPU \
        scripts/train.py \
        --config config.yaml \
        2>&1 | tee train_$(date +%Y%m%d_%H%M%S).log
else
    echo -e "${GREEN}Using single GPU${NC}"
    python3 scripts/train.py \
        --config config.yaml \
        2>&1 | tee train_$(date +%Y%m%d_%H%M%S).log
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
