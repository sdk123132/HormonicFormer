#!/bin/bash
# HormonicFormer v3 - DCU一键部署脚本
# 在SSH连接后执行: bash setup.sh

set -e

echo "=========================================="
echo "  HormonicFormer v3 - DCU部署"
echo "=========================================="

# 工作目录
WORK_DIR="/work/home/root/hormonic"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo ""
echo "[1/6] 检查环境..."
echo "  Python: $(python3 --version 2>/dev/null || echo '未找到')"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未找到')"
echo "  GPU: $(rocm-smi --showproductname 2>/dev/null | head -1 || echo '检查中...')"

echo ""
echo "[2/6] 配置华为镜像..."
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'EOF'
[global]
index-url = https://repo.huaweicloud.com/repository/pypi/simple
trusted-host = repo.huaweicloud.com
EOF
export HF_ENDPOINT=https://hf-mirror.com

echo ""
echo "[3/6] 安装依赖..."
pip install -q torch torchvision torchaudio --index-url https://repo.huaweicloud.com/repository/pypi/simple 2>/dev/null || echo "PyTorch已安装"
pip install -q pyyaml tqdm numpy --index-url https://repo.huaweicloud.com/repository/pypi/simple

echo ""
echo "[4/6] 等待文件上传..."
echo "  请在本地执行:"
echo "  scp -P 50068 -r hormonic_v3 root@ssh.dzai.scnet.cn:$WORK_DIR/"
echo ""
read -p "文件上传完成后按Enter继续..."

echo ""
echo "[5/6] 检查文件..."
if [ -f "config.yaml" ]; then
    echo "  ✓ 文件已上传"
    ls -la
else
    echo "  ✗ 文件未找到，请检查上传路径"
    exit 1
fi

echo ""
echo "[6/6] 运行测试..."
bash test.sh

echo ""
echo "=========================================="
echo "  部署完成!"
echo "=========================================="
echo ""
echo "启动训练:"
echo "  bash launch.sh"
echo ""
echo "监控训练:"
echo "  tail -f train_*.log"
echo ""
