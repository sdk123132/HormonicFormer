#!/bin/bash
# DCU状态检查脚本

echo "========================================"
echo "DCU训练状态检查"
echo "时间: $(date)"
echo "========================================"
echo

echo "[1] Python进程:"
ps aux | grep python | grep -v grep | head -5
echo

echo "[2] 训练日志 (最后20行):"
if [ -f /root/private_data/hormonic_v3/runs_2_3_final.log ]; then
    tail -20 /root/private_data/hormonic_v3/runs_2_3_final.log
else
    echo "日志文件不存在"
fi
echo

echo "[3] GPU状态:"
dcu-smi 2>/dev/null || echo "dcu-smi不可用"
echo

echo "[4] 输出文件:"
ls -lh /root/private_data/hormonic_v3/*.pt 2>/dev/null | tail -5
ls -lh /root/private_data/hormonic_v3/*.json 2>/dev/null | tail -5
echo

echo "[5] 磁盘空间:"
df -h /root
echo

echo "========================================"
echo "检查完成"
echo "========================================"
