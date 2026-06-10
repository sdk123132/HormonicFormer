#!/usr/bin/env python3
"""检查DCU训练状态"""

import subprocess
import os
from datetime import datetime

print("="*60)
print(f"DCU状态检查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# 检查进程
print("\n[1] 检查Python进程:")
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
python_processes = [line for line in result.stdout.split('\n') if 'python' in line.lower()]
for p in python_processes[:5]:
    print(f"  {p}")

# 检查日志文件
print("\n[2] 检查日志文件:")
log_files = [
    '/root/private_data/hormonic_v3/runs_2_3_final.log',
    '/root/private_data/hormonic_v3/run_2.log',
    '/root/private_data/hormonic_v3/run_3.log',
]

for log_file in log_files:
    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        print(f"  {log_file}: {size/1024:.1f} KB")
        
        # 显示最后10行
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"    最后几行:")
                for line in lines[-5:]:
                    print(f"      {line.strip()}")
    else:
        print(f"  {log_file}: 不存在")

# 检查GPU
print("\n[3] GPU状态:")
try:
    result = subprocess.run(['dcu-smi'], capture_output=True, text=True, timeout=10)
    print(result.stdout[:500] if result.stdout else "dcu-smi 无输出")
except:
    print("  无法运行 dcu-smi")

# 检查输出文件
print("\n[4] 检查输出文件:")
output_dirs = [
    '/root/private_data/hormonic_v3/',
    '/root/private_data/hormonic_v3/runs/',
]

for d in output_dirs:
    if os.path.exists(d):
        files = os.listdir(d)
        pt_files = [f for f in files if f.endswith('.pt')]
        json_files = [f for f in files if f.endswith('.json')]
        print(f"  {d}:")
        print(f"    .pt文件: {len(pt_files)}")
        print(f"    .json文件: {len(json_files)}")
        if pt_files:
            print(f"    模型: {pt_files[-3:]}")
        if json_files:
            print(f"    结果: {json_files[-3:]}")

print("\n" + "="*60)
print("检查完成")
print("="*60)
