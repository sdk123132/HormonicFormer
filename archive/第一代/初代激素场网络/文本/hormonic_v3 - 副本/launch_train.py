#!/usr/bin/env python3
"""训练启动脚本 - 处理环境变量和编码问题"""
import os
import sys

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加路径
sys.path.insert(0, str(os.path.dirname(__file__)))

# 运行主训练脚本
with open('monitored_train.py', 'r', encoding='utf-8') as f:
    exec(f.read())