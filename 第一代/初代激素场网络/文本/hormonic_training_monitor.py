#!/usr/bin/env python3
"""
HormonicFormer 训练监控脚本
每轮结束后自动保存 epoch 报告到桌面
"""
import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path

# 配置
LOG_FILE = r"C:\Users\MR\Desktop\Kimi_Agent_模型评估\hormonic_v3\logs\local_train_20260525_1400.log"
OUTPUT_DIR = r"C:\Users\MR\Desktop\训练报告"
CHECK_INTERVAL = 60  # 每60秒检查一次

def extract_epoch_summary(lines):
    """从日志中提取 epoch 结束时的汇总信息
    
    匹配格式:
      9 | 1.2345     | 45.6%    | 1.1876   | 43.2% | 45.6%| 123s | L0:sparsity=23.4%
    """
    summaries = []
    
    # 查找 epoch 汇总表格行
    # 格式: epoch | train_loss | train_acc | val_loss | val_acc | best_acc | time | sparsity_info
    epoch_pattern = r'^\s*(\d+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)%\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)%\s+\|\s+([\d.]+)%\s+\|\s+(\d+)s'
    
    for line in lines:
        match = re.search(epoch_pattern, line)
        if match:
            epoch_num = match.group(1)
            train_loss = match.group(2)
            train_acc = match.group(3)
            val_loss = match.group(4)
            val_acc = match.group(5)
            best_acc = match.group(6)
            elapsed = match.group(7)
            summaries.append({
                'epoch': epoch_num,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_acc': best_acc,
                'elapsed': elapsed,
                'raw_line': line.strip()
            })
    
    return summaries

def extract_da_info(lines):
    """提取 DA/CB/G_sparsity 等信息"""
    da_pattern = r'DA:\s+([\d.]+).*?CB:\s+([\d.]+).*?G_sparsity:\s+([\d.]+%)'
    
    for line in reversed(lines):
        match = re.search(da_pattern, line)
        if match:
            return {
                'DA': match.group(1),
                'CB': match.group(2),
                'G_sparsity': match.group(3)
            }
    return None

def save_epoch_report(epoch_data, output_dir):
    """保存 epoch 报告到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"epoch_{epoch_data['epoch']}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    content = f"""================================================================================
HormonicFormer 训练报告 - Epoch {epoch_data['epoch']}
================================================================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

训练指标:
  Train Loss: {epoch_data['train_loss']}
  Train Acc:  {epoch_data['train_acc']}%

验证指标:
  Val Loss:   {epoch_data['val_loss']}
  Val Acc:    {epoch_data['val_acc']}%

神经调质状态:
"""
    
    if 'DA' in epoch_data:
        content += f"""  DA (多巴胺):     {epoch_data['DA']}
  CB (大麻素):     {epoch_data['CB']}
  G_sparsity:      {epoch_data['G_sparsity']}
"""
    else:
        content += "  (暂无 DA/CB 数据)\n"
    
    content += f"""
================================================================================
原始日志:
{epoch_data.get('raw_line', 'N/A')}
================================================================================
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[监控] 已保存 Epoch {epoch_data['epoch']} 报告: {filepath}")
    return filepath

def monitor_training():
    """主监控循环"""
    print(f"[监控] 启动 HormonicFormer 训练监控")
    print(f"[监控] 日志文件: {LOG_FILE}")
    print(f"[监控] 输出目录: {OUTPUT_DIR}")
    print(f"[监控] 检查间隔: {CHECK_INTERVAL}秒")
    print("-" * 70)
    
    last_size = 0
    processed_epochs = set()
    
    try:
        while True:
            if os.path.exists(LOG_FILE):
                current_size = os.path.getsize(LOG_FILE)
                
                if current_size > last_size:
                    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    # 提取所有 epoch 汇总
                    summaries = extract_epoch_summary(lines)
                    
                    for summary in summaries:
                        epoch_num = summary['epoch']
                        
                        # 只处理新完成的 epoch
                        if epoch_num not in processed_epochs:
                            # 尝试提取 DA/CB 信息
                            da_info = extract_da_info(lines)
                            if da_info:
                                summary.update(da_info)
                            
                            # 保存报告
                            save_epoch_report(summary, OUTPUT_DIR)
                            processed_epochs.add(epoch_num)
                            
                            # 同时更新汇总文件
                            update_summary_file(summaries, OUTPUT_DIR)
                    
                    last_size = current_size
            
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n[监控] 已停止")

def update_summary_file(all_summaries, output_dir):
    """更新汇总文件"""
    summary_path = os.path.join(output_dir, "training_summary.txt")
    
    content = f"""================================================================================
HormonicFormer 训练汇总
================================================================================
更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
总 Epochs: {len(all_summaries)}

进度表:
Epoch | Train Loss | Train Acc | Val Loss  | Val Acc  | DA    | CB    | G_sparse
------|------------|-----------|-----------|----------|-------|-------|----------
"""
    
    for s in all_summaries:
        da = s.get('DA', '-')
        cb = s.get('CB', '-')
        g = s.get('G_sparsity', '-')
        content += f"{s['epoch']:>5} | {s['train_loss']:>10} | {s['train_acc']:>9}% | {s['val_loss']:>9} | {s['val_acc']:>8}% | {da:>5} | {cb:>5} | {g:>8}\n"
    
    content += "=" * 80 + "\n"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    monitor_training()
