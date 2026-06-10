"""
训练监控器 - 实时监控HormonicFormer训练数据
支持: TensorBoard、WebSocket实时推送、CSV日志
"""
import os
import time
import json
import csv
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


class TrainingMonitor:
    """训练监控器主类"""
    
    def __init__(self, config, log_dir=None):
        self.config = config
        self.log_dir = Path(log_dir or config['train']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 监控数据缓存
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'da': [],
            'cb': [],
            'gpu_memory': [],
            'epoch_times': [],
            'hebbian_stats': []
        }
        
        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # CSV日志文件
        self.csv_file = open(self.log_dir / 'training_metrics.csv', 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'epoch', 'timestamp', 'train_loss', 'train_acc', 'val_loss', 
            'val_acc', 'lr', 'da', 'cb', 'gpu_memory_mb', 'epoch_time_s'
        ])
        
        # 最近一次数据（用于实时推送）
        self.last_data = {}
        
        # 训练开始时间
        self.start_time = time.time()
        
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, 
                  lr=0.0, da=0.5, cb=0.0, gpu_memory=0, epoch_time=0.0):
        """记录一个epoch的训练数据"""
        # 添加到缓存
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['lr'].append(lr)
        self.metrics['da'].append(da)
        self.metrics['cb'].append(cb)
        self.metrics['gpu_memory'].append(gpu_memory)
        self.metrics['epoch_times'].append(epoch_time)
        
        # 写入TensorBoard
        self.tb_writer.add_scalar('Loss/train', train_loss, epoch)
        self.tb_writer.add_scalar('Loss/val', val_loss, epoch)
        self.tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
        self.tb_writer.add_scalar('LearningRate', lr, epoch)
        self.tb_writer.add_scalar('Neuromodulator/DA', da, epoch)
        self.tb_writer.add_scalar('Neuromodulator/CB', cb, epoch)
        self.tb_writer.add_scalar('System/GPU_Memory_MB', gpu_memory, epoch)
        self.tb_writer.add_scalar('System/Epoch_Time_S', epoch_time, epoch)
        
        # 写入CSV
        self.csv_writer.writerow([
            epoch, datetime.now().isoformat(),
            train_loss, train_acc, val_loss, val_acc,
            lr, da, cb, gpu_memory, epoch_time
        ])
        self.csv_file.flush()
        
        # 更新最近数据
        self.last_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr,
            'da': da,
            'cb': cb,
            'gpu_memory': gpu_memory,
            'epoch_time': epoch_time,
            'elapsed_time': time.time() - self.start_time
        }
        
    def log_hebbian_stats(self, epoch, stats):
        """记录Hebbian统计信息"""
        self.metrics['hebbian_stats'].append({
            'epoch': epoch,
            'stats': stats
        })
        
        # 写入TensorBoard
        for stat in stats:
            layer = stat['layer']
            self.tb_writer.add_scalar(f'Hebbian/Layer{layer}_G_mean', stat['G_mean'], epoch)
            self.tb_writer.add_scalar(f'Hebbian/Layer{layer}_G_std', stat['G_std'], epoch)
            self.tb_writer.add_scalar(f'Hebbian/Layer{layer}_sparsity', stat['G_sparsity'], epoch)
            self.tb_writer.add_scalar(f'Hebbian/Layer{layer}_alive_ratio', stat['alive_ratio'], epoch)
        
        self.last_data['hebbian_stats'] = stats
        
    def log_batch(self, epoch, batch_idx, loss, acc, lr):
        """记录batch级别的数据"""
        global_step = (epoch - 1) * 1000 + batch_idx
        self.tb_writer.add_scalar('Loss/batch', loss, global_step)
        self.tb_writer.add_scalar('Accuracy/batch', acc, global_step)
        
    def get_recent_data(self):
        """获取最近的监控数据（用于实时推送）"""
        return self.last_data
    
    def get_all_metrics(self):
        """获取所有历史指标"""
        return self.metrics
    
    def get_summary(self):
        """获取训练摘要"""
        if not self.metrics['val_acc']:
            return None
            
        best_idx = int(torch.argmax(torch.tensor(self.metrics['val_acc'])).item())
        return {
            'total_epochs': len(self.metrics['train_loss']),
            'best_val_acc': self.metrics['val_acc'][best_idx],
            'best_epoch': best_idx + 1,
            'avg_epoch_time': sum(self.metrics['epoch_times']) / len(self.metrics['epoch_times']),
            'current_da': self.metrics['da'][-1] if self.metrics['da'] else 0.5,
            'current_lr': self.metrics['lr'][-1] if self.metrics['lr'] else 0.0
        }
        
    def close(self):
        """关闭监控器"""
        self.tb_writer.close()
        self.csv_file.close()


class GPUMonitor:
    """GPU显存监控器"""
    
    @staticmethod
    def get_memory_usage(device_id=0):
        """获取GPU显存使用情况"""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'total': 0}
            
        allocated = torch.cuda.memory_allocated(device_id) / 1024**2
        reserved = torch.cuda.memory_reserved(device_id) / 1024**2
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**2
        
        return {
            'allocated': int(allocated),
            'reserved': int(reserved),
            'total': int(total),
            'used_percent': (allocated / total) * 100
        }
    
    @staticmethod
    def log_memory_usage(logger=None):
        """打印显存使用日志"""
        if not torch.cuda.is_available():
            return 0
            
        mem = GPUMonitor.get_memory_usage()
        msg = f"[GPU] 已用: {mem['allocated']}MB / 预留: {mem['reserved']}MB / 总共: {mem['total']}MB ({mem['used_percent']:.1f}%)"
        
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        return mem['allocated']
