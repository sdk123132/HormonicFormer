"""
Ablation Study - Sequence Tasks
Test contribution of each HormonicFormer v3 component
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime

from hormonicformer_v3 import HormonicFormer


class CopyTaskDataset(Dataset):
    """Copy Task Dataset"""
    def __init__(self, seq_len, num_samples=10000):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.vocab_size = 10  # 数字 0-9
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = x.clone()
        return x, y


class CopyTaskWrapper(nn.Module):
    """Copy Task 包装器"""
    def __init__(self, config, vocab_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        config['model']['seq_len'] = seq_len
        config['model']['patch_size'] = 1
        config['model']['n_classes'] = vocab_size
        
        self.model = HormonicFormer(config)
        d_model = config['model']['d_model']
        self.model.classifier = nn.Linear(d_model, vocab_size)
    
    def forward(self, seq):
        B, S = seq.shape
        vals = (seq.float() / (self.vocab_size - 1)) * 2 - 1
        img = vals.view(B, 1, 1, S)
        
        x = self.model.patch_embed(img)
        x = x.flatten(2).transpose(1, 2)
        
        for block in self.model.blocks:
            x = block(x)
        
        logits = self.model.classifier(x)
        return logits


def get_base_config():
    """基础配置（所有组件开启）"""
    return {
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'seq_len': 128,
            'n_steps': 4,
            'n_classes': 10,
            'patch_size': 1,
            'D0_amp': 0.002,
            'D0_phase': 0.002,
            'dt': 0.02,
            'noise_scale': 0.0,
            'dropout': 0.1,
            'ei_balance': {'enabled': True, 'target_ratio': 4.0},
            'sensory_feedback': {'enabled': True, 'top_k': 8},
            'hebbian': {'enabled': True, 'lr': 0.001, 'decay': 0.99},
            'cross_freq_coupling': {'enabled': True},
            'energy_constraint': {'enabled': True}
        },
        'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': True},
        'bwo': {'use_bwo': False},
        'pc': {'use_pc': False}
    }


def run_experiment(name, config, seq_len=128, epochs=5):
    """运行单个消融实验"""
    print(f'\n{"="*60}')
    print(f'实验: {name}')
    print(f'{"="*60}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据集
    train_ds = CopyTaskDataset(seq_len, 8000)
    val_ds = CopyTaskDataset(seq_len, 2000)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # 模型
    model = CopyTaskWrapper(config, 10, seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    results = {'name': name, 'epochs': []}
    
    for epoch in range(epochs):
        # 训练
        model.train()
        total_correct, total = 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 10), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            pred = logits.argmax(dim=-1)
            total_correct += (pred == y).sum().item()
            total += y.numel()
        
        train_acc = total_correct / total
        
        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=-1)
                val_correct += (pred == y).sum().item()
                val_total += y.numel()
        
        val_acc = val_correct / val_total
        
        results['epochs'].append({
            'epoch': epoch + 1,
            'train_acc': round(train_acc * 100, 2),
            'val_acc': round(val_acc * 100, 2)
        })
        
        print(f'Epoch {epoch+1}: Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%')
    
    return results


def main():
    """运行所有消融实验"""
    print('HormonicFormer v3 - 消融实验（序列任务）')
    print(f'开始时间: {datetime.now()}')
    
    all_results = []
    
    # 1. 完整模型（基线）
    config = get_base_config()
    results = run_experiment('完整模型 (Full)', config, seq_len=128, epochs=5)
    all_results.append(results)
    
    # 2. 关闭扩散项
    config = get_base_config()
    config['model']['D0_amp'] = 0.0
    config['model']['D0_phase'] = 0.0
    results = run_experiment('无扩散项 (No Diffusion)', config, seq_len=128, epochs=5)
    all_results.append(results)
    
    # 3. 冻结反应（dt=0）
    config = get_base_config()
    config['model']['dt'] = 0.0
    results = run_experiment('冻结反应 (dt=0)', config, seq_len=128, epochs=5)
    all_results.append(results)
    
    # 4. 关闭 Hebbian
    config = get_base_config()
    config['model']['hebbian']['enabled'] = False
    results = run_experiment('无 Hebbian', config, seq_len=128, epochs=5)
    all_results.append(results)
    
    # 5. 关闭感觉反馈
    config = get_base_config()
    config['model']['sensory_feedback']['enabled'] = False
    results = run_experiment('无感觉反馈', config, seq_len=128, epochs=5)
    all_results.append(results)
    
    # 6. 关闭 E/I 平衡
    config = get_base_config()
    config['model']['ei_balance']['enabled'] = False
    results = run_experiment('无 E/I 平衡', config, seq_len=128, epochs=5)
    all_results.append(results)
    
    # 7. 关闭跨频耦合
    config = get_base_config()
    config['model']['cross_freq_coupling']['enabled'] = False
    results = run_experiment('无跨频耦合', config, seq_len=128, epochs=5)
    all_results.append(results)
    
    # 8. 关闭能量约束
    config = get_base_config()
    config['model']['energy_constraint']['enabled'] = False
    results = run_experiment('无能量约束', config, seq_len=128, epochs=5)
    all_results.append(results)
    
    # 保存结果
    output = {
        'timestamp': str(datetime.now()),
        'task': 'Copy Task S=128',
        'results': all_results
    }
    
    with open('ablation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f'\n{"="*60}')
    print('消融实验完成！')
    print(f'结果保存到: ablation_results.json')
    print(f'结束时间: {datetime.now()}')
    print(f'{"="*60}')
    
    # 打印汇总表
    print('\n消融实验汇总（Epoch 5 验证准确率）：')
    print('-'*60)
    for r in all_results:
        final_acc = r['epochs'][-1]['val_acc'] if r['epochs'] else 0
        print(f'{r["name"]:<25}: {final_acc:.1f}%')


if __name__ == '__main__':
    main()
