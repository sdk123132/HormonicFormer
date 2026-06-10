"""
Quick test for HormonicFormer v3 Copy Task
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from models.hormonicformer_v3 import HormonicFormer
from copy_task_dataset import CopyTaskDataset

# Load config
with open('local_v3_copytask_5070.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['model']['seq_len'] = 16
config['model']['n_classes'] = 11

print('Config loaded')
print(f"  d_model: {config['model']['d_model']}")
print(f"  seq_len: {config['model']['seq_len']}")
print(f"  n_classes: {config['model']['n_classes']}")

# Create model
print('\nCreating model...')
model = HormonicFormer(config)
print(f'  Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

# Create dummy input
print('\nTesting forward...')
dataset = CopyTaskDataset(seq_len=16, vocab_size=10, num_samples=2, d_model=128)
x, y = dataset[0]
print(f'  Input shape: {x.shape}')  # [16, 128]
print(f'  Target shape: {y.shape}')  # [16]

# Batch
x_batch = x.unsqueeze(0)  # [1, 16, 128]
print(f'  Batch input: {x_batch.shape}')

# Forward
logits = model(x_batch, targets=None)
print(f'  Logits shape: {logits.shape}')  # Should be [1, 16, 11]

print('\nTest passed!')
