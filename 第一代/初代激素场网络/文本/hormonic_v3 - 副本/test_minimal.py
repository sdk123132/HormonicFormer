"""
Minimal test - just model creation
"""
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
print('Importing...')
from hormonicformer_v3 import HormonicFormer
print('Imported!')

config = {
    'model': {
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'seq_len': 128,
        'n_steps': 2,
        'n_classes': 10,
        'patch_size': 1,
        'D0_amp': 0.002,
        'D0_phase': 0.002,
        'dt': 0.02,
        'noise_scale': 0.0,
        'dropout': 0.1,
        'ei_balance': {'enabled': False},
        'sensory_feedback': {'enabled': False},
        'hebbian': {'enabled': False},
        'cross_freq_coupling': {'enabled': False},
        'energy_constraint': {'enabled': False}
    },
    'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
    'bwo': {'use_bwo': False},
    'pc': {'use_pc': False}
}

print('Creating model...')
model = HormonicFormer(config)
print(f'Model created! Params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M')

print('Testing forward...')
x = torch.randn(2, 1, 1, 128)
out = model.patch_embed(x)
print(f'patch_embed output: {out.shape}')

print('All tests passed!')
