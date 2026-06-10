"""
Quick test for Copy Task
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add paths
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

from hormonicformer_v3 import HormonicFormer

config = {
    'model': {
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'seq_len': 16,
        'n_steps': 3,
        'n_classes': 10,
        'patch_size': 1,
        'D0_amp': 0.002,
        'D0_phase': 0.002,
        'dt': 0.02,
        'noise_scale': 0.01,
        'dropout': 0.1,
        'ei_balance': {
            'enabled': True,
            'tau_e': 2.0,
            'tau_i': 1.0,
            'gamma_e': 1.0,
            'gamma_i': 0.8,
            'w_inh': 0.3,
            'inh_radius': 3
        },
        'sensory_feedback': {
            'enabled': True,
            'feedback_strength': 0.3,
            'feedback_freq': 1
        },
        'hebbian': {
            'enabled': True,
            'eta_hebb': 0.001,
            'eta_anti': 0.0005,
            'sync_threshold': 0.5,
            'tau_hebb': 10.0,
            'decay': 0.999
        },
        'cross_freq_coupling': {'enabled': False},
        'energy_constraint': {'enabled': False}
    },
    'neuromod': {
        'da_init': 0.5,
        'da_min': 0.1,
        'da_max': 0.9,
        'use_cb': False
    },
    'bwo': {
        'use_bwo': True,
        'evolve_interval': 5,
        'flip_ratio': 0.3
    },
    'pc': {
        'use_pc': True,
        'pred_hidden_mult': 4,
        'aux_weight': 0.01
    }
}

print('Creating model...')
model = HormonicFormer(config)
print(f'Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

# Test forward
print('\nTesting forward...')
B, S = 2, 16
seq = torch.randint(0, 10, (B, S))
vals = (seq.float() / 9) * 2 - 1
img = vals.view(B, 1, 1, S)
print(f'Input shape: {img.shape}')

try:
    x = model.patch_embed(img)
    print(f'After patch_embed: {x.shape}')
    x = x.flatten(2).transpose(1, 2)
    print(f'After flatten: {x.shape}')
    
    x_embed = x.detach().clone()
    for block in model.blocks:
        x = block(x, x_embed=x_embed if block.use_feedback else None)
    print(f'After blocks: {x.shape}')
    
    # Replace classifier
    model.classifier = nn.Linear(64, 10)
    logits = model.classifier(x)
    print(f'Logits shape: {logits.shape}')
    
    print('\nTest passed!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
