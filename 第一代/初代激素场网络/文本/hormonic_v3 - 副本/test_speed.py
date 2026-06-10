"""
Test model creation speed
"""
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import time
import torch

print('Importing HormonicFormer...')
start = time.time()
from hormonicformer_v3 import HormonicFormer
print(f'Import took {time.time() - start:.2f}s')

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
        'ei_balance': {'enabled': True, 'tau_e': 2.0, 'tau_i': 1.0, 'gamma_e': 1.0, 'gamma_i': 0.8, 'w_inh': 0.3, 'inh_radius': 3},
        'sensory_feedback': {'enabled': True, 'feedback_strength': 0.3, 'feedback_freq': 1},
        'hebbian': {'enabled': True, 'eta_hebb': 0.001, 'eta_anti': 0.0005, 'sync_threshold': 0.5, 'tau_hebb': 10.0, 'decay': 0.999},
        'cross_freq_coupling': {'enabled': False},
        'energy_constraint': {'enabled': False}
    },
    'neuromod': {'da_init': 0.5, 'da_min': 0.1, 'da_max': 0.9, 'use_cb': False},
    'bwo': {'use_bwo': True, 'evolve_interval': 5, 'flip_ratio': 0.3},
    'pc': {'use_pc': True, 'pred_hidden_mult': 4, 'aux_weight': 0.01}
}

print('Creating model...')
start = time.time()
try:
    model = HormonicFormer(config)
    print(f'Model created in {time.time() - start:.2f}s')
    print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
