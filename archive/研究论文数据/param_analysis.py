import sys
sys.path.insert(0, r'C:\Users\MR\Desktop')
from hormonic_v7r3_validated import HormonicBlockV7r3
import torch
import torch.nn as nn

for s in [50, 100, 200, 500, 1000]:
    config = {
        'model': {
            'd_model': 32,
            'seq_len': s,
            'n_layers': 2,
            'n_heads': 4,
            'n_cgl_steps': 5,
            'D0_amp': 0.002,
            'D0_phase': 0.002,
            'cgl_dt': 0.02,
            'noise_scale': 0.001,
            'dropout': 0.1,
        },
        'use_neuromod': True,
        'use_pac': True,
        'use_pc': False,
        'g_coupling_strength': 0.1,
        'hebbian': {},
        'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05},
    }
    
    class TestModel(nn.Module):
        def __init__(self, seq_len):
            super().__init__()
            self.input_proj = nn.Linear(2, 64)
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, 64) * 0.02)
            self.blocks = nn.ModuleList([
                HormonicBlockV7r3(32, seq_len, config)
                for _ in range(2)
            ])
            self.output_norm = nn.LayerNorm(64)
            self.predictor = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
            )
    
    model = TestModel(s)
    total = sum(p.numel() for p in model.parameters())
    print(f"S={s}: {total:,} params")
    
    params = sorted([(n, p.numel()) for n, p in model.named_parameters()], 
                    key=lambda x: -x[1])
    for n, c in params[:10]:
        pct = c / total * 100
        print(f"  {n}: {c:,} ({pct:.1f}%)")
    print()
