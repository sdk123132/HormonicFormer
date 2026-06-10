import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, r'C:\Users\MR\Desktop')
sys.path.insert(0, r'C:\Users\MR\Desktop\论文\关于场物理的神经框架\第二代')

import torch
from hormonic_cifar10 import HormonicCIFAR10

CONFIG = {
    'model': {'d_model': 128, 'n_layers': 6, 'n_heads': 4, 'dropout': 0.1, 'n_cgl_steps': 15,
              'D0_amp': 0.002, 'D0_phase': 0.002, 'cgl_dt': 0.02, 'noise_scale': 0.001},
    'use_neuromod': True, 'use_pac': True, 'use_pc': False, 'g_coupling_strength': 0.1,
    'neuromod': {'da_init': 2.5, 'da_ema_alpha': 0.9, 'da_var_alpha': 0.9, 'da_min': 0.1, 'da_max': 0.9,
                 'use_cb': True, 'cb_gain': 2.0, 'cb_threshold': 0.1, 'tau_cb': 10.0, 'cb_dt': 0.05},
    'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05},
    'hebbian': {'eta_potentiate': 0.001, 'eta_depress': 0.0005, 'sync_threshold': 0.3, 'decay': 0.999},
}

model = HormonicCIFAR10(CONFIG)
model = model.cuda()

# 做一次 forward
x = torch.randint(0, 10, (2, 3, 32, 32)).float().cuda()
model.train()
logits, loss = model(x, torch.randint(0, 10, (2,)).cuda())

print("After 1 forward pass:")
try:
    diag = model.get_diagnostics()
    for k, v in diag.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"ERROR: {e}")

# 调用 compute_da
nm = model.backbone.blocks[0].neuromod
print(f"\nDirect access:")
print(f"  da_ema: {nm.da_ema.item():.4f}")
print(f"  cb_state: {nm.cb_state.item():.4f}")
print(f"  stp.u: {nm.stp.u.mean().item():.4f}")
print(f"  stp.r: {nm.stp.r.mean().item():.4f}")
print(f"  prev_amp: {nm.prev_amp}")

# 调用 compute_da 后再看
nm.compute_da(loss.item())
print(f"\nAfter compute_da:")
print(f"  da_ema: {nm.da_ema.item():.4f}")
diag2 = model.get_diagnostics()
for k, v in diag2.items():
    print(f"  {k}: {v}")
