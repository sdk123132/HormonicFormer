import torch
import os

ckpt_path = 'checkpoints/epoch_5.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

print('=== Epoch 5 Checkpoint ===')
state = ckpt['model']
print('State dict keys count:', len(state.keys()))
print('\nAll keys:')
for k in sorted(state.keys()):
    print(f'  {k}')
