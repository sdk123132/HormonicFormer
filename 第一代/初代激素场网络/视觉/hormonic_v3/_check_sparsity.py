import torch
import os
import sys

cp_path = 'checkpoints/epoch_5.pt'
if os.path.exists(cp_path):
    ckpt = torch.load(cp_path, map_location='cpu')
    print('=== Epoch 5 Checkpoint ===')
    print('Keys:', list(ckpt.keys()))
    if 'metrics' in ckpt:
        m = ckpt['metrics']
        print('Metrics:', m)
    if 'model_state' in ckpt:
        state = ckpt['model_state']
        for name in ['G', 'DA', 'CB']:
            if name in state:
                param = state[name]
                total = param.numel()
                zeros = (param == 0).sum().item()
                sparsity = zeros / total * 100
                print(f'{name} sparsity: {sparsity:.2f}% ({zeros}/{total})')
            else:
                print(f'{name} not found in state_dict')
        # List all keys
        print('\\nState keys:', list(state.keys()))
else:
    print('Checkpoint not found')
