import torch
import sys

ckpt_path = r'C:\Users\MR\Desktop\论文\关于场物理的神经框架（失败）\第一代\初代激素场网络\视觉\hormonic_v3\checkpoints\epoch_5.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

print('Checkpoint keys:', list(ckpt.keys()))

# Look for sparsity info in state_dict
if 'state_dict' in ckpt:
    state = ckpt['state_dict']
    sparsity_keys = [k for k in state.keys() if 'alive_mask' in k or 'sparsity' in k]
    print('Sparsity-related keys:', sparsity_keys[:20])
    
    # Calculate sparsity from alive_mask
    for key in sparsity_keys:
        if 'alive_mask' in key:
            mask = state[key]
            if hasattr(mask, 'float'):
                sparsity = 1.0 - mask.float().mean().item()
                print(f'{key}: sparsity = {sparsity:.4f}')

# Also check for config
if 'config' in ckpt:
    print('Config:', ckpt['config'])

# Check for blocks with hebbian
if 'state_dict' in ckpt:
    block_keys = [k for k in state.keys() if 'blocks.' in k and 'hebbian' in k]
    print('Hebbian block keys:', len(block_keys))
    for key in block_keys[:10]:
        print(f'  {key}')
