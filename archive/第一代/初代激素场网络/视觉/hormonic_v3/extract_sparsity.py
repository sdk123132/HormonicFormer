import torch
import os

def analyze_checkpoint(ckpt_path, name):
    print(f'\n=== {name} ===')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model']
    
    # Find all alive_mask keys
    alive_keys = [k for k in state.keys() if 'alive_mask' in k]
    print(f'Found {len(alive_keys)} alive_mask tensors')
    
    total_alive = 0
    total_params = 0
    
    for k in alive_keys:
        mask = state[k]
        alive = mask.sum().item()
        total = mask.numel()
        sparsity = 1.0 - (alive / total)
        print(f'  {k}: alive={alive}/{total}, sparsity={sparsity:.4f} ({sparsity*100:.2f}%)')
        total_alive += alive
        total_params += total
    
    if total_params > 0:
        overall_sparsity = 1.0 - (total_alive / total_params)
        print(f'\n  Overall G_sparsity: {overall_sparsity:.4f} ({overall_sparsity*100:.2f}%)')
        print(f'  Total alive: {total_alive}/{total_params}')
    
    return overall_sparsity if total_params > 0 else 0

# Analyze both checkpoints
sparsity_5 = analyze_checkpoint('checkpoints/epoch_5.pt', 'Epoch 5')
sparsity_10 = analyze_checkpoint('checkpoints/epoch_10.pt', 'Epoch 10')

print(f'\n=== BWO Effect on Sparsity ===')
print(f'Epoch 5 G_sparsity: {sparsity_5*100:.2f}%')
print(f'Epoch 10 G_sparsity: {sparsity_10*100:.2f}%')
print(f'Change: {(sparsity_10 - sparsity_5)*100:.2f}%')
