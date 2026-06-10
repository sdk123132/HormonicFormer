import torch

ckpt = torch.load('checkpoints/epoch_5.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('Epoch:', ckpt.get('epoch', 'N/A'))
print('DA:', ckpt.get('DA', 'N/A'))
print('CB:', ckpt.get('CB', 'N/A'))

# Get sparsity info
state = ckpt['model']
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
    print(f'Overall G_sparsity: {overall_sparsity:.4f} ({overall_sparsity*100:.2f}%)')
