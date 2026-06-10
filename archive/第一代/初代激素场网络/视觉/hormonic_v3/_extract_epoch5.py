import torch
import sys
import os

hv3_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, hv3_path)

ckpt_path = os.path.join(hv3_path, 'checkpoints', 'epoch_5.pt')
ckpt = torch.load(ckpt_path, map_location='cpu')

print('=== Checkpoint Keys ===')
for k in ckpt.keys():
    print(f'  {k}')

print()
print('=== Train Acc ===')
print(f"  {ckpt.get('train_acc', 'N/A')}")

print('=== Val Acc ===')
print(f"  {ckpt.get('val_acc', 'N/A')}")

print('=== DA ===')
print(f"  {ckpt.get('DA', 'N/A')}")

print('=== CB ===')
print(f"  {ckpt.get('CB', 'N/A')}")

print('=== G Sparsity ===')
if 'G_sparsity' in ckpt:
    print(f"  {ckpt['G_sparsity']}")
elif 'model_state_dict' in ckpt:
    # Calculate from model
    state = ckpt['model_state_dict']
    for k, v in state.items():
        if 'G' in k and v.numel() > 0:
            total = v.numel()
            nonzero = (v != 0).sum().item()
            sparsity = 1.0 - (nonzero / total)
            print(f"  {k}: sparsity={sparsity:.4f} ({nonzero}/{total})")
else:
    print('  Not found')

print()
print('=== BWO Changed Sparsity? ===')
# Check if sparsity changed from epoch 4
epoch4_path = os.path.join(hv3_path, 'checkpoints', 'epoch_4.pt')
epoch5_path = os.path.join(hv3_path, 'checkpoints', 'epoch_5.pt')
try:
    ckpt4 = torch.load(epoch4_path, map_location='cpu')
    ckpt5 = torch.load(epoch5_path, map_location='cpu')
    
    if 'G_sparsity' in ckpt4 and 'G_sparsity' in ckpt5:
        print(f"  Epoch 4 G_sparsity: {ckpt4['G_sparsity']}")
        print(f"  Epoch 5 G_sparsity: {ckpt5['G_sparsity']}")
        print(f"  Changed: {ckpt4['G_sparsity'] != ckpt5['G_sparsity']}")
    else:
        print('  G_sparsity key not found in both checkpoints')
        # Try to infer from log
        print('  (Check log for BWO DEBUG messages to confirm sparsity changes)')
except Exception as e:
    print(f'  Error comparing: {e}')
