import torch
import os

ckpt_path = 'checkpoints/epoch_5.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

print('=== Epoch 5 Checkpoint ===')
print('Keys:', list(ckpt.keys()))

if 'model' in ckpt:
    model = ckpt['model']
    print('\nModel type:', type(model))
    if hasattr(model, 'state_dict'):
        state = model.state_dict()
        print('State dict keys count:', len(state.keys()))
        
        # Look for hebbian-related keys
        hebbian_keys = [k for k in state.keys() if 'hebbian' in k.lower() or 'alive' in k.lower()]
        print(f'\nHebbian-related keys ({len(hebbian_keys)}):')
        for k in hebbian_keys[:20]:
            print(f'  {k}')
        
        # Check for G, DA, CB in state
        for name in ['G', 'DA', 'CB']:
            matching = [k for k in state.keys() if name in k]
            if matching:
                print(f'\n{name} found in keys: {matching[:5]}')

# Check epoch_10 as well
print('\n\n=== Epoch 10 Checkpoint ===')
ckpt10 = torch.load('checkpoints/epoch_10.pt', map_location='cpu')
print('Keys:', list(ckpt10.keys()))
