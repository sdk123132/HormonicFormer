"""
Test DataLoader
"""
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'field')

import torch
from torch.utils.data import Dataset, DataLoader

class CopyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        s = self.data[idx]
        return s, s.clone()

print('Creating dataset...')
ds = CopyDataset(10, 16, 1000)
print(f'Dataset: {len(ds)}')

print('Creating DataLoader...')
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
print(f'Loader: {len(loader)} batches')

print('Iterating...')
for i, (x, y) in enumerate(loader):
    print(f'Batch {i}: x={x.shape}, y={y.shape}')
    if i >= 2:
        break

print('Done!')
