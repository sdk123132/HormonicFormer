"""
Simple test for char-level LM with HormonicFormer v3
Minimal changes to test the concept
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple char-level dataset
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, text, seq_len=128):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        data = [self.stoi[ch] for ch in text]
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y

# Test with simple text
text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them."""

print(f'Text length: {len(text)}')
print(f'Unique chars: {len(set(text))}')

# Create dataset
dataset = CharDataset(text, seq_len=32)
print(f'Dataset size: {len(dataset)}')
print(f'Vocab size: {dataset.vocab_size}')

# Test one sample
x, y = dataset[0]
print(f'Input shape: {x.shape}')
print(f'Target shape: {y.shape}')
print(f'Input (decoded): ""{ "".join([dataset.itos[i.item()] for i in x]) }""')
print(f'Target (decoded): ""{ "".join([dataset.itos[i.item()] for i in y]) }""')

print('\nChar LM dataset test passed!')
