"""
HormonicFormer v3 Character-Level Language Model
Simplified version for RTX 5070 8GB
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.hormonicformer_v3 import HormonicFormer


class CharDataset(torch.utils.data.Dataset):
    """Character-level language model dataset"""
    def __init__(self, text, seq_len=128, train=True, train_split=0.9):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        data = [self.stoi[ch] for ch in text]
        data = torch.tensor(data, dtype=torch.long)
        
        n = int(train_split * len(data))
        if train:
            self.data = data[:n]
        else:
            self.data = data[n:]
        
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


class HormonicFormerLM(nn.Module):
    """Wrapper for HormonicFormer to do language modeling"""
    def __init__(self, base_model, vocab_size, seq_len, d_model):
        super().__init__()
        self.base = base_model
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Add token embedding and LM head
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Share weights
        self.lm_head.weight = self.token_embedding.weight
        
    def forward(self, input_ids, targets=None):
        B, S = input_ids.shape
        
        # Embed tokens
        x = self.token_embedding(input_ids)  # [B, S, d_model]
        x = x + self.pos_embedding[:, :S, :]
        
        # Pass through HormonicFormer (treat as sequence input)
        # We need to modify the forward to accept pre-embedded input
        # For now, use a workaround: pass through blocks directly
        
        # Get the blocks from base model
        x_embed = x.detach().clone()
        aux_loss = 0
        
        for i, block in enumerate(self.base.blocks):
            x_out = block(x, x_embed=x_embed if block.use_feedback else None)
            if self.base.predictor is not None and i > 0:
                pred = self.base.predictor(x)
                aux_loss = aux_loss + F.mse_loss(pred, x_out.detach())
            x = x_out
        
        # LM head
        logits = self.lm_head(x)  # [B, S, vocab_size]
        
        if targets is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1)
            )
            total_loss = ce_loss + self.base.aux_weight * aux_loss
            return logits, total_loss
        
        return logits


def download_tiny_shakespeare(path):
    """Download Tiny Shakespeare dataset"""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f'Downloading Tiny Shakespeare...')
    urllib.request.urlretrieve(url, path)
    print(f'Saved to {path}')


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        logits, loss = model(inputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f'  Batch[{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}')
    
    return total_loss / num_batches


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits, loss = model(inputs, targets)
            
            # Calculate accuracy (next token prediction)
            shift_logits = logits[:, :-1, :]
            shift_targets = targets[:, 1:]
            preds = shift_logits.argmax(dim=-1)
            acc = (preds == shift_targets).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
            num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def generate_text(model, dataset, device, prompt="To be, or not to be", length=200):
    """Generate text from prompt"""
    model.eval()
    
    # Encode prompt
    prompt_ids = [dataset.stoi.get(c, 0) for c in prompt]
    x = torch.tensor([prompt_ids[-model.base.config['model']['seq_len']:]], dtype=torch.long).to(device)
    
    generated = list(prompt)
    
    with torch.no_grad():
        for _ in range(length):
            # Forward
            logits = model(x)
            
            # Sample next token
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(dataset.itos[next_token])
            
            # Update input
            x = torch.cat([x[:, 1:], torch.tensor([[next_token]], device=device)], dim=1)
    
    return ''.join(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/char_lm.yaml')
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.seq_len:
        config['model']['seq_len'] = args.seq_len
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    
    # Download data
    data_path = config['train']['data_path']
    download_tiny_shakespeare(data_path)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f'\nDataset: Tiny Shakespeare')
    print(f'  Total chars: {len(text)}')
    
    # Create datasets
    train_ds = CharDataset(text, config['model']['seq_len'], train=True)
    val_ds = CharDataset(text, config['model']['seq_len'], train=False)
    
    print(f'  Train samples: {len(train_ds)}')
    print(f'  Val samples: {len(val_ds)}')
    print(f'  Vocab size: {train_ds.vocab_size}')
    
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'],
                           shuffle=False, num_workers=0)
    
    # Create base model (image cls config but we'll override)
    # Need to modify config for image cls to work
    base_config = config.copy()
    base_config['model']['n_classes'] = config['model']['vocab_size']
    base_config['model']['patch_size'] = 1
    
    base_model = HormonicFormer(base_config).to(device)
    
    # Wrap with LM head
    model = HormonicFormerLM(
        base_model,
        vocab_size=train_ds.vocab_size,
        seq_len=config['model']['seq_len'],
        d_model=config['model']['d_model']
    ).to(device)
    
    print(f'\nModel: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    
    # Training
    best_loss = float('inf')
    for epoch in range(config['train']['epochs']):
        print(f'\nEpoch {epoch+1}/{config["train"]["epochs"]}')
        
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, device)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%')
        
        # Generate sample
        if epoch % 5 == 0:
            generated = generate_text(model, train_ds, device, 
                                     prompt="To be, or not to be", length=100)
            print(f'\nGenerated: {generated[:150]}...')
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'vocab': {'stoi': train_ds.stoi, 'itos': train_ds.itos},
                'config': config
            }, 'best_char_lm.pt')
            print(f'  Saved best model')
    
    print(f'\nBest Val Loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
