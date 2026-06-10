"""
HormonicFormer v3 Copy Task Training Script for RTX 5070 8GB
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.hormonicformer_v3 import HormonicFormer
from copy_task_dataset import CopyTaskDataset, CopyTaskValDataset


def parse_args():
    parser = argparse.ArgumentParser(description='HormonicFormer v3 Copy Task')
    parser.add_argument('--config', type=str, default='local_v3_copytask_5070.yaml')
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    return parser.parse_args()


def load_config(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.seq_len:
        config['model']['seq_len'] = args.seq_len
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
        
    return config


def train_epoch(model, loader, optimizer, criterion, device, epoch, config):
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)  # [B, S, d_model]
        targets = targets.to(device)  # [B, S]
        
        optimizer.zero_grad()
        
        # Forward - v3 returns logits when targets is None
        logits = model(inputs, targets=None)
        
        # Compute loss on copy portion only
        copy_len = config['model']['seq_len'] // 4
        ce_loss = criterion(logits[:, -copy_len:].reshape(-1, logits.size(-1)), 
                           targets[:, -copy_len:].reshape(-1))
        
        # Backward
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
        optimizer.step()
        
        # Metrics
        preds = logits.argmax(dim=-1)
        acc = (preds[:, -copy_len:] == targets[:, -copy_len:]).float().mean().item()
        
        total_loss += ce_loss.item()
        total_acc += acc
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'  Batch[{batch_idx}/{len(loader)}] Loss: {ce_loss.item():.4f} Acc: {acc*100:.1f}%')
    
    return total_loss / num_batches, total_acc / num_batches


def validate(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs, targets=None)
            
            copy_len = config['model']['seq_len'] // 4
            ce_loss = criterion(logits[:, -copy_len:].reshape(-1, logits.size(-1)),
                               targets[:, -copy_len:].reshape(-1))
            
            preds = logits.argmax(dim=-1)
            acc = (preds[:, -copy_len:] == targets[:, -copy_len:]).float().mean().item()
            
            total_loss += ce_loss.item()
            total_acc += acc
            num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def main():
    args = parse_args()
    config = load_config(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    
    # Model
    model = HormonicFormer(config).to(device)
    
    print(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
    
    # Datasets
    train_ds = CopyTaskDataset(
        seq_len=config['model']['seq_len'],
        vocab_size=config['model']['n_classes'] - 1,
        num_samples=config['train']['num_samples'],
        d_model=config['model']['d_model']
    )
    val_ds = CopyTaskValDataset(
        seq_len=config['model']['seq_len'],
        vocab_size=config['model']['n_classes'] - 1,
        num_samples=config['train']['num_val_samples'],
        d_model=config['model']['d_model']
    )
    
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'],
                            shuffle=False, num_workers=0)
    
    print(f'Train: {len(train_ds)} samples, Val: {len(val_ds)} samples')
    print(f'Seq len: {config["model"]["seq_len"]}, Copy len: {config["model"]["seq_len"]//4}')
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training
    best_acc = 0
    for epoch in range(config['train']['epochs']):
        print(f'\nEpoch {epoch+1}/{config["train"]["epochs"]}')
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, 
                                          device, epoch, config)
        val_loss, val_acc = validate(model, val_loader, criterion, device, config)
        
        print(f'Train: Loss={train_loss:.4f} Acc={train_acc*100:.1f}%')
        print(f'Val:   Loss={val_loss:.4f} Acc={val_acc*100:.1f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, 'best_copytask.pt')
            print(f'  Saved best: {val_acc*100:.1f}%')
    
    print(f'\nBest Val Acc: {best_acc*100:.1f}%')


if __name__ == '__main__':
    main()
