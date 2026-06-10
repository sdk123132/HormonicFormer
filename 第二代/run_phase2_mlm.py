"""
阶段 2：序列任务验证（字符级 MLM）
使用 WikiText-2 前 100万字符
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, r'C:\Users\MR\Desktop')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import urllib.request

# 配置
CONFIG = {
    'model': {
        'd_model': 128,
        'seq_len': 128,
        'n_layers': 4,
        'n_heads': 4,
        'vocab_size': 100,  # 字符级
        'dropout': 0.1,
        'n_cgl_steps': 10,
        'D0_amp': 0.002,
        'D0_phase': 0.002,
        'cgl_dt': 0.02,
        'noise_scale': 0.001,
    },
    'use_neuromod': True,
    'use_pac': True,
    'use_pc': False,
    'g_coupling_strength': 0.1,
    'neuromod': {
        'da_init': 2.5,
        'da_ema_alpha': 0.9,
        'da_var_alpha': 0.9,
        'da_min': 0.1,
        'da_max': 0.9,
        'use_cb': True,
        'cb_gain': 2.0,
        'cb_threshold': 0.25,
        'tau_cb': 10.0,
        'cb_dt': 0.05,
    },
    'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05},
    'hebbian': {'eta_potentiate': 0.001, 'eta_depress': 0.0005, 
                'sync_threshold': 0.3, 'decay': 0.999},
}

TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 10,
    'lr': 3e-4,
    'weight_decay': 0.05,
}

# 下载 WikiText-2
def download_wikitext2():
    url = 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt'
    path = './data/wikitext2_train.txt'
    if not os.path.exists(path):
        print('Downloading WikiText-2...')
        urllib.request.urlretrieve(url, path)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text[:1000000]  # 前100万字符

# 字符级 tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.char2idx.get(ch, 0) for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx2char.get(i, '?') for i in indices])

# 数据集
class CharMLMDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=128, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.data = tokenizer.encode(text)
    
    def __len__(self):
        return len(self.data) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        tokens = self.data[start:end]
        
        # Mask tokens
        masked = tokens.copy()
        labels = [-100] * len(tokens)
        for i in range(len(tokens)):
            if torch.rand(1).item() < self.mask_prob:
                labels[i] = tokens[i]
                if torch.rand(1).item() < 0.8:
                    masked[i] = self.tokenizer.char2idx.get('<mask>', 0)
                elif torch.rand(1).item() < 0.5:
                    masked[i] = torch.randint(0, self.tokenizer.vocab_size, (1,)).item()
        
        return torch.tensor(masked), torch.tensor(labels)

# 主函数
def main():
    print("=" * 60)
    print("阶段 2：字符级 MLM 验证")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 数据
    print("\nLoading WikiText-2...")
    text = download_wikitext2()
    print(f"Text length: {len(text)} chars")
    
    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    CONFIG['model']['vocab_size'] = tokenizer.vocab_size
    
    dataset = CharMLMDataset(text, tokenizer, seq_len=128)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    trainloader = DataLoader(trainset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    valloader = DataLoader(valset, batch_size=TRAIN_CONFIG['batch_size'])
    print(f"Train: {len(trainset)} batches, Val: {len(valset)} batches")
    
    # 模型
    print("\nBuilding model...")
    from hormonic_v7r3_validated import HormonicFormerV7r3
    model = HormonicFormerV7r3(CONFIG).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['lr'], 
                                   weight_decay=TRAIN_CONFIG['weight_decay'])
    
    # TensorBoard
    log_dir = 'runs/phase2_mlm'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 训练
    print(f"\nTraining {TRAIN_CONFIG['epochs']} epochs...")
    best_loss = float('inf')
    
    for epoch in range(TRAIN_CONFIG['epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            # MLM loss
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), 
                                  labels.view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'  [{batch_idx}/{len(trainloader)}] Loss: {loss.item():.4f}')
                writer.add_scalar('Train/loss', loss.item(), epoch * len(trainloader) + batch_idx)
        
        avg_loss = total_loss / len(trainloader)
        print(f'\nEpoch {epoch+1}: Train Loss = {avg_loss:.4f}')
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), 
                                      labels.view(-1), ignore_index=-100)
                val_loss += loss.item()
        
        val_loss /= len(valloader)
        print(f'Val Loss = {val_loss:.4f}')
        
        writer.add_scalar('Epoch/train_loss', avg_loss, epoch)
        writer.add_scalar('Epoch/val_loss', val_loss, epoch)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'{log_dir}/best_model.pt')
            print('*** New best model saved! ***')
    
    writer.close()
    print(f"\nBest Val Loss: {best_loss:.4f}")

if __name__ == '__main__':
    main()
