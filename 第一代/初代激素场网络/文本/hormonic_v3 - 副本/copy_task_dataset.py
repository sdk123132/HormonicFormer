"""
Copy Task Dataset for HormonicFormer v3
"""
import torch
from torch.utils.data import Dataset


class CopyTaskDataset(Dataset):
    """
    Copy Task: 输入 [seq, marker, padding], 输出 [padding, seq]
    
    Args:
        seq_len: 总序列长度
        copy_len: 需要复制的序列长度 (默认 seq_len // 4)
        vocab_size: 词汇表大小 (不含特殊标记)
        num_samples: 样本数量
        d_model: 模型维度 (用于扩展输入)
    """
    def __init__(self, seq_len=128, copy_len=None, vocab_size=10, 
                 num_samples=10000, d_model=128, seed=42):
        self.seq_len = seq_len
        self.copy_len = copy_len or seq_len // 4
        self.vocab_size = vocab_size  # 0-9
        self.num_samples = num_samples
        self.d_model = d_model
        
        # Special tokens
        self.MARKER = vocab_size  # 标记复制开始 (10)
        self.PAD = 0  # 填充
        
        # Fix seed for reproducibility
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequence to copy (1-9, avoid 0 which is padding)
        src = torch.randint(1, self.vocab_size, (self.copy_len,), generator=self.rng)
        
        # Build input: [src, MARKER, padding]
        input_seq = torch.zeros(self.seq_len, dtype=torch.long)
        input_seq[:self.copy_len] = src
        input_seq[self.copy_len] = self.MARKER
        # rest is padding (0)
        
        # Build target: [padding, src]
        target = torch.zeros(self.seq_len, dtype=torch.long)
        target[self.seq_len - self.copy_len:] = src
        
        # Expand input to [seq_len, d_model] for HormonicFormer
        # One-hot encode (vocab_size + 1 for marker)
        input_emb = torch.nn.functional.one_hot(input_seq, num_classes=self.vocab_size + 1).float()
        
        # Project to d_model using random projection (fixed)
        proj = torch.randn(self.vocab_size + 1, self.d_model) * 0.1
        input_emb = input_emb @ proj  # [seq_len, d_model]
            
        return input_emb, target


class CopyTaskValDataset(CopyTaskDataset):
    """Validation set with different seed"""
    def __init__(self, seq_len=128, copy_len=None, vocab_size=10, 
                 num_samples=1000, d_model=128):
        super().__init__(seq_len, copy_len, vocab_size, num_samples, d_model, seed=2024)
