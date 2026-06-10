"""
HormonicFormer 层次化解析测试 - ListOps任务
输入: [MAX [MIN 3 4] [MEDIAN 1 2 3]]
输出: 解析结果（数值）
测试模型理解嵌套结构和层次化依赖的能力
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
from datetime import datetime
import random

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{datetime.now().strftime('%H:%M:%S')}] 设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============== ListOps数据生成 ==============

class ListOpsGenerator:
    """生成ListOps数据"""
    def __init__(self, max_depth=3, max_value=10):
        self.ops = ['MAX', 'MIN', 'MED', 'SUM']
        self.max_depth = max_depth
        self.max_value = max_value
        
    def generate_expression(self, depth=0):
        """递归生成表达式"""
        if depth >= self.max_depth or random.random() < 0.3:
            # 叶子节点：数值
            return random.randint(1, self.max_value)
        else:
            # 内部节点：操作符 + 子表达式
            op = random.choice(self.ops)
            n_args = random.randint(2, 4)
            args = [self.generate_expression(depth + 1) for _ in range(n_args)]
            return [op] + args
    
    def evaluate(self, expr):
        """计算表达式值"""
        if isinstance(expr, int):
            return expr
        
        op = expr[0]
        args = [self.evaluate(arg) for arg in expr[1:]]
        
        if op == 'MAX':
            return max(args)
        elif op == 'MIN':
            return min(args)
        elif op == 'MED':
            return sorted(args)[len(args) // 2]
        elif op == 'SUM':
            return sum(args)
        else:
            return 0
    
    def to_sequence(self, expr):
        """将表达式转为token序列"""
        tokens = []
        self._to_sequence_recursive(expr, tokens)
        return tokens
    
    def _to_sequence_recursive(self, expr, tokens):
        if isinstance(expr, int):
            tokens.append(f'NUM_{expr}')
        else:
            tokens.append(expr[0])  # 操作符
            for arg in expr[1:]:
                self._to_sequence_recursive(arg, tokens)
            tokens.append('END')
    
    def generate_batch(self, batch_size, seq_len):
        """生成一批数据"""
        vocab = self.ops + [f'NUM_{i}' for i in range(1, self.max_value + 1)] + ['END', 'PAD']
        token_to_id = {token: i for i, token in enumerate(vocab)}
        
        src_list = []
        tgt_list = []
        
        for _ in range(batch_size):
            expr = self.generate_expression()
            tokens = self.to_sequence(expr)
            
            # 填充或截断到seq_len
            if len(tokens) < seq_len:
                tokens = tokens + ['PAD'] * (seq_len - len(tokens))
            else:
                tokens = tokens[:seq_len]
            
            # 目标值（归一化到0-1）
            result = self.evaluate(expr)
            tgt = (result - 1) / (self.max_value * 4)  # 归一化
            
            src_ids = [token_to_id.get(t, 0) for t in tokens]
            src_list.append(src_ids)
            tgt_list.append(tgt)
        
        return torch.tensor(src_list), torch.tensor(tgt_list).float()

# ============== 模型定义 ==============

class ListOpsHormonic(nn.Module):
    """HormonicFormer for ListOps"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 全局池化 + 回归头
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        B, S = x.shape
        x = self.embedding(x) + self.pos_embed[:, :S, :]
        x = self.encoder(x)
        
        # 全局平均池化
        x = x.transpose(1, 2)  # [B, d_model, S]
        x = self.pool(x).squeeze(-1)  # [B, d_model]
        
        return self.regressor(x).squeeze(-1)

class ListOpsTransformer(nn.Module):
    """Transformer for ListOps"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        B, S = x.shape
        x = self.embedding(x) + self.pos_embed[:, :S, :]
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.regressor(x).squeeze(-1)

class ListOpsLSTM(nn.Module):
    """LSTM for ListOps"""
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True, bidirectional=True)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        # 取最后一个时间步
        x = x[:, -1, :]
        return self.regressor(x).squeeze(-1)

# ============== 训练函数 ==============

def train_model(model, seq_len, epochs=30, batch_size=8):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    generator = ListOpsGenerator(max_depth=3, max_value=10)
    
    losses = []
    
    print(f"\n训练配置: seq_len={seq_len}, epochs={epochs}, batch_size={batch_size}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # 每个epoch训练100个batch
        for _ in range(100):
            src, tgt = generator.generate_batch(batch_size, seq_len)
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / 100
        losses.append(avg_loss)
        
        if epoch % 5 == 0:
            # 评估
            model.eval()
            with torch.no_grad():
                src, tgt = generator.generate_batch(50, seq_len)
                src, tgt = src.to(device), tgt.to(device)
                output = model(src)
                eval_loss = criterion(output, tgt).item()
                # 计算MAE
                mae = torch.abs(output - tgt).mean().item()
            print(f"  Epoch {epoch}: Train Loss={avg_loss:.4f}, Eval Loss={eval_loss:.4f}, MAE={mae:.4f}")
    
    return {
        'losses': losses,
        'final_loss': losses[-1]
    }

# ============== 主实验 ==============

def run_experiment(seq_lengths=[32, 64, 128]):
    """运行ListOps实验"""
    
    print("=" * 70)
    print("HormonicFormer 层次化解析测试 - ListOps")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print(f"序列长度: {seq_lengths}")
    print("=" * 70)
    
    vocab_size = 20  # MAX, MIN, MED, SUM, NUM_1-10, END, PAD
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n{'='*70}")
        print(f"序列长度: {seq_len}")
        print(f"{'='*70}")
        
        # 模型配置
        d_model, n_layers, n_heads = 64, 2, 4
        
        results[seq_len] = {}
        
        # 1. HormonicFormer
        print(f"\n[1/3] HormonicFormer (d_model={d_model}, n_layers={n_layers})")
        try:
            model = ListOpsHormonic(vocab_size, d_model, n_layers, n_heads, seq_len)
            result = train_model(model, seq_len, epochs=30, batch_size=8)
            results[seq_len]['hormonic'] = result
            print(f"  [OK] 完成: Final Loss={result['final_loss']:.4f}")
        except Exception as e:
            print(f"  [FAIL] 失败: {e}")
            results[seq_len]['hormonic'] = {'error': str(e)}
        
        # 2. Transformer
        print(f"\n[2/3] Transformer (d_model={d_model}, n_layers={n_layers})")
        try:
            model = ListOpsTransformer(vocab_size, d_model, n_layers, n_heads, seq_len)
            result = train_model(model, seq_len, epochs=30, batch_size=8)
            results[seq_len]['transformer'] = result
            print(f"  [OK] 完成: Final Loss={result['final_loss']:.4f}")
        except Exception as e:
            print(f"  [FAIL] 失败: {e}")
            results[seq_len]['transformer'] = {'error': str(e)}
        
        # 3. LSTM
        print(f"\n[3/3] LSTM (d_model={d_model}, n_layers={n_layers})")
        try:
            model = ListOpsLSTM(vocab_size, d_model, n_layers)
            result = train_model(model, seq_len, epochs=30, batch_size=8)
            results[seq_len]['lstm'] = result
            print(f"  [OK] 完成: Final Loss={result['final_loss']:.4f}")
        except Exception as e:
            print(f"  [FAIL] 失败: {e}")
            results[seq_len]['lstm'] = {'error': str(e)}
    
    # 保存结果
    output_file = f'C:/Users/MR/Desktop/论文/关于场物理的神经框架/研究论文数据/listops_results_{datetime.now().strftime("%m%d_%H%M")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("实验完成!")
    print(f"结果保存: {output_file}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # 打印汇总
    print("\n结果汇总 (Final Loss):")
    print(f"{'Seq Len':<10} {'Hormonic':<15} {'Transformer':<15} {'LSTM':<15}")
    print("-" * 60)
    for seq_len in seq_lengths:
        h_loss = results[seq_len].get('hormonic', {}).get('final_loss', 'N/A')
        t_loss = results[seq_len].get('transformer', {}).get('final_loss', 'N/A')
        l_loss = results[seq_len].get('lstm', {}).get('final_loss', 'N/A')
        
        h_str = f"{h_loss:.4f}" if isinstance(h_loss, float) else str(h_loss)
        t_str = f"{t_loss:.4f}" if isinstance(t_loss, float) else str(t_loss)
        l_str = f"{l_loss:.4f}" if isinstance(l_loss, float) else str(l_loss)
        
        print(f"{seq_len:<10} {h_str:<15} {t_str:<15} {l_str:<15}")
    
    return results

if __name__ == '__main__':
    if len(sys.argv) > 1:
        seq_lengths = [int(x) for x in sys.argv[1].split(',')]
    else:
        seq_lengths = [32, 64, 128]
    
    run_experiment(seq_lengths)
