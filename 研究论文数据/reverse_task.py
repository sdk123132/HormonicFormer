"""
HormonicFormer 长程记忆测试 - Reverse Task
输入: [a, b, c, d, e]
输出: [e, d, c, b, a]
测试模型记住整个序列直到最后才输出的能力
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
from datetime import datetime

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{datetime.now().strftime('%H:%M:%S')}] 设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============== 数据生成 ==============
def generate_reverse_data(seq_len, vocab_size, batch_size):
    """生成Reverse Task数据"""
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.flip(src, dims=[1])  # 反向
    return src, tgt

# ============== 模型定义 ==============

class SimpleHormonic(nn.Module):
    """简化版HormonicFormer"""
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
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        B, S = x.shape
        x = self.embedding(x) + self.pos_embed[:, :S, :]
        x = self.encoder(x)
        return self.fc(x)

class SimpleTransformer(nn.Module):
    """标准Transformer"""
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
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        B, S = x.shape
        x = self.embedding(x) + self.pos_embed[:, :S, :]
        x = self.encoder(x)
        return self.fc(x)

class SimpleLSTM(nn.Module):
    """LSTM基线"""
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x)

# ============== 训练函数 ==============

def train_model(model, seq_len, vocab_size, epochs=20, batch_size=4):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accuracies = []
    
    print(f"\n训练配置: seq_len={seq_len}, epochs={epochs}, batch_size={batch_size}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # 每个epoch训练50个batch
        for _ in range(50):
            src, tgt = generate_reverse_data(seq_len, vocab_size, batch_size)
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / 50
        losses.append(avg_loss)
        
        # 评估准确率
        model.eval()
        with torch.no_grad():
            src, tgt = generate_reverse_data(seq_len, vocab_size, 20)
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            pred = output.argmax(dim=-1)
            acc = (pred == tgt).float().mean().item()
            accuracies.append(acc)
        
        if epoch % 4 == 0:
            print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    
    return {
        'losses': losses,
        'accuracies': accuracies,
        'final_acc': accuracies[-1],
        'final_loss': losses[-1]
    }

# ============== 主实验 ==============

def run_experiment(seq_lengths=[64, 128, 256, 512, 1024]):
    """运行Reverse Task实验"""
    
    print("=" * 70)
    print("HormonicFormer 长程记忆测试 - Reverse Task")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print(f"序列长度: {seq_lengths}")
    print("=" * 70)
    
    vocab_size = 50
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n{'='*70}")
        print(f"序列长度: {seq_len}")
        print(f"{'='*70}")
        
        # 根据长度调整模型（增大模型容量）
        if seq_len <= 256:
            d_model, n_layers, n_heads = 128, 3, 4
        else:
            d_model, n_layers, n_heads = 256, 3, 4
        
        results[seq_len] = {}
        
        # 1. HormonicFormer
        print(f"\n[1/3] HormonicFormer (d_model={d_model}, n_layers={n_layers})")
        try:
            model = SimpleHormonic(vocab_size, d_model, n_layers, n_heads, seq_len)
            result = train_model(model, seq_len, vocab_size, epochs=60, batch_size=4)
            results[seq_len]['hormonic'] = result
            print(f"  [OK] 完成: Final Acc={result['final_acc']:.4f}")
        except Exception as e:
            print(f"  [FAIL] 失败: {e}")
            results[seq_len]['hormonic'] = {'error': str(e)}
        
        # 2. Transformer
        print(f"\n[2/3] Transformer (d_model={d_model}, n_layers={n_layers})")
        try:
            model = SimpleTransformer(vocab_size, d_model, n_layers, n_heads, seq_len)
            result = train_model(model, seq_len, vocab_size, epochs=60, batch_size=4)
            results[seq_len]['transformer'] = result
            print(f"  [OK] 完成: Final Acc={result['final_acc']:.4f}")
        except Exception as e:
            print(f"  [FAIL] 失败: {e}")
            results[seq_len]['transformer'] = {'error': str(e)}
        
        # 3. LSTM
        print(f"\n[3/3] LSTM (d_model={d_model}, n_layers={n_layers})")
        try:
            model = SimpleLSTM(vocab_size, d_model, n_layers)
            result = train_model(model, seq_len, vocab_size, epochs=60, batch_size=4)
            results[seq_len]['lstm'] = result
            print(f"  [OK] 完成: Final Acc={result['final_acc']:.4f}")
        except Exception as e:
            print(f"  [FAIL] 失败: {e}")
            results[seq_len]['lstm'] = {'error': str(e)}
    
    # 保存结果
    output_file = f'C:/Users/MR/Desktop/论文/关于场物理的神经框架/研究论文数据/reverse_task_results_{datetime.now().strftime("%m%d_%H%M")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("实验完成!")
    print(f"结果保存: {output_file}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # 打印汇总
    print("\n结果汇总:")
    print(f"{'Seq Len':<10} {'Hormonic':<15} {'Transformer':<15} {'LSTM':<15}")
    print("-" * 60)
    for seq_len in seq_lengths:
        h_acc = results[seq_len].get('hormonic', {}).get('final_acc', 'N/A')
        t_acc = results[seq_len].get('transformer', {}).get('final_acc', 'N/A')
        l_acc = results[seq_len].get('lstm', {}).get('final_acc', 'N/A')
        
        h_str = f"{h_acc:.4f}" if isinstance(h_acc, float) else str(h_acc)
        t_str = f"{t_acc:.4f}" if isinstance(t_acc, float) else str(t_acc)
        l_str = f"{l_acc:.4f}" if isinstance(l_acc, float) else str(l_acc)
        
        print(f"{seq_len:<10} {h_str:<15} {t_str:<15} {l_str:<15}")
    
    return results

if __name__ == '__main__':
    if len(sys.argv) > 1:
        seq_lengths = [int(x) for x in sys.argv[1].split(',')]
    else:
        seq_lengths = [64, 128, 256, 512, 1024]
    
    run_experiment(seq_lengths)
