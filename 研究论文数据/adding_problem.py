"""
HormonicFormer 长程记忆测试 - Adding Problem
经典长程依赖基准：记住两个远距离标记位置的数值并求和
可以做距离扫描，展示HormonicFormer在长距离上的优势
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
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
def generate_adding_data(n_samples, seq_len, min_dist=None):
    """
    生成Adding Problem数据
    输入: [seq_len, 2] - 第一列是随机数(0-1), 第二列是mask(0/1)
    输出: 标量 - 被mask标记的两个数的和
    
    min_dist: mask两个1之间的最小距离（控制长程依赖难度）
    """
    X = np.zeros((n_samples, seq_len, 2), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    
    for i in range(n_samples):
        # 第一列：随机数 U(0, 1)
        X[i, :, 0] = np.random.uniform(0, 1, seq_len)
        
        # 第二列：mask，只有两个位置是1
        if min_dist is None:
            # 随机选两个不同位置
            positions = np.random.choice(seq_len, 2, replace=False)
        else:
            # 控制最小距离
            pos1 = np.random.randint(0, seq_len // 4)  # 前1/4
            pos2 = np.random.randint(max(pos1 + min_dist, seq_len * 3 // 4), seq_len)  # 后1/4
            positions = [pos1, pos2]
        
        X[i, positions[0], 1] = 1.0
        X[i, positions[1], 1] = 1.0
        
        # 输出：被标记的两个数的和
        y[i, 0] = X[i, positions[0], 0] + X[i, positions[1], 0]
    
    return X, y

# ============== 模型定义 ==============
class AddingHormonic(nn.Module):
    """HormonicFormer for Adding Problem"""
    def __init__(self, d_model, n_layers, n_heads, seq_len):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.predictor(x)

class AddingTransformer(nn.Module):
    """Transformer for Adding Problem"""
    def __init__(self, d_model, n_layers, n_heads, seq_len):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.predictor(x)

class AddingLSTM(nn.Module):
    """LSTM for Adding Problem"""
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.predictor(x)

# ============== 训练函数 ==============
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = criterion(val_output, y_val_t).item()
            history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Train={avg_loss:.6f}, Val={val_loss:.6f}")
    
    # 基线MSE（预测均值）
    baseline_mse = ((y_val - y_val.mean()) ** 2).mean()
    
    history['best_val_loss'] = best_val_loss
    history['baseline_mse'] = float(baseline_mse)
    history['relative_improvement'] = float(1 - best_val_loss / baseline_mse) * 100
    
    return history

# ============== 主实验：距离扫描 ==============
def run_experiment():
    """运行Adding Problem距离扫描实验"""
    
    print("=" * 70)
    print("HormonicFormer 长程记忆测试 - Adding Problem 距离扫描")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 实验配置：不同序列长度（距离扫描）
    seq_lengths = [50, 100, 200, 500, 1000]
    n_train = 5000
    n_val = 1000
    epochs = 50
    
    d_model = 64
    n_layers = 2
    n_heads = 4
    
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n{'='*70}")
        print(f"序列长度: S={seq_len}")
        print(f"{'='*70}")
        
        # 生成数据（控制最小距离为序列长度的一半）
        min_dist = seq_len // 2
        X_train, y_train = generate_adding_data(n_train, seq_len, min_dist)
        X_val, y_val = generate_adding_data(n_val, seq_len, min_dist)
        
        print(f"数据生成完成: train={n_train}, val={n_val}")
        print(f"最小标记距离: {min_dist}")
        print(f"目标值范围: [{y_train.min():.3f}, {y_train.max():.3f}]")
        
        results[f'S={seq_len}'] = {}
        
        # 1. HormonicFormer
        print(f"\n  [1/3] HormonicFormer (d={d_model}, L={n_layers})")
        try:
            model = AddingHormonic(d_model, n_layers, n_heads, seq_len)
            params = sum(p.numel() for p in model.parameters())
            print(f"    参数量: {params:,}")
            result = train_model(model, X_train, y_train, X_val, y_val, epochs)
            results[f'S={seq_len}']['hormonic'] = {
                'best_val_loss': result['best_val_loss'],
                'baseline_mse': result['baseline_mse'],
                'relative_improvement': result['relative_improvement'],
                'params': params
            }
            print(f"    [OK] Val MSE={result['best_val_loss']:.6f}, 相对提升={result['relative_improvement']:.1f}%")
        except Exception as e:
            print(f"    [FAIL] {e}")
            results[f'S={seq_len}']['hormonic'] = {'error': str(e)}
        
        # 2. Transformer
        print(f"\n  [2/3] Transformer (d={d_model}, L={n_layers})")
        try:
            model = AddingTransformer(d_model, n_layers, n_heads, seq_len)
            params = sum(p.numel() for p in model.parameters())
            print(f"    参数量: {params:,}")
            result = train_model(model, X_train, y_train, X_val, y_val, epochs)
            results[f'S={seq_len}']['transformer'] = {
                'best_val_loss': result['best_val_loss'],
                'baseline_mse': result['baseline_mse'],
                'relative_improvement': result['relative_improvement'],
                'params': params
            }
            print(f"    [OK] Val MSE={result['best_val_loss']:.6f}, 相对提升={result['relative_improvement']:.1f}%")
        except Exception as e:
            print(f"    [FAIL] {e}")
            results[f'S={seq_len}']['transformer'] = {'error': str(e)}
        
        # 3. LSTM
        print(f"\n  [3/3] LSTM (d={d_model}, L={n_layers})")
        try:
            model = AddingLSTM(d_model, n_layers)
            params = sum(p.numel() for p in model.parameters())
            print(f"    参数量: {params:,}")
            result = train_model(model, X_train, y_train, X_val, y_val, epochs)
            results[f'S={seq_len}']['lstm'] = {
                'best_val_loss': result['best_val_loss'],
                'baseline_mse': result['baseline_mse'],
                'relative_improvement': result['relative_improvement'],
                'params': params
            }
            print(f"    [OK] Val MSE={result['best_val_loss']:.6f}, 相对提升={result['relative_improvement']:.1f}%")
        except Exception as e:
            print(f"    [FAIL] {e}")
            results[f'S={seq_len}']['lstm'] = {'error': str(e)}
    
    # 保存结果
    output_file = f'C:/Users/MR/Desktop/论文/关于场物理的神经框架/研究论文数据/adding_problem_results_{datetime.now().strftime("%m%d_%H%M")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("实验完成!")
    print(f"结果保存: {output_file}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("结果汇总 (Val MSE, 越低越好)")
    print("=" * 70)
    print(f"{'Seq Len':<12} {'Hormonic':<15} {'Transformer':<15} {'LSTM':<15}")
    print("-" * 60)
    for key in results:
        h = results[key].get('hormonic', {}).get('best_val_loss', 'N/A')
        t = results[key].get('transformer', {}).get('best_val_loss', 'N/A')
        l = results[key].get('lstm', {}).get('best_val_loss', 'N/A')
        
        h_str = f"{h:.6f}" if isinstance(h, float) else str(h)
        t_str = f"{t:.6f}" if isinstance(t, float) else str(t)
        l_str = f"{l:.6f}" if isinstance(l, float) else str(l)
        
        print(f"{key:<12} {h_str:<15} {t_str:<15} {l_str:<15}")
    
    print("\n相对提升 (vs 基线预测均值):")
    print(f"{'Seq Len':<12} {'Hormonic':<15} {'Transformer':<15} {'LSTM':<15}")
    print("-" * 60)
    for key in results:
        h = results[key].get('hormonic', {}).get('relative_improvement', 'N/A')
        t = results[key].get('transformer', {}).get('relative_improvement', 'N/A')
        l = results[key].get('lstm', {}).get('relative_improvement', 'N/A')
        
        h_str = f"{h:.1f}%" if isinstance(h, float) else str(h)
        t_str = f"{t:.1f}%" if isinstance(t, float) else str(t)
        l_str = f"{l:.1f}%" if isinstance(l, float) else str(l)
        
        print(f"{key:<12} {h_str:<15} {t_str:<15} {l_str:<15}")
    
    return results

if __name__ == '__main__':
    run_experiment()
