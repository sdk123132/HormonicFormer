"""
HormonicFormer 真实对比实验 - Adding Problem + ETT
使用真实 HormonicFormerV7r3（CGL场演化 + Hebbian学习 + STP + PAC）
vs 标准 Transformer vs LSTM
"""

import sys
sys.path.insert(0, r'C:\Users\MR\Desktop')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from datetime import datetime
from hormonic_v7r3_validated import HormonicFormerV7r3, HormonicBlockV7r3

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{datetime.now().strftime('%H:%M:%S')}] 设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============== 数据生成 ==============
def generate_adding_data(n_samples, seq_len, min_dist=None):
    """Adding Problem数据生成"""
    X = np.zeros((n_samples, seq_len, 2), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    for i in range(n_samples):
        X[i, :, 0] = np.random.uniform(0, 1, seq_len)
        if min_dist is None:
            positions = np.random.choice(seq_len, 2, replace=False)
        else:
            pos1 = np.random.randint(0, seq_len // 4)
            pos2 = np.random.randint(max(pos1 + min_dist, seq_len * 3 // 4), seq_len)
            positions = [pos1, pos2]
        X[i, positions[0], 1] = 1.0
        X[i, positions[1], 1] = 1.0
        y[i, 0] = X[i, positions[0], 0] + X[i, positions[1], 0]
    return X, y

def generate_ett_data(n_samples=10000, n_features=7):
    """ETT风格周期性数据"""
    np.random.seed(42)
    t = np.arange(n_samples)
    data = np.zeros((n_samples, n_features))
    daily = 10 * np.sin(2 * np.pi * t / 24)
    weekly = 5 * np.sin(2 * np.pi * t / 168)
    trend = 0.001 * t
    noise = np.random.randn(n_samples) * 2
    data[:, 0] = 50 + daily + weekly + trend + noise
    for i in range(1, n_features):
        phase = np.random.uniform(0, 2*np.pi)
        amp = np.random.uniform(5, 15)
        data[:, i] = 50 + amp * np.sin(2 * np.pi * t / 24 + phase) + np.random.randn(n_samples) * np.random.uniform(1, 3)
    return data.astype(np.float32)

def create_ett_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, 0])
    return np.array(X), np.array(y)

# ============== 真实HormonicFormer包装器 ==============
class HormonicAdding(nn.Module):
    """使用真实HormonicBlockV7r3的Adding Problem模型"""
    def __init__(self, d_model, n_layers, seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 输入投影: 2维 -> d_model*2（实部+虚部）
        self.input_proj = nn.Linear(2, d_model * 2)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model * 2) * 0.02)
        
        # 真实HormonicFormer配置
        config = {
            'model': {
                'd_model': d_model,
                'seq_len': seq_len,
                'n_layers': n_layers,
                'n_heads': 4,
                'n_cgl_steps': 5,
                'D0_amp': 0.002,
                'D0_phase': 0.002,
                'cgl_dt': 0.02,
                'noise_scale': 0.001,
                'dropout': 0.1,
            },
            'use_neuromod': True,
            'use_pac': n_layers > 1,
            'use_pc': False,
            'g_coupling_strength': 0.1,
            'hebbian': {},
            'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05},
        }
        
        # 真实HormonicFormer块
        self.blocks = nn.ModuleList([
            HormonicBlockV7r3(d_model, seq_len, config)
            for _ in range(n_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model * 2)
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x):
        # x: [B, seq_len, 2]
        B, S, _ = x.shape
        
        # 投影到复数表示
        h = self.input_proj(x) + self.pos_embed[:, :S, :]  # [B, S, d_model*2]
        
        # 转换为psi: [B, S, D, 2]
        psi = h.reshape(B, S, self.d_model, 2)
        
        # 通过真实HormonicFormer块
        for block in self.blocks:
            psi = block(psi)
        
        # 输出
        h = self.output_norm(psi.reshape(B, S, self.d_model * 2))
        h = h.mean(dim=1)  # 全局池化
        return self.predictor(h)

class HormonicETT(nn.Module):
    """使用真实HormonicBlockV7r3的ETT时序预测模型"""
    def __init__(self, input_dim, d_model, n_layers, seq_len, pred_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        self.input_proj = nn.Linear(input_dim, d_model * 2)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model * 2) * 0.02)
        
        config = {
            'model': {
                'd_model': d_model,
                'seq_len': seq_len,
                'n_layers': n_layers,
                'n_heads': 4,
                'n_cgl_steps': 5,
                'D0_amp': 0.002,
                'D0_phase': 0.002,
                'cgl_dt': 0.02,
                'noise_scale': 0.001,
                'dropout': 0.1,
            },
            'use_neuromod': True,
            'use_pac': n_layers > 1,
            'use_pc': False,
            'g_coupling_strength': 0.1,
            'hebbian': {},
            'stp': {'U': 0.2, 'tau_f': 1.0, 'tau_d': 3.0, 'dt': 0.05},
        }
        
        self.blocks = nn.ModuleList([
            HormonicBlockV7r3(d_model, seq_len, config)
            for _ in range(n_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model * 2)
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, pred_len)
        )
    
    def forward(self, x):
        B, S, _ = x.shape
        h = self.input_proj(x) + self.pos_embed[:, :S, :]
        psi = h.reshape(B, S, self.d_model, 2)
        for block in self.blocks:
            psi = block(psi)
        h = self.output_norm(psi.reshape(B, S, self.d_model * 2))
        h = h.mean(dim=1)
        return self.predictor(h)

# ============== 标准Transformer基线 ==============
class TransformerAdding(nn.Module):
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
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        return self.predictor(x.mean(dim=1))

class TransformerETT(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, seq_len, pred_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_model // 2, pred_len)
        )
    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        return self.predictor(x.mean(dim=1))

# ============== LSTM基线 ==============
class LSTMAdding(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        return self.predictor(x[:, -1, :])

class LSTMETT(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, pred_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_model // 2, pred_len)
        )
    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        return self.predictor(x[:, -1, :])

# ============== 训练函数 ==============
def train_regression(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, lr=1e-3):
    """训练回归模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
            history['val_loss'].append(val_loss)
        
        if val_loss < best_val:
            best_val = val_loss
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Train={avg_loss:.6f}, Val={val_loss:.6f}")
    
    baseline_mse = float(((y_val - y_val.mean()) ** 2).mean())
    history['best_val_loss'] = best_val
    history['baseline_mse'] = baseline_mse
    history['relative_improvement'] = float(1 - best_val / baseline_mse) * 100
    return history

# ============== 主实验 ==============
def run_adding_experiment():
    """Adding Problem距离扫描"""
    print("\n" + "=" * 70)
    print("实验1: Adding Problem 距离扫描（真实HormonicFormer）")
    print("=" * 70)
    
    seq_lengths = [50, 100, 200, 500]
    d_model = 32  # HormonicFormer用d_model=32（实际嵌入维度=64因为实部+虚部）
    n_layers = 2
    n_heads = 4
    n_train = 5000
    n_val = 1000
    epochs = 50
    
    results = {}
    
    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"序列长度: S={seq_len}")
        print(f"{'='*60}")
        
        min_dist = seq_len // 2
        X_train, y_train = generate_adding_data(n_train, seq_len, min_dist)
        X_val, y_val = generate_adding_data(n_val, seq_len, min_dist)
        
        print(f"数据: train={n_train}, val={n_val}, min_dist={min_dist}")
        
        results[f'S={seq_len}'] = {}
        
        # 1. 真实HormonicFormer
        print(f"\n  [1/3] HormonicFormer (真实CGL+Hebbian+STP, d={d_model})")
        try:
            model = HormonicAdding(d_model, n_layers, seq_len)
            params = sum(p.numel() for p in model.parameters())
            print(f"    参数量: {params:,}")
            result = train_regression(model, X_train, y_train, X_val, y_val, epochs)
            results[f'S={seq_len}']['hormonic'] = {
                'best_val_loss': result['best_val_loss'],
                'baseline_mse': result['baseline_mse'],
                'relative_improvement': result['relative_improvement'],
                'params': params
            }
            print(f"    [OK] Val MSE={result['best_val_loss']:.6f}, 提升={result['relative_improvement']:.1f}%")
        except Exception as e:
            print(f"    [FAIL] {e}")
            import traceback; traceback.print_exc()
            results[f'S={seq_len}']['hormonic'] = {'error': str(e)}
        
        # 2. 标准Transformer（同d_model=64匹配HormonicFormer的嵌入维度）
        print(f"\n  [2/3] Transformer (标准, d={d_model*2})")
        try:
            model = TransformerAdding(d_model * 2, n_layers, n_heads, seq_len)
            params = sum(p.numel() for p in model.parameters())
            print(f"    参数量: {params:,}")
            result = train_regression(model, X_train, y_train, X_val, y_val, epochs)
            results[f'S={seq_len}']['transformer'] = {
                'best_val_loss': result['best_val_loss'],
                'baseline_mse': result['baseline_mse'],
                'relative_improvement': result['relative_improvement'],
                'params': params
            }
            print(f"    [OK] Val MSE={result['best_val_loss']:.6f}, 提升={result['relative_improvement']:.1f}%")
        except Exception as e:
            print(f"    [FAIL] {e}")
            results[f'S={seq_len}']['transformer'] = {'error': str(e)}
        
        # 3. LSTM
        print(f"\n  [3/3] LSTM (d={d_model*2})")
        try:
            model = LSTMAdding(d_model * 2, n_layers)
            params = sum(p.numel() for p in model.parameters())
            print(f"    参数量: {params:,}")
            result = train_regression(model, X_train, y_train, X_val, y_val, epochs)
            results[f'S={seq_len}']['lstm'] = {
                'best_val_loss': result['best_val_loss'],
                'baseline_mse': result['baseline_mse'],
                'relative_improvement': result['relative_improvement'],
                'params': params
            }
            print(f"    [OK] Val MSE={result['best_val_loss']:.6f}, 提升={result['relative_improvement']:.1f}%")
        except Exception as e:
            print(f"    [FAIL] {e}")
            results[f'S={seq_len}']['lstm'] = {'error': str(e)}
    
    return results

def run_ett_experiment():
    """ETT时序预测"""
    print("\n" + "=" * 70)
    print("实验2: ETT时序预测（真实HormonicFormer）")
    print("=" * 70)
    
    data = generate_ett_data(n_samples=10000, n_features=7)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / (data_std + 1e-8)
    
    n_train = int(len(data) * 0.7)
    train_data = data[:n_train]
    val_data = data[n_train:n_train+int(len(data)*0.2)]
    
    configs = [
        {'seq_len': 48, 'pred_len': 12, 'name': '2d->12h'},
        {'seq_len': 96, 'pred_len': 24, 'name': '4d->24h'},
        {'seq_len': 168, 'pred_len': 48, 'name': '7d->48h'},
    ]
    
    d_model = 32
    n_layers = 2
    n_heads = 4
    input_dim = 7
    
    results = {}
    
    for cfg in configs:
        seq_len = cfg['seq_len']
        pred_len = cfg['pred_len']
        name = cfg['name']
        
        print(f"\n{'='*60}")
        print(f"配置: {name} (输入={seq_len}, 预测={pred_len})")
        print(f"{'='*60}")
        
        X_train, y_train = create_ett_sequences(train_data, seq_len, pred_len)
        X_val, y_val = create_ett_sequences(val_data, seq_len, pred_len)
        
        print(f"训练: {len(X_train)}, 验证: {len(X_val)}")
        
        results[name] = {}
        
        # 1. 真实HormonicFormer
        print(f"\n  [1/3] HormonicFormer (真实CGL+Hebbian+STP)")
        try:
            model = HormonicETT(input_dim, d_model, n_layers, seq_len, pred_len)
            params = sum(p.numel() for p in model.parameters())
            print(f"    参数量: {params:,}")
            result = train_regression(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
            results[name]['hormonic'] = {
                'best_val_loss': result['best_val_loss'],
                'params': params
            }
            print(f"    [OK] Best Val Loss: {result['best_val_loss']:.4f}")
        except Exception as e:
            print(f"    [FAIL] {e}")
            import traceback; traceback.print_exc()
            results[name]['hormonic'] = {'error': str(e)}
        
        # 2. Transformer
        print(f"\n  [2/3] Transformer (d={d_model*2})")
        try:
            model = TransformerETT(input_dim, d_model * 2, n_layers, n_heads, seq_len, pred_len)
            params = sum(p.numel() for p in model.parameters())
            print(f"    参数量: {params:,}")
            result = train_regression(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
            results[name]['transformer'] = {
                'best_val_loss': result['best_val_loss'],
                'params': params
            }
            print(f"    [OK] Best Val Loss: {result['best_val_loss']:.4f}")
        except Exception as e:
            print(f"    [FAIL] {e}")
            results[name]['transformer'] = {'error': str(e)}
        
        # 3. LSTM
        print(f"\n  [3/3] LSTM (d={d_model*2})")
        try:
            model = LSTMETT(input_dim, d_model * 2, n_layers, pred_len)
            params = sum(p.numel() for p in model.parameters())
            print(f"    参数量: {params:,}")
            result = train_regression(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
            results[name]['lstm'] = {
                'best_val_loss': result['best_val_loss'],
                'params': params
            }
            print(f"    [OK] Best Val Loss: {result['best_val_loss']:.4f}")
        except Exception as e:
            print(f"    [FAIL] {e}")
            results[name]['lstm'] = {'error': str(e)}
    
    return results

# ============== 运行所有 ==============
if __name__ == '__main__':
    all_results = {}
    
    # Adding Problem
    all_results['adding_problem'] = run_adding_experiment()
    
    # ETT
    all_results['ett_prediction'] = run_ett_experiment()
    
    # 保存
    output_file = f'C:/Users/MR/Desktop/论文/关于场物理的神经框架/研究论文数据/real_hormonic_results_{datetime.now().strftime("%m%d_%H%M")}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("所有实验完成!")
    print(f"结果保存: {output_file}")
    print(f"{'='*70}")
    
    # 汇总
    print("\n" + "=" * 70)
    print("Adding Problem 汇总 (Val MSE)")
    print("=" * 70)
    print(f"{'Seq Len':<12} {'Hormonic':<15} {'Transformer':<15} {'LSTM':<15}")
    print("-" * 60)
    for key in all_results.get('adding_problem', {}):
        h = all_results['adding_problem'][key].get('hormonic', {}).get('best_val_loss', 'N/A')
        t = all_results['adding_problem'][key].get('transformer', {}).get('best_val_loss', 'N/A')
        l = all_results['adding_problem'][key].get('lstm', {}).get('best_val_loss', 'N/A')
        h_str = f"{h:.6f}" if isinstance(h, float) else str(h)
        t_str = f"{t:.6f}" if isinstance(t, float) else str(t)
        l_str = f"{l:.6f}" if isinstance(l, float) else str(l)
        print(f"{key:<12} {h_str:<15} {t_str:<15} {l_str:<15}")
    
    print("\n" + "=" * 70)
    print("ETT时序预测 汇总 (Val MSE)")
    print("=" * 70)
    print(f"{'Config':<15} {'Hormonic':<12} {'Transformer':<12} {'LSTM':<12}")
    print("-" * 55)
    for key in all_results.get('ett_prediction', {}):
        h = all_results['ett_prediction'][key].get('hormonic', {}).get('best_val_loss', 'N/A')
        t = all_results['ett_prediction'][key].get('transformer', {}).get('best_val_loss', 'N/A')
        l = all_results['ett_prediction'][key].get('lstm', {}).get('best_val_loss', 'N/A')
        h_str = f"{h:.4f}" if isinstance(h, float) else str(h)
        t_str = f"{t:.4f}" if isinstance(t, float) else str(t)
        l_str = f"{l:.4f}" if isinstance(l, float) else str(l)
        print(f"{key:<15} {h_str:<12} {t_str:<12} {l_str:<12}")
    
    # 参数量对比
    print("\n" + "=" * 70)
    print("参数量对比")
    print("=" * 70)
    for task in all_results:
        for cfg in all_results[task]:
            print(f"\n{task} - {cfg}:")
            for model_name in all_results[task][cfg]:
                p = all_results[task][cfg][model_name].get('params', 'N/A')
                p_str = f"{p:,}" if isinstance(p, int) else str(p)
                print(f"  {model_name}: {p_str}")
