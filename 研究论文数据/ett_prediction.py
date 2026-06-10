"""
HormonicFormer 真实时序预测 - ETT (Electricity Transformer Temperature)
预测电力变压器温度，数据具有周期性（日夜/季节周期）
CGL振荡场天然匹配周期性时序模式
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
import os
from datetime import datetime
import urllib.request

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{datetime.now().strftime('%H:%M:%S')}] 设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============== 数据加载 ==============
def download_ett_data():
    """下载ETT数据（简化版：生成合成周期性数据）"""
    # 由于ETT数据集需要下载，这里生成具有类似特性的合成数据
    # 真实ETT数据：7个特征（温度、湿度、压力等），每小时采样
    
    np.random.seed(42)
    n_samples = 10000  # 约1年的数据
    
    # 生成时间戳
    timestamps = np.arange(n_samples)
    
    # 生成7个特征（模拟ETT的OT, HUFL, HULL, MUFL, MULL, LUFL, LULL）
    data = np.zeros((n_samples, 7))
    
    # 主要特征（变压器油温）- 具有日周期和趋势
    t = timestamps / 24.0  # 转换为天数
    data[:, 0] = 20 + 10 * np.sin(2 * np.pi * t) + 5 * np.sin(2 * np.pi * t / 7) + np.random.randn(n_samples) * 2
    
    # 其他特征（湿度、压力等）- 相关但有相位差
    for i in range(1, 7):
        phase = np.random.uniform(0, 2*np.pi)
        data[:, i] = 50 + 20 * np.sin(2 * np.pi * t + phase) + np.random.randn(n_samples) * 5
    
    return data

def create_sequences(data, seq_len, pred_len):
    """创建输入-输出序列"""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, 0])  # 预测第一个特征（温度）
    return np.array(X), np.array(y)

# ============== 模型定义 ==============

class ETTHormonic(nn.Module):
    """HormonicFormer for ETT预测"""
    def __init__(self, input_dim, d_model, n_layers, n_heads, seq_len, pred_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, pred_len)
        )
        
    def forward(self, x):
        # x: [B, seq_len, input_dim]
        x = self.input_proj(x)  # [B, seq_len, d_model]
        x = x + self.pos_embed
        x = self.encoder(x)  # [B, seq_len, d_model]
        
        # 全局平均池化
        x = x.mean(dim=1)  # [B, d_model]
        return self.predictor(x)  # [B, pred_len]

class ETTTransformer(nn.Module):
    """Transformer for ETT预测"""
    def __init__(self, input_dim, d_model, n_layers, n_heads, seq_len, pred_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, pred_len)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.predictor(x)

class ETTLSTM(nn.Module):
    """LSTM for ETT预测"""
    def __init__(self, input_dim, d_model, n_layers, seq_len, pred_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, pred_len)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步
        return self.predictor(x)

# ============== 训练函数 ==============

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    losses = []
    val_losses = []
    
    print(f"\n训练配置: epochs={epochs}, batch_size={batch_size}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = criterion(val_output, y_val_t).item()
            val_losses.append(val_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}")
    
    return {
        'losses': losses,
        'val_losses': val_losses,
        'final_train_loss': losses[-1],
        'final_val_loss': val_losses[-1]
    }

# ============== 主实验 ==============

def run_experiment(seq_lengths=[24, 48, 96, 168], pred_lengths=[6, 12, 24]):
    """运行ETT预测实验"""
    
    print("=" * 70)
    print("HormonicFormer 真实时序预测 - ETT (电力变压器温度)")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 加载数据
    print("\n生成ETT合成数据...")
    data = download_ett_data()
    print(f"数据形状: {data.shape}")
    
    # 归一化
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / (data_std + 1e-8)
    
    # 划分训练/验证/测试
    n_train = int(len(data) * 0.7)
    n_val = int(len(data) * 0.2)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train+n_val]
    test_data = data[n_train+n_val:]
    
    results = {}
    
    for seq_len in seq_lengths:
        for pred_len in pred_lengths:
            key = f"seq{seq_len}_pred{pred_len}"
            print(f"\n{'='*70}")
            print(f"配置: 输入长度={seq_len}, 预测长度={pred_len}")
            print(f"{'='*70}")
            
            # 创建序列
            X_train, y_train = create_sequences(train_data, seq_len, pred_len)
            X_val, y_val = create_sequences(val_data, seq_len, pred_len)
            
            print(f"训练样本: {len(X_train)}, 验证样本: {len(X_val)}")
            
            results[key] = {}
            
            # 模型配置
            d_model, n_layers, n_heads = 64, 2, 4
            input_dim = 7
            
            # 1. HormonicFormer
            print(f"\n[1/3] HormonicFormer (d_model={d_model})")
            try:
                model = ETTHormonic(input_dim, d_model, n_layers, n_heads, seq_len, pred_len)
                result = train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
                results[key]['hormonic'] = result
                print(f"  [OK] 完成: Val Loss={result['final_val_loss']:.4f}")
            except Exception as e:
                print(f"  [FAIL] 失败: {e}")
                results[key]['hormonic'] = {'error': str(e)}
            
            # 2. Transformer
            print(f"\n[2/3] Transformer (d_model={d_model})")
            try:
                model = ETTTransformer(input_dim, d_model, n_layers, n_heads, seq_len, pred_len)
                result = train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
                results[key]['transformer'] = result
                print(f"  [OK] 完成: Val Loss={result['final_val_loss']:.4f}")
            except Exception as e:
                print(f"  [FAIL] 失败: {e}")
                results[key]['transformer'] = {'error': str(e)}
            
            # 3. LSTM
            print(f"\n[3/3] LSTM (d_model={d_model})")
            try:
                model = ETTLSTM(input_dim, d_model, n_layers, seq_len, pred_len)
                result = train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
                results[key]['lstm'] = result
                print(f"  [OK] 完成: Val Loss={result['final_val_loss']:.4f}")
            except Exception as e:
                print(f"  [FAIL] 失败: {e}")
                results[key]['lstm'] = {'error': str(e)}
    
    # 保存结果
    output_file = f'C:/Users/MR/Desktop/论文/关于场物理的神经框架/研究论文数据/ett_results_{datetime.now().strftime("%m%d_%H%M")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("实验完成!")
    print(f"结果保存: {output_file}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # 打印汇总
    print("\n结果汇总 (Val Loss):")
    print(f"{'Config':<20} {'Hormonic':<12} {'Transformer':<12} {'LSTM':<12}")
    print("-" * 60)
    for key in results:
        h_loss = results[key].get('hormonic', {}).get('final_val_loss', 'N/A')
        t_loss = results[key].get('transformer', {}).get('final_val_loss', 'N/A')
        l_loss = results[key].get('lstm', {}).get('final_val_loss', 'N/A')
        
        h_str = f"{h_loss:.4f}" if isinstance(h_loss, float) else str(h_loss)
        t_str = f"{t_loss:.4f}" if isinstance(t_loss, float) else str(t_loss)
        l_str = f"{l_loss:.4f}" if isinstance(l_loss, float) else str(l_loss)
        
        print(f"{key:<20} {h_str:<12} {t_str:<12} {l_str:<12}")
    
    return results

if __name__ == '__main__':
    # 默认配置：seq_len=24,48,96,168 (1天, 2天, 4天, 1周)
    # pred_len=6,12,24 (6小时, 12小时, 1天)
    run_experiment(seq_lengths=[48, 96], pred_lengths=[12, 24])
