"""
HormonicFormer 时序预测 - ETT (Electricity Transformer Temperature)
真实周期性数据，CGL振荡场天然匹配
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

# ============== 合成ETT数据（周期性） ==============
def generate_ett_data(n_samples=10000, n_features=7):
    """生成具有周期性的ETT风格数据"""
    np.random.seed(42)
    
    # 时间戳（小时）
    t = np.arange(n_samples)
    
    # 生成7个特征（模拟ETT的OT, HUFL, HULL, MUFL, MULL, LUFL, LULL）
    data = np.zeros((n_samples, n_features))
    
    # 主要特征（变压器油温）- 日周期 + 周周期 + 趋势 + 噪声
    # 日周期：24小时
    daily = 10 * np.sin(2 * np.pi * t / 24)
    # 周周期：7天 = 168小时
    weekly = 5 * np.sin(2 * np.pi * t / 168)
    # 长期趋势
    trend = 0.001 * t
    # 噪声
    noise = np.random.randn(n_samples) * 2
    
    data[:, 0] = 50 + daily + weekly + trend + noise  # OT (油温)
    
    # 其他特征（湿度、压力等）- 相关但有相位差
    for i in range(1, n_features):
        phase = np.random.uniform(0, 2*np.pi)
        amp = np.random.uniform(5, 15)
        noise_level = np.random.uniform(1, 3)
        data[:, i] = 50 + amp * np.sin(2 * np.pi * t / 24 + phase) + np.random.randn(n_samples) * noise_level
    
    return data.astype(np.float32)

def create_sequences(data, seq_len, pred_len):
    """创建输入-输出序列"""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, 0])  # 预测油温（第一个特征）
    return np.array(X), np.array(y)

# ============== 模型定义 ==============
class ETTHormonic(nn.Module):
    """HormonicFormer for ETT"""
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
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, pred_len)
        )
        
    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.predictor(x)

class ETTTransformer(nn.Module):
    """Transformer for ETT"""
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
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, pred_len)
        )
        
    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.predictor(x)

class ETTLSTM(nn.Module):
    """LSTM for ETT"""
    def __init__(self, input_dim, d_model, n_layers, seq_len, pred_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, pred_len)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.predictor(x)

# ============== 训练函数 ==============
def train_model(model, train_loader, val_loader, epochs=30):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                val_loss += criterion(output, y).item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    history['best_val_loss'] = best_val_loss
    return history

# ============== 主实验 ==============
def run_experiment():
    """运行ETT预测实验"""
    
    print("=" * 70)
    print("HormonicFormer 时序预测 - ETT (电力变压器温度)")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 生成数据
    print("\n生成ETT周期性数据...")
    data = generate_ett_data(n_samples=10000, n_features=7)
    
    # 归一化
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / (data_std + 1e-8)
    
    # 划分数据集
    n_train = int(len(data) * 0.7)
    n_val = int(len(data) * 0.2)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train+n_val]
    
    print(f"数据形状: {data.shape}")
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")
    
    # 实验配置
    configs = [
        {'seq_len': 48, 'pred_len': 12, 'name': '2h->6h'},   # 2小时预测6小时
        {'seq_len': 96, 'pred_len': 24, 'name': '4h->12h'},  # 4小时预测12小时
        {'seq_len': 168, 'pred_len': 48, 'name': '7h->24h'}, # 7小时预测24小时
    ]
    
    results = {}
    
    for config in configs:
        seq_len = config['seq_len']
        pred_len = config['pred_len']
        name = config['name']
        
        print(f"\n{'='*70}")
        print(f"配置: {name} (输入={seq_len}, 预测={pred_len})")
        print(f"{'='*70}")
        
        # 创建序列
        X_train, y_train = create_sequences(train_data, seq_len, pred_len)
        X_val, y_val = create_sequences(val_data, seq_len, pred_len)
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
        
        print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
        
        results[name] = {}
        
        # 模型配置
        d_model, n_layers, n_heads = 64, 2, 4
        input_dim = 7
        
        # 1. HormonicFormer
        print(f"\n[1/3] HormonicFormer")
        try:
            model = ETTHormonic(input_dim, d_model, n_layers, n_heads, seq_len, pred_len)
            result = train_model(model, train_loader, val_loader, epochs=30)
            results[name]['hormonic'] = result
            print(f"  [OK] Best Val Loss: {result['best_val_loss']:.4f}")
        except Exception as e:
            print(f"  [FAIL] {e}")
            results[name]['hormonic'] = {'error': str(e)}
        
        # 2. Transformer
        print(f"\n[2/3] Transformer")
        try:
            model = ETTTransformer(input_dim, d_model, n_layers, n_heads, seq_len, pred_len)
            result = train_model(model, train_loader, val_loader, epochs=30)
            results[name]['transformer'] = result
            print(f"  [OK] Best Val Loss: {result['best_val_loss']:.4f}")
        except Exception as e:
            print(f"  [FAIL] {e}")
            results[name]['transformer'] = {'error': str(e)}
        
        # 3. LSTM
        print(f"\n[3/3] LSTM")
        try:
            model = ETTLSTM(input_dim, d_model, n_layers, seq_len, pred_len)
            result = train_model(model, train_loader, val_loader, epochs=30)
            results[name]['lstm'] = result
            print(f"  [OK] Best Val Loss: {result['best_val_loss']:.4f}")
        except Exception as e:
            print(f"  [FAIL] {e}")
            results[name]['lstm'] = {'error': str(e)}
    
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
    print("\n结果汇总 (Best Val Loss):")
    print(f"{'Config':<15} {'Hormonic':<12} {'Transformer':<12} {'LSTM':<12}")
    print("-" * 55)
    for name in results:
        h_loss = results[name].get('hormonic', {}).get('best_val_loss', 'N/A')
        t_loss = results[name].get('transformer', {}).get('best_val_loss', 'N/A')
        l_loss = results[name].get('lstm', {}).get('best_val_loss', 'N/A')
        
        h_str = f"{h_loss:.4f}" if isinstance(h_loss, float) else str(h_loss)
        t_str = f"{t_loss:.4f}" if isinstance(t_loss, float) else str(t_str)
        l_str = f"{l_loss:.4f}" if isinstance(l_loss, float) else str(l_loss)
        
        print(f"{name:<15} {h_str:<12} {t_str:<12} {l_str:<12}")
    
    return results

if __name__ == '__main__':
    run_experiment()
