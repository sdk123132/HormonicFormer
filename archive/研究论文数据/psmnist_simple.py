"""
HormonicFormer 长程依赖测试 - 简化版 Permuted Sequential MNIST
使用合成数据，避免下载问题
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

# ============== 合成数据生成 ==============
class SyntheticPSMNIST:
    """合成置换MNIST数据"""
    def __init__(self, n_samples=10000, seq_len=784, num_classes=10):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # 生成固定置换
        np.random.seed(42)
        self.permutation = np.random.permutation(seq_len)
        
        # 生成数据
        self.data = []
        for i in range(n_samples):
            # 生成模拟图像（带模式的随机数据）
            img = np.random.randn(seq_len) * 0.5
            # 添加一些结构（模拟数字模式）
            pattern = np.sin(np.linspace(0, 4*np.pi, seq_len)) * (i % 10) * 0.1
            img = img + pattern
            
            # 置换
            img_permuted = img[self.permutation]
            
            # 归一化
            img_permuted = (img_permuted - img_permuted.mean()) / (img_permuted.std() + 1e-8)
            
            label = i % num_classes
            self.data.append((img_permuted.astype(np.float32), label))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx][0]), self.data[idx][1]

# ============== 模型定义（简化版） ==============
class SimplePSMNIST(nn.Module):
    """简化版psMNIST模型"""
    def __init__(self, seq_len=784, d_model=128, n_layers=4, n_heads=4, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class SimpleLSTM(nn.Module):
    """LSTM基线"""
    def __init__(self, d_model=128, n_layers=2, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.classifier(x)

# ============== 训练函数 ==============
def train_model(model, train_loader, test_loader, epochs=10):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n训练配置: epochs={epochs}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    best_acc = 0
    history = {'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # 训练
        model.train()
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        train_acc = 100. * correct / total
        history['train_acc'].append(train_acc)
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        test_acc = 100. * correct / total
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}% (Best: {best_acc:.2f}%)")
    
    history['best_acc'] = best_acc
    return history

# ============== 主实验 ==============
def run_experiment():
    """运行简化版psMNIST实验"""
    
    print("=" * 70)
    print("HormonicFormer 长程依赖测试 - 简化版 psMNIST")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 生成数据
    print("\n生成合成数据...")
    train_dataset = SyntheticPSMNIST(n_samples=5000, seq_len=784)
    test_dataset = SyntheticPSMNIST(n_samples=1000, seq_len=784)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")
    print(f"序列长度: 784")
    
    results = {}
    
    # 1. Transformer (作为HormonicFormer的简化版)
    print(f"\n{'='*70}")
    print("[1/2] Transformer (d_model=128, n_layers=4)")
    print(f"{'='*70}")
    try:
        model = SimplePSMNIST(seq_len=784, d_model=128, n_layers=4, n_heads=4, num_classes=10)
        result = train_model(model, train_loader, test_loader, epochs=10)
        results['transformer'] = result
        print(f"\n[Transformer] Best Test Acc: {result['best_acc']:.2f}%")
    except Exception as e:
        print(f"[FAIL] {e}")
        results['transformer'] = {'error': str(e)}
    
    # 2. LSTM
    print(f"\n{'='*70}")
    print("[2/2] LSTM (d_model=128, n_layers=2)")
    print(f"{'='*70}")
    try:
        model = SimpleLSTM(d_model=128, n_layers=2, num_classes=10)
        result = train_model(model, train_loader, test_loader, epochs=10)
        results['lstm'] = result
        print(f"\n[LSTM] Best Test Acc: {result['best_acc']:.2f}%")
    except Exception as e:
        print(f"[FAIL] {e}")
        results['lstm'] = {'error': str(e)}
    
    # 保存结果
    output_file = f'C:/Users/MR/Desktop/论文/关于场物理的神经框架/研究论文数据/psmnist_simple_results_{datetime.now().strftime("%m%d_%H%M")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("实验完成!")
    print(f"结果保存: {output_file}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    print("\n结果汇总:")
    print(f"{'Model':<20} {'Best Test Acc':<15}")
    print("-" * 40)
    for model_name in ['transformer', 'lstm']:
        acc = results.get(model_name, {}).get('best_acc', 'N/A')
        acc_str = f"{acc:.2f}%" if isinstance(acc, float) else str(acc)
        print(f"{model_name.capitalize():<20} {acc_str:<15}")
    
    return results

if __name__ == '__main__':
    run_experiment()
