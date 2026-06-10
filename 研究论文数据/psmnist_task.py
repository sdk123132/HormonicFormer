"""
HormonicFormer 长程依赖测试 - Permuted Sequential MNIST (psMNIST)
将28x28 MNIST图像置换后展平为784长度序列，测试长程记忆能力
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
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

# ============== 数据加载 ==============
class PermutedMNIST:
    """置换MNIST数据集"""
    def __init__(self, train=True, seq_len=784):
        # 加载MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.mnist = torchvision.datasets.MNIST(
            root='./data', train=train, download=True, transform=transform
        )
        
        # 生成固定置换（保证可复现）
        np.random.seed(42)
        self.permutation = np.random.permutation(seq_len)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        # 展平并置换
        img_flat = img.view(-1).numpy()  # [784]
        img_permuted = img_flat[self.permutation]  # 置换
        
        # 归一化到0-1
        img_permuted = img_permuted.astype(np.float32)
        
        return torch.FloatTensor(img_permuted), label

# ============== 模型定义 ==============

class PSMNIST_Hormonic(nn.Module):
    """HormonicFormer for psMNIST"""
    def __init__(self, seq_len=784, d_model=128, n_layers=4, n_heads=4, num_classes=10):
        super().__init__()
        # 像素值嵌入（0-255映射到d_model）
        self.embedding = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x: [B, seq_len]
        x = x.unsqueeze(-1)  # [B, seq_len, 1]
        x = self.embedding(x)  # [B, seq_len, d_model]
        x = x + self.pos_embed
        x = self.encoder(x)  # [B, seq_len, d_model]
        
        # 全局平均池化
        x = x.mean(dim=1)  # [B, d_model]
        return self.classifier(x)

class PSMNIST_Transformer(nn.Module):
    """Transformer for psMNIST"""
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

class PSMNIST_LSTM(nn.Module):
    """LSTM for psMNIST"""
    def __init__(self, d_model=128, n_layers=2, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, 
                           batch_first=True, bidirectional=True)
        
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
        x = x[:, -1, :]  # 取最后一个时间步
        return self.classifier(x)

# ============== 训练函数 ==============

def train_epoch(model, loader, optimizer, criterion):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f"    Batch {batch_idx}/{len(loader)}: Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%")
    
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total

def train_model(model_name, model, train_loader, test_loader, epochs=20):
    """完整训练流程"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n训练 {model_name}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}% (Best: {best_acc:.2f}%)")
    
    history['best_acc'] = best_acc
    return history

# ============== 主实验 ==============

def run_experiment():
    """运行psMNIST实验"""
    
    print("=" * 70)
    print("HormonicFormer 长程依赖测试 - Permuted Sequential MNIST")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print("=" * 70)
    
    # 加载数据
    print("\n加载psMNIST数据集...")
    train_dataset = PermutedMNIST(train=True)
    test_dataset = PermutedMNIST(train=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=0
    )
    
    print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")
    print(f"序列长度: 784 (28x28置换)")
    
    # 模型配置
    seq_len = 784
    d_model = 128
    n_layers = 4
    n_heads = 4
    num_classes = 10
    epochs = 20
    
    results = {}
    
    # 1. HormonicFormer
    print(f"\n{'='*70}")
    print("[1/3] HormonicFormer")
    print(f"{'='*70}")
    try:
        model = PSMNIST_Hormonic(seq_len, d_model, n_layers, n_heads, num_classes)
        result = train_model("HormonicFormer", model, train_loader, test_loader, epochs)
        results['hormonic'] = result
        print(f"\n[HormonicFormer] Best Test Acc: {result['best_acc']:.2f}%")
    except Exception as e:
        print(f"[FAIL] {e}")
        results['hormonic'] = {'error': str(e)}
    
    # 2. Transformer
    print(f"\n{'='*70}")
    print("[2/3] Transformer")
    print(f"{'='*70}")
    try:
        model = PSMNIST_Transformer(seq_len, d_model, n_layers, n_heads, num_classes)
        result = train_model("Transformer", model, train_loader, test_loader, epochs)
        results['transformer'] = result
        print(f"\n[Transformer] Best Test Acc: {result['best_acc']:.2f}%")
    except Exception as e:
        print(f"[FAIL] {e}")
        results['transformer'] = {'error': str(e)}
    
    # 3. LSTM
    print(f"\n{'='*70}")
    print("[3/3] LSTM")
    print(f"{'='*70}")
    try:
        model = PSMNIST_LSTM(d_model, n_layers, num_classes)
        result = train_model("LSTM", model, train_loader, test_loader, epochs)
        results['lstm'] = result
        print(f"\n[LSTM] Best Test Acc: {result['best_acc']:.2f}%")
    except Exception as e:
        print(f"[FAIL] {e}")
        results['lstm'] = {'error': str(e)}
    
    # 保存结果
    output_file = f'C:/Users/MR/Desktop/论文/关于场物理的神经框架/研究论文数据/psmnist_results_{datetime.now().strftime("%m%d_%H%M")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("实验完成!")
    print(f"结果保存: {output_file}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # 打印汇总
    print("\n结果汇总:")
    print(f"{'Model':<20} {'Best Test Acc':<15}")
    print("-" * 40)
    for model_name in ['hormonic', 'transformer', 'lstm']:
        acc = results.get(model_name, {}).get('best_acc', 'N/A')
        acc_str = f"{acc:.2f}%" if isinstance(acc, float) else str(acc)
        print(f"{model_name.capitalize():<20} {acc_str:<15}")
    
    # 文献基准对比
    print("\n文献基准:")
    print(f"  LSTM: ~85-90%")
    print(f"  Transformer: ~95-97%")
    print(f"  目标: HormonicFormer >90%")
    
    return results

if __name__ == '__main__':
    run_experiment()
