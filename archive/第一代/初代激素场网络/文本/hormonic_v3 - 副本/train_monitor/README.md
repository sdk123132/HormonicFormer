# HormonicFormer 训练监控面板

## 功能特性

- 📊 **实时监控**: 通过WebSocket实时推送训练数据
- 📈 **可视化图表**: 损失曲线、准确率曲线、学习率变化、神经调质变化
- 🧬 **Hebbian统计**: 实时显示突触可塑性状态
- 📱 **响应式界面**: 支持桌面和移动设备
- 💾 **数据持久化**: TensorBoard日志和CSV文件记录

## 项目结构

```
train_monitor/
├── __init__.py           # 模块导出
├── training_monitor.py   # 训练监控核心类
├── web_server.py         # Flask + WebSocket服务器
├── templates/
│   └── index.html        # 监控面板前端页面
└── README.md             # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
# 使用Anaconda环境
C:\Users\MR\anaconda3\python.exe -m pip install flask flask-socketio eventlet tensorboard pillow
```

### 2. 启动训练

方法一：使用PowerShell脚本（推荐）

```powershell
powershell -ExecutionPolicy Bypass -File start_training.ps1
```

方法二：使用Python启动脚本

```bash
C:\Users\MR\anaconda3\python.exe launch_train.py --config local_config.yaml --device cuda
```

方法三：直接运行监控训练脚本

```bash
# 设置环境变量
set KMP_DUPLICATE_LIB_OK=TRUE

# 启动训练
C:\Users\MR\anaconda3\python.exe monitored_train.py --config local_config.yaml --device cuda --monitor-port 5000
```

### 3. 访问监控面板

在浏览器中打开：`http://localhost:5000`

## 监控指标

### 实时统计
- **最佳验证准确率**: 训练过程中的最高验证准确率
- **训练/验证损失**: 当前epoch的损失值
- **训练/验证准确率**: 当前epoch的准确率
- **学习率**: 当前学习率
- **多巴胺(DA)**: 神经调质系统中的DA值
- **GPU显存**: 当前显存使用量

### 图表展示
- **损失曲线**: 训练损失和验证损失的变化趋势
- **准确率曲线**: 训练准确率和验证准确率的变化趋势
- **学习率变化**: 学习率调度曲线
- **DA/CB变化**: 神经调质变化曲线

### Hebbian统计
- **G均值**: G矩阵的平均值
- **G标准差**: G矩阵的标准差
- **稀疏度**: G矩阵的稀疏程度
- **存活比例**: 存活连接的比例

## 配置说明

### 监控服务器配置

```bash
# 修改监控端口（默认5000）
python monitored_train.py --monitor-port 8080
```

### 训练配置

配置文件 `local_config.yaml` 已针对RTX 5070 8GB显存优化：

- `batch_size: 64` - 适配8GB显存
- `d_model: 128` - 精简模型参数
- `n_layers: 2` - 减少层数加速训练
- `use_amp: false` - 关闭混合精度避免NaN问题

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      训练监控系统                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    WebSocket    ┌──────────────────────┐    │
│  │   训练脚本    │ ──────────────> │   Flask服务器        │    │
│  │ monitored.py │                 │      web_server.py   │    │
│  └──────────────┘                 └──────────┬───────────┘    │
│         │                                    │                │
│         │ 保存日志                           │ HTTP           │
│         ▼                                    ▼                │
│  ┌──────────────┐                   ┌──────────────────────┐  │
│  │ TensorBoard  │                   │   监控面板           │  │
│  │   日志文件    │                   │    index.html        │  │
│  │    CSV文件   │                   │   (Chart.js图表)     │  │
│  └──────────────┘                   └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `training_monitor.py` | 训练监控核心类，处理数据收集和日志记录 |
| `web_server.py` | Flask Web服务器，提供API和WebSocket推送 |
| `templates/index.html` | 监控面板前端页面 |
| `monitored_train.py` | 集成监控功能的训练脚本 |
| `launch_train.py` | 训练启动脚本（处理环境变量） |
| `start_training.ps1` | PowerShell启动脚本 |

## 快捷键

- **Ctrl+C**: 停止训练
- **F5**: 刷新监控页面

## 注意事项

1. 确保Anaconda环境已激活且PyTorch版本正确
2. 首次运行会自动下载FashionMNIST数据集
3. 监控服务器默认使用5000端口，请确保该端口未被占用
4. 训练过程中建议保持浏览器打开以获取实时更新

## 故障排除

### 常见问题

**Q: 监控页面无法访问**

A: 检查端口是否被占用，尝试更换端口：
```bash
python monitored_train.py --monitor-port 8080
```

**Q: 出现OpenMP错误**

A: 确保设置了环境变量：
```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

**Q: GPU显存不足**

A: 减小batch_size或使用梯度累积：
```yaml
train:
  batch_size: 32
  accumulation_steps: 4  # 等效batch_size 128
```

## 联系方式

如有问题请查看项目根目录的README.md或联系开发者。