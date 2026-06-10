"""
训练监控Web服务器 - Flask + WebSocket
提供实时监控界面和API接口
支持从日志文件加载历史训练数据
"""
import os
import json
import csv
import threading
import re
from datetime import datetime
from io import StringIO
from flask import Flask, render_template, jsonify, send_from_directory, Response
from flask_socketio import SocketIO, emit

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'hormonic_monitor_secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# 全局监控数据存储
monitor_data = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'metrics': {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'da': [],
        'cb': []
    },
    'hebbian_stats': [],
    'gpu_memory': 0,
    'start_time': None,
    'elapsed_time': 0
}

# 数据更新锁
data_lock = threading.Lock()

# 日志文件路径
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')


def load_history_data():
    """从日志文件加载历史训练数据"""
    global monitor_data
    
    if not os.path.exists(LOG_DIR):
        return
    
    # 找到最新的日志文件
    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
    if not log_files:
        return
    
    # 按修改时间排序，取最新的
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(LOG_DIR, x)), reverse=True)
    latest_log = os.path.join(LOG_DIR, log_files[0])
    
    try:
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 提取epoch数据：格式类似 "Epoch 1: train_loss=2.1123 train_acc=20.5% val_loss=1.9373 val_acc=26.4% DA=0.326 CB=0.000"
        epoch_pattern = re.compile(
            r'Epoch (\d+): train_loss=([\d.]+) train_acc=([\d.]+)% val_loss=([\d.]+) val_acc=([\d.]+)% DA=([\d.]+) CB=([\d.]+)'
        )
        
        matches = epoch_pattern.findall(content)
        
        if matches:
            with data_lock:
                # 清空现有数据
                for key in monitor_data['metrics']:
                    monitor_data['metrics'][key] = []
                
                for match in matches:
                    epoch = int(match[0])
                    train_loss = float(match[1])
                    train_acc = float(match[2])
                    val_loss = float(match[3])
                    val_acc = float(match[4])
                    da = float(match[5])
                    cb = float(match[6])
                    
                    monitor_data['metrics']['train_loss'].append(train_loss)
                    monitor_data['metrics']['train_acc'].append(train_acc)
                    monitor_data['metrics']['val_loss'].append(val_loss)
                    monitor_data['metrics']['val_acc'].append(val_acc)
                    monitor_data['metrics']['da'].append(da)
                    monitor_data['metrics']['cb'].append(cb)
                    # 学习率需要从batch日志中提取，这里先用默认值
                    monitor_data['metrics']['lr'].append(0.0005)
                
                monitor_data['current_epoch'] = len(matches)
                monitor_data['total_epochs'] = len(matches)
                
            print(f"Loaded {len(matches)} epochs from {latest_log}")
    
    except Exception as e:
        print(f"Error loading history data: {e}")


# 启动时加载历史数据
load_history_data()


@app.route('/')
def index():
    """主监控页面"""
    return render_template('index.html')


@app.route('/api/metrics')
def get_metrics():
    """获取所有监控数据"""
    with data_lock:
        return jsonify(monitor_data)


@app.route('/api/summary')
def get_summary():
    """获取训练摘要"""
    with data_lock:
        metrics = monitor_data['metrics']
        if not metrics['val_acc']:
            return jsonify({'error': 'No data yet'})
            
        best_idx = max(range(len(metrics['val_acc'])), key=lambda i: metrics['val_acc'][i])
        return jsonify({
            'total_epochs': len(metrics['train_loss']),
            'best_val_acc': metrics['val_acc'][best_idx],
            'best_epoch': best_idx + 1,
            'current_epoch': monitor_data['current_epoch'],
            'is_training': monitor_data['is_training'],
            'elapsed_time': monitor_data['elapsed_time']
        })


@app.route('/api/table_data')
def get_table_data():
    """获取表格数据（用于显示和下载）"""
    with data_lock:
        metrics = monitor_data['metrics']
        epochs = len(metrics['train_loss'])
        table_data = []
        
        for i in range(epochs):
            table_data.append({
                'epoch': i + 1,
                'train_loss': round(metrics['train_loss'][i], 4) if i < len(metrics['train_loss']) else '-',
                'train_acc': round(metrics['train_acc'][i], 2) if i < len(metrics['train_acc']) else '-',
                'val_loss': round(metrics['val_loss'][i], 4) if i < len(metrics['val_loss']) else '-',
                'val_acc': round(metrics['val_acc'][i], 2) if i < len(metrics['val_acc']) else '-',
                'lr': round(metrics['lr'][i], 6) if i < len(metrics['lr']) else '-',
                'da': round(metrics['da'][i], 3) if i < len(metrics['da']) else '-',
                'cb': round(metrics['cb'][i], 3) if i < len(metrics['cb']) else '-'
            })
        
        return jsonify({
            'data': table_data,
            'summary': {
                'total_epochs': epochs,
                'best_val_acc': max(metrics['val_acc']) if metrics['val_acc'] else 0,
                'best_epoch': metrics['val_acc'].index(max(metrics['val_acc'])) + 1 if metrics['val_acc'] else 0
            }
        })


@app.route('/api/download/csv')
def download_csv():
    """下载训练数据CSV文件"""
    with data_lock:
        metrics = monitor_data['metrics']
        epochs = len(metrics['train_loss'])
        
        # 创建CSV内容
        output = StringIO()
        writer = csv.writer(output)
        
        # 表头
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc (%)', 'Val Loss', 'Val Acc (%)', 'Learning Rate', 'DA', 'CB'])
        
        # 数据行
        for i in range(epochs):
            writer.writerow([
                i + 1,
                round(metrics['train_loss'][i], 4) if i < len(metrics['train_loss']) else '',
                round(metrics['train_acc'][i], 2) if i < len(metrics['train_acc']) else '',
                round(metrics['val_loss'][i], 4) if i < len(metrics['val_loss']) else '',
                round(metrics['val_acc'][i], 2) if i < len(metrics['val_acc']) else '',
                round(metrics['lr'][i], 6) if i < len(metrics['lr']) else '',
                round(metrics['da'][i], 3) if i < len(metrics['da']) else '',
                round(metrics['cb'][i], 3) if i < len(metrics['cb']) else ''
            ])
        
        output.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 返回CSV响应
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=training_data_{timestamp}.csv',
                'Content-Type': 'text/csv; charset=utf-8'
            }
        )


@app.route('/api/download/json')
def download_json():
    """下载训练数据JSON文件"""
    with data_lock:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 返回JSON响应
        return Response(
            json.dumps(monitor_data, indent=2, default=str),
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename=training_data_{timestamp}.json',
                'Content-Type': 'application/json; charset=utf-8'
            }
        )


@socketio.on('connect')
def handle_connect():
    """WebSocket连接处理"""
    print(f"Client connected: {datetime.now()}")
    with data_lock:
        emit('init_data', monitor_data)


@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket断开处理"""
    print(f"Client disconnected: {datetime.now()}")


def update_monitor_data(new_data):
    """更新监控数据（线程安全）"""
    global monitor_data
    with data_lock:
        # 更新状态
        if 'is_training' in new_data:
            monitor_data['is_training'] = new_data['is_training']
            if new_data['is_training'] and monitor_data['start_time'] is None:
                monitor_data['start_time'] = datetime.now()
        
        # 更新指标
        if 'metrics' in new_data:
            for key, value in new_data['metrics'].items():
                if key in monitor_data['metrics']:
                    monitor_data['metrics'][key].append(value)
        
        # 更新当前epoch
        if 'current_epoch' in new_data:
            monitor_data['current_epoch'] = new_data['current_epoch']
        
        if 'total_epochs' in new_data:
            monitor_data['total_epochs'] = new_data['total_epochs']
        
        # 更新Hebbian统计
        if 'hebbian_stats' in new_data:
            monitor_data['hebbian_stats'] = new_data['hebbian_stats']
        
        # 更新GPU显存
        if 'gpu_memory' in new_data:
            monitor_data['gpu_memory'] = new_data['gpu_memory']
        
        # 更新时间
        if monitor_data['start_time']:
            monitor_data['elapsed_time'] = (datetime.now() - monitor_data['start_time']).total_seconds()
    
    # 通过WebSocket推送更新
    socketio.emit('data_update', monitor_data)


def run_server(host='0.0.0.0', port=5000, debug=False):
    """启动Web服务器"""
    print(f"Starting training monitor server on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, use_reloader=False)


if __name__ == '__main__':
    run_server(debug=True)
