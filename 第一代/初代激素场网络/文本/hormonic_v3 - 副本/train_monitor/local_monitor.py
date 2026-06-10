"""
本地训练监控软件 - 基于Tkinter的桌面应用
支持实时监控、数据表格展示、图表显示和数据下载
"""
import os
import sys
import json
import csv
import re
import threading
import time
from datetime import datetime
from io import StringIO

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError as e:
    print(f"需要安装依赖: {e}")
    sys.exit(1)

# 全局监控数据
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

class TrainingMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HormonicFormer 训练监控")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # 数据更新锁
        self.data_lock = threading.Lock()
        
        # 初始化UI
        self.setup_ui()
        
        # 加载历史数据
        self.load_history_data()
        
        # 定时更新UI
        self.update_interval = 2000  # 2秒更新一次
        self.schedule_update()
    
    def setup_ui(self):
        # 创建主框架
        self.main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧面板 - 统计信息
        self.left_panel = ttk.Frame(self.main_frame, width=300)
        self.main_frame.add(self.left_panel, weight=1)
        
        # 右侧面板 - 图表和表格
        self.right_panel = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.main_frame.add(self.right_panel, weight=3)
        
        # 统计卡片
        self.setup_stats_cards()
        
        # 图表区域
        self.setup_charts()
        
        # 数据表格区域
        self.setup_data_table()
        
        # 下载按钮
        self.setup_download_buttons()
    
    def setup_stats_cards(self):
        """设置统计信息卡片"""
        stats_frame = ttk.LabelFrame(self.left_panel, text="训练统计", padding=10)
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_vars = {
            'best_val_acc': tk.StringVar(value="0.00%"),
            'train_loss': tk.StringVar(value="0.0000"),
            'train_acc': tk.StringVar(value="0.00%"),
            'val_loss': tk.StringVar(value="0.0000"),
            'val_acc': tk.StringVar(value="0.00%"),
            'lr': tk.StringVar(value="0.000000"),
            'da': tk.StringVar(value="0.500"),
            'gpu_memory': tk.StringVar(value="0 MB")
        }
        
        stats_labels = [
            ("最佳验证准确率", 'best_val_acc'),
            ("当前训练损失", 'train_loss'),
            ("当前训练准确率", 'train_acc'),
            ("当前验证损失", 'val_loss'),
            ("当前验证准确率", 'val_acc'),
            ("学习率", 'lr'),
            ("多巴胺 (DA)", 'da'),
            ("GPU显存", 'gpu_memory')
        ]
        
        for label_text, var_name in stats_labels:
            frame = ttk.Frame(stats_frame)
            frame.pack(fill=tk.X, pady=3)
            
            ttk.Label(frame, text=label_text, width=15).pack(side=tk.LEFT)
            ttk.Label(frame, textvariable=self.stats_vars[var_name], 
                      font=('Arial', 10, 'bold'), foreground='#0066cc').pack(side=tk.RIGHT)
        
        # 训练状态
        self.status_var = tk.StringVar(value="空闲")
        status_frame = ttk.Frame(stats_frame)
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, text="训练状态:", width=15).pack(side=tk.LEFT)
        status_label = ttk.Label(status_frame, textvariable=self.status_var, font=('Arial', 10, 'bold'))
        status_label.pack(side=tk.RIGHT)
        self.status_label = status_label
        
        # 进度条
        progress_frame = ttk.LabelFrame(stats_frame, text="训练进度", padding=5)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.StringVar(value="Epoch 0 / 0")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # 时间显示
        time_frame = ttk.LabelFrame(stats_frame, text="时间信息", padding=5)
        time_frame.pack(fill=tk.X, pady=5)
        
        self.elapsed_time_var = tk.StringVar(value="已运行: 00:00:00")
        ttk.Label(time_frame, textvariable=self.elapsed_time_var).pack(fill=tk.X)
        
        self.remaining_time_var = tk.StringVar(value="预计剩余: --:--:--")
        ttk.Label(time_frame, textvariable=self.remaining_time_var).pack(fill=tk.X)
    
    def setup_charts(self):
        """设置图表区域"""
        chart_frame = ttk.LabelFrame(self.right_panel, text="训练图表", padding=10)
        self.right_panel.add(chart_frame, weight=2)
        
        # 创建两个子图
        self.fig = Figure(figsize=(10, 4), dpi=100)
        
        # 损失曲线
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_title('损失曲线')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True)
        
        # 准确率曲线
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_title('准确率曲线')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.grid(True)
        
        # 神经调制曲线
        self.fig2 = Figure(figsize=(10, 3), dpi=100)
        self.ax3 = self.fig2.add_subplot(111)
        self.ax3.set_title('DA / CB 变化')
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('Value')
        self.ax3.grid(True)
        
        # 画布
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.canvas2 = FigureCanvasTkAgg(self.fig2, chart_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)
    
    def setup_data_table(self):
        """设置数据表格"""
        table_frame = ttk.LabelFrame(self.right_panel, text="训练数据表格", padding=10)
        self.right_panel.add(table_frame, weight=2)
        
        # 创建树状视图作为表格
        columns = ('epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'da', 'cb')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        # 设置列标题
        self.tree.heading('epoch', text='Epoch')
        self.tree.heading('train_loss', text='Train Loss')
        self.tree.heading('train_acc', text='Train Acc (%)')
        self.tree.heading('val_loss', text='Val Loss')
        self.tree.heading('val_acc', text='Val Acc (%)')
        self.tree.heading('lr', text='LR')
        self.tree.heading('da', text='DA')
        self.tree.heading('cb', text='CB')
        
        # 设置列宽度
        self.tree.column('epoch', width=60, anchor=tk.CENTER)
        self.tree.column('train_loss', width=100, anchor=tk.E)
        self.tree.column('train_acc', width=100, anchor=tk.E)
        self.tree.column('val_loss', width=100, anchor=tk.E)
        self.tree.column('val_acc', width=100, anchor=tk.E)
        self.tree.column('lr', width=90, anchor=tk.E)
        self.tree.column('da', width=60, anchor=tk.E)
        self.tree.column('cb', width=60, anchor=tk.E)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # 空数据提示
        if not monitor_data['metrics']['train_loss']:
            self.tree.insert('', tk.END, values=('等待训练数据...', '', '', '', '', '', '', ''))
    
    def setup_download_buttons(self):
        """设置下载按钮"""
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="📥 下载 CSV", command=self.download_csv).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="📥 下载 JSON", command=self.download_json).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="🔄 刷新数据", command=self.load_history_data).pack(side=tk.RIGHT, padx=5)
    
    def load_history_data(self):
        """从日志文件加载历史训练数据"""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        
        if not os.path.exists(log_dir):
            return
        
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if not log_files:
            return
        
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
        latest_log = os.path.join(log_dir, log_files[0])
        
        try:
            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 提取epoch数据
            epoch_pattern = re.compile(
                r'Epoch (\d+): train_loss=([\d.]+) train_acc=([\d.]+)% val_loss=([\d.]+) val_acc=([\d.]+)% DA=([\d.]+) CB=([\d.]+)'
            )
            
            matches = epoch_pattern.findall(content)
            
            if matches:
                with self.data_lock:
                    for key in monitor_data['metrics']:
                        monitor_data['metrics'][key] = []
                    
                    for match in matches:
                        monitor_data['metrics']['train_loss'].append(float(match[1]))
                        monitor_data['metrics']['train_acc'].append(float(match[2]))
                        monitor_data['metrics']['val_loss'].append(float(match[3]))
                        monitor_data['metrics']['val_acc'].append(float(match[4]))
                        monitor_data['metrics']['da'].append(float(match[5]))
                        monitor_data['metrics']['cb'].append(float(match[6]))
                        monitor_data['metrics']['lr'].append(0.0005)
                    
                    monitor_data['current_epoch'] = len(matches)
                    monitor_data['total_epochs'] = len(matches)
                
                self.update_ui()
                print(f"Loaded {len(matches)} epochs from {latest_log}")
        
        except Exception as e:
            print(f"Error loading history data: {e}")
    
    def update_ui(self):
        """更新UI显示"""
        with self.data_lock:
            metrics = monitor_data['metrics']
            
            # 更新统计信息
            if metrics['val_acc']:
                best_acc = max(metrics['val_acc'])
                self.stats_vars['best_val_acc'].set(f"{best_acc:.2f}%")
            else:
                self.stats_vars['best_val_acc'].set("0.00%")
            
            if metrics['train_loss']:
                self.stats_vars['train_loss'].set(f"{metrics['train_loss'][-1]:.4f}")
            if metrics['train_acc']:
                self.stats_vars['train_acc'].set(f"{metrics['train_acc'][-1]:.2f}%")
            if metrics['val_loss']:
                self.stats_vars['val_loss'].set(f"{metrics['val_loss'][-1]:.4f}")
            if metrics['val_acc']:
                self.stats_vars['val_acc'].set(f"{metrics['val_acc'][-1]:.2f}%")
            if metrics['lr']:
                self.stats_vars['lr'].set(f"{metrics['lr'][-1]:.6f}")
            if metrics['da']:
                self.stats_vars['da'].set(f"{metrics['da'][-1]:.3f}")
            
            self.stats_vars['gpu_memory'].set(f"{monitor_data['gpu_memory']} MB")
            
            # 更新状态
            status = "训练中" if monitor_data['is_training'] else "空闲"
            self.status_var.set(status)
            self.status_label.config(foreground='#00cc00' if monitor_data['is_training'] else '#666666')
            
            # 更新进度
            current = monitor_data['current_epoch']
            total = monitor_data['total_epochs']
            self.progress_var.set(f"Epoch {current} / {total}")
            
            if total > 0:
                self.progress_bar['value'] = (current / total) * 100
            else:
                self.progress_bar['value'] = 0
            
            # 更新时间
            elapsed = monitor_data['elapsed_time']
            self.elapsed_time_var.set(f"已运行: {self.format_time(elapsed)}")
            
            if current > 0 and total > 0:
                avg_time = elapsed / current
                remaining = (total - current) * avg_time
                self.remaining_time_var.set(f"预计剩余: {self.format_time(remaining)}")
            else:
                self.remaining_time_var.set("预计剩余: --:--:--")
            
            # 更新图表
            self.update_charts()
            
            # 更新表格
            self.update_table()
    
    def update_charts(self):
        """更新图表"""
        metrics = monitor_data['metrics']
        epochs = list(range(1, len(metrics['train_loss']) + 1))
        
        # 清空旧图表
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        if epochs:
            # 损失曲线
            self.ax1.plot(epochs, metrics['train_loss'], label='Train Loss', color='#60a5fa')
            self.ax1.plot(epochs, metrics['val_loss'], label='Val Loss', color='#f87171')
            self.ax1.set_title('损失曲线')
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Loss')
            self.ax1.legend()
            self.ax1.grid(True)
            
            # 准确率曲线
            self.ax2.plot(epochs, metrics['train_acc'], label='Train Acc', color='#34d399')
            self.ax2.plot(epochs, metrics['val_acc'], label='Val Acc', color='#a78bfa')
            self.ax2.set_title('准确率曲线')
            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('Accuracy (%)')
            self.ax2.legend()
            self.ax2.grid(True)
            self.ax2.set_ylim(0, 100)
            
            # 神经调制曲线
            self.ax3.plot(epochs, metrics['da'], label='DA', color='#f472b6')
            self.ax3.plot(epochs, metrics['cb'], label='CB', color='#a3e635')
            self.ax3.set_title('DA / CB 变化')
            self.ax3.set_xlabel('Epoch')
            self.ax3.set_ylabel('Value')
            self.ax3.legend()
            self.ax3.grid(True)
            self.ax3.set_ylim(0, 1)
        
        self.canvas.draw()
        self.canvas2.draw()
    
    def update_table(self):
        """更新数据表格"""
        # 清空现有数据
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        metrics = monitor_data['metrics']
        
        if not metrics['train_loss']:
            self.tree.insert('', tk.END, values=('等待训练数据...', '', '', '', '', '', '', ''))
            return
        
        # 找到最佳验证准确率
        best_epoch = 0
        best_val_acc = 0
        if metrics['val_acc']:
            best_val_acc = max(metrics['val_acc'])
            best_epoch = metrics['val_acc'].index(best_val_acc) + 1
        
        # 添加数据行
        for i in range(len(metrics['train_loss'])):
            values = (
                i + 1,
                f"{metrics['train_loss'][i]:.4f}",
                f"{metrics['train_acc'][i]:.2f}",
                f"{metrics['val_loss'][i]:.4f}",
                f"{metrics['val_acc'][i]:.2f}",
                f"{metrics['lr'][i]:.6f}",
                f"{metrics['da'][i]:.3f}",
                f"{metrics['cb'][i]:.3f}"
            )
            self.tree.insert('', tk.END, values=values)
    
    def download_csv(self):
        """下载CSV文件"""
        with self.data_lock:
            metrics = monitor_data['metrics']
            epochs = len(metrics['train_loss'])
            
            if epochs == 0:
                messagebox.showwarning("警告", "没有数据可下载")
                return
            
            output = StringIO()
            writer = csv.writer(output)
            
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc (%)', 'Val Loss', 'Val Acc (%)', 'Learning Rate', 'DA', 'CB'])
            
            for i in range(epochs):
                writer.writerow([
                    i + 1,
                    round(metrics['train_loss'][i], 4),
                    round(metrics['train_acc'][i], 2),
                    round(metrics['val_loss'][i], 4),
                    round(metrics['val_acc'][i], 2),
                    round(metrics['lr'][i], 6),
                    round(metrics['da'][i], 3),
                    round(metrics['cb'][i], 3)
                ])
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'training_data_{timestamp}.csv'
            
            # 选择保存位置
            filepath = filedialog.asksaveasfilename(
                defaultextension='.csv',
                filetypes=[('CSV文件', '*.csv')],
                initialfile=filename
            )
            
            if filepath:
                with open(filepath, 'w', encoding='utf-8', newline='') as f:
                    f.write(output.getvalue())
                messagebox.showinfo("成功", f"数据已保存到:\n{filepath}")
    
    def download_json(self):
        """下载JSON文件"""
        with self.data_lock:
            if not monitor_data['metrics']['train_loss']:
                messagebox.showwarning("警告", "没有数据可下载")
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'training_data_{timestamp}.json'
            
            filepath = filedialog.asksaveasfilename(
                defaultextension='.json',
                filetypes=[('JSON文件', '*.json')],
                initialfile=filename
            )
            
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(monitor_data, f, indent=2, default=str)
                messagebox.showinfo("成功", f"数据已保存到:\n{filepath}")
    
    def schedule_update(self):
        """定时更新UI"""
        self.update_ui()
        self.root.after(self.update_interval, self.schedule_update)
    
    @staticmethod
    def format_time(seconds):
        """格式化时间显示"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

if __name__ == '__main__':
    # 设置OpenMP环境变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    root = tk.Tk()
    app = TrainingMonitorApp(root)
    root.mainloop()
