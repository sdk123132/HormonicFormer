# HormonicFormer V3 训练监控启动脚本 (PowerShell)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "==============================="
Write-Host "HormonicFormer V3 训练监控面板"
Write-Host "==============================="
Write-Host ""
Write-Host "正在启动训练监控服务器..."
Write-Host "监控面板地址: http://localhost:5000"
Write-Host ""

# 设置Python路径
$pythonPath = "C:\Users\MR\anaconda3\python.exe"

# 启动训练（带监控）
& $pythonPath monitored_train.py --config local_config.yaml --device cuda --monitor-port 5000