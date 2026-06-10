@echo off
echo ===============================
echo HormonicFormer V3 训练监控面板
echo ===============================
echo.
echo 正在启动训练监控服务器...
echo 监控面板地址: http://localhost:5000
echo.

:: 设置Python路径（根据你的Anaconda环境）
set PYTHON_PATH=C:\Users\MR\anaconda3\python.exe

:: 启动训练（带监控）
%PYTHON_PATH% monitored_train.py --config local_config.yaml --device cuda --monitor-port 5000

pause