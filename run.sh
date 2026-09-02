#!/usr/bin/env bash
# 设备工况智能预测系统 —— 一键启动脚本
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo " 设备工况智能预测系统 启动"
echo "=========================================="

# 1. 安装依赖（可选：如已安装可注释）
# pip3 install -r requirements.txt

# 2. 若数据库为空，初始化演示数据
if [ ! -f data/app.db ]; then
  echo "[init] 初始化演示数据（预处理 + 训练 + 优化）..."
  python3 -m backend.seed
fi

# 3. 启动后端服务
echo "[start] 后端服务启动中: http://127.0.0.1:8000"
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
