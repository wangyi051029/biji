#!/usr/bin/env bash
# ============================================================
# 设备工况智能预测系统 —— 一键启动脚本
# 用法：
#   ./scripts/run.sh          # 首次运行（含数据初始化）
#   ./scripts/run.sh --demo   # 仅启动服务（数据已初始化）
# ============================================================
set -e
cd "$(dirname "$0")/.."

MODE="${1:-init}"

if [ "$MODE" = "init" ]; then
  echo ">> [1/2] 初始化演示数据（预处理 + 训练 + 优化）..."
  python3 -m backend.seed --samples 1200 --epochs 35 --hidden 96
fi

echo ">> [2/2] 启动后端服务 http://localhost:8000"
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
