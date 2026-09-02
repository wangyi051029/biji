# -*- coding: utf-8 -*-
"""
演示数据初始化脚本
==================
一键完成：初始化数据库 → 预处理全部子集 → 训练 FD001 基准模型 → 智能优化
        → 演示预测 → 记录实验结果。

运行方式：
  python3 -m backend.seed            # 使用默认参数
  python3 -m backend.seed --samples 1000 --epochs 30
"""

import argparse
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import database as db
from backend import services


def seed(samples=1500, epochs=40, hidden=96, subset="FD001",
         prune_ratio=0.3, quant_bits=8, predict_unit=2):
    t0 = time.time()
    print("=" * 60)
    print("设备工况智能预测系统 —— 演示数据初始化")
    print("=" * 60)

    # 1. 初始化数据库
    db.init_db()
    for t in ["models", "predictions", "experiments", "datasets"]:
        conn = db.get_conn()
        conn.execute(f"DELETE FROM {t}")
        conn.commit()
        conn.close()
    print("[1/5] 数据库已初始化")

    # 2. 预处理全部子集（统计学习）
    services.preprocess_all()
    datasets = db.list_datasets()
    print(f"[2/5] 数据预处理完成，共 {len(datasets)} 个子集入库")
    for d in datasets:
        print(f"      {d['name']}: 训练 {d['n_train']} 样本, "
              f"特征 {d['n_features']}, 噪声率 {d['noise_rate']:.1%}")

    # 3. 训练基准模型（深度学习 LSTM）
    t1 = time.time()
    model = services.train_model(
        subset, n_samples=samples, epochs=epochs, hidden=hidden,
        name=f"{subset}_LSTM")
    print(f"[3/5] 基准模型训练完成：{model['name']}  "
          f"MAE={model['mae']} RMSE={model['rmse']} "
          f"参数={model['params_k']}K 推理={model['inference_ms']}ms "
          f"(耗时 {time.time()-t1:.1f}s)")

    # 4. 智能计算优化（剪枝 + 量化）
    opt = services.optimize_model(subset, model["name"],
                                  prune_ratio, quant_bits)
    c = opt["compare"]
    print(f"[4/5] 智能优化完成：参数减少 {c['param_reduction']:.1%}，"
          f"模型大小减少 {c['size_reduction']:.1%}")

    # 5. 演示预测（测试集某单元）
    pred = services.predict_rul(subset, model["name"], predict_unit)
    print(f"[5/5] 演示预测：单元 {predict_unit} RUL ≈ {pred['pred_rul']} 周期"
          f"（真值 {pred['true_rul']}）")

    # 6. 实验记录
    metrics = {
        "baseline_mae": model["mae"], "baseline_rmse": model["rmse"],
        "baseline_params_k": model["params_k"],
        "baseline_infer_ms": model["inference_ms"],
        "optimized_param_reduction": c["param_reduction"],
        "optimized_size_reduction": c["size_reduction"],
        "optimized_mae": c["optimized"]["metric"]["mae"],
        "total_seconds": round(time.time() - t0, 1),
    }
    db.add_experiment("演示实验-基准vs优化", {
        "subset": subset, "samples": samples, "epochs": epochs,
        "prune_ratio": prune_ratio, "quant_bits": quant_bits,
    }, metrics)
    print(f"\n实验记录已保存，总耗时 {metrics['total_seconds']}s")
    print("初始化完成。启动系统：python3 -m uvicorn backend.main:app --port 8000")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--subset", type=str, default="FD001")
    p.add_argument("--prune", type=float, default=0.3)
    p.add_argument("--bits", type=int, default=8)
    args = p.parse_args()
    seed(args.samples, args.epochs, args.hidden, args.subset,
         args.prune, args.bits)
