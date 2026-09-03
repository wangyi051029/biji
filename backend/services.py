# -*- coding: utf-8 -*-
"""
业务逻辑服务层
==============
串联 数据(统计学习) → 模型(深度学习) → 优化(智能计算优化) 的完整业务闭环，
并持久化到 SQLite 数据库。

主要能力：
  - 数据服务：预处理指定子集、查询数据集元信息
  - 模型服务：训练 LSTM 基准模型、在线预测 RUL
  - 优化服务：模型剪枝、参数量化、压缩前后对比
"""

import os
import json
import numpy as np
import pandas as pd

from . import database as db
from .algorithms import statistical as stat
from .algorithms import deep_learning as dl
from .algorithms import optimization as opt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

DEFAULT_FEATURES = ["cycle"] + [f"s{i}" for i in range(2, 22)]


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def _raw_path(subset):
    return os.path.join(RAW_DIR, f"train_{subset}.txt")


def get_subset_list():
    return ["FD001", "FD002", "FD003", "FD004"]


# ---------------- 数据服务 ----------------
def preprocess_subset(subset, save_csv=True, persist=True):
    """
    对指定子集执行完整预处理流水线（统计学习方法），
    返回预处理后的划分与元信息，并可持久化。
    """
    raw = stat.load_raw_train(_raw_path(subset))
    splits, meta, scaler = stat.preprocess_pipeline(raw)

    if save_csv:
        out = os.path.join(PROCESSED_DIR, subset)
        os.makedirs(out, exist_ok=True)
        keep_cols = ["unit", "cycle"] + meta["feature_selection"]["selected_features"] \
                    + ["RUL", "is_noise", "max_cycle"]
        for name in ["train", "val", "test"]:
            splits[name][keep_cols].to_csv(os.path.join(out, f"{name}_clean.csv"),
                                           index=False)
        with open(os.path.join(out, "feature_selection.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta["feature_selection"], f, ensure_ascii=False, indent=2)
        # 保存标准化参数（供手动输入数据预测时使用同一套标准化）
        with open(os.path.join(out, "scaler.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "features": list(meta["features"]),
            }, f, ensure_ascii=False, indent=2)

    if persist:
        fs = meta["feature_selection"]
        db.upsert_dataset(
            name=subset,
            description=f"C-MAPSS {subset} 预处理后数据",
            n_train=meta["n_train"], n_test=meta["n_test"],
            n_features=len(fs["selected_features"]),
            noise_rate=meta["noise_rate"],
        )
    return {"subset": subset, "meta": meta}


def preprocess_all():
    results = []
    for s in get_subset_list():
        results.append(preprocess_subset(s))
    return results


def load_processed_train(subset):
    """加载预处理后的训练集 CSV（若不存在则先预处理）。"""
    path = os.path.join(PROCESSED_DIR, subset, "train_clean.csv")
    if not os.path.exists(path):
        preprocess_subset(subset)
    return pd.read_csv(path)


def load_processed_test(subset):
    path = os.path.join(PROCESSED_DIR, subset, "test_clean.csv")
    if not os.path.exists(path):
        preprocess_subset(subset)
    return pd.read_csv(path)


# ---------------- 模型服务 ----------------
def _feature_cols(subset):
    fs_path = os.path.join(PROCESSED_DIR, subset, "feature_selection.json")
    if os.path.exists(fs_path):
        with open(fs_path, encoding="utf-8") as f:
            fs = json.load(f)
        return ["cycle"] + fs["selected_features"]
    return DEFAULT_FEATURES


def _prepare_windows(subset, n_samples=60):
    """从预处理后的训练集构造滑动窗口样本（限制样本量以便普通PC快速训练）。"""
    train = load_processed_train(subset)
    feats = _feature_cols(subset)
    X, y, units, last_cycles = dl.make_windows(train, feats)
    # 抽样，控制训练规模
    if len(X) > n_samples:
        idx = np.random.RandomState(0).choice(len(X), n_samples, replace=False)
        X, y, units, last_cycles = X[idx], y[idx], units[idx], last_cycles[idx]
    return X, y, units, last_cycles, feats


def train_model(subset, n_samples=60, window=30, epochs=10, hidden=64,
                lr=1e-3, name=None):
    """
    训练 LSTM 基准模型并持久化。
    返回模型信息 dict。
    """
    ensure_dirs()
    X, y, units, last_cycles, feats = _prepare_windows(subset, n_samples)
    if len(X) == 0:
        raise ValueError(f"{subset} 无足够数据构造窗口样本")
    n_feat = X.shape[2]

    predictor = dl.RULPredictor(
        input_size=n_feat, hidden_size=hidden, window=window,
        epochs=epochs, lr=lr)
    # 划分 train/val
    split = int(len(X) * 0.8)
    predictor.fit(X[:split], y[:split], X[split:], y[split:], verbose=True)

    # 在测试窗口上评估
    test_X, test_y = X[split:], y[split:]
    metrics = predictor.evaluate(test_X, test_y)
    infer_ms = predictor.inference_time(test_X)
    params_k = predictor.params_size()

    model_name = name or f"{subset}_LSTM"
    predictor.save(os.path.join(MODEL_DIR, f"{model_name}.pt"),
                   meta={"subset": subset, "features": feats, "window": window})

    model_id = db.add_model(
        name=model_name, algo_type="baseline", dataset=subset,
        params_k=round(params_k, 2),
        model_bytes=int(params_k * 1000 * 4),
        inference_ms=round(infer_ms, 4),
        mae=round(metrics["mae"], 4), rmse=round(metrics["rmse"], 4),
    )
    return {
        "id": model_id, "name": model_name, "algo_type": "baseline",
        "subset": subset, "params_k": round(params_k, 2),
        "inference_ms": round(infer_ms, 4),
        "mae": round(metrics["mae"], 4), "rmse": round(metrics["rmse"], 4),
        "features": feats, "window": window,
    }


def predict_rul(subset, model_name, unit, n_cycles=30):
    """
    在线预测：给定某台发动机最近 n_cycles 个周期，预测 RUL。
    - 从预处理后的测试集加载该单元数据（标准化特征），取最后 n_cycles 周期。
    - 加载模型进行预测，并将记录持久化到数据库。
    返回预测结果。
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    meta_path = model_path + ".meta.json"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型 {model_name} 不存在，请先训练")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    feats = meta["features"]
    window = meta["window"]

    # 从预处理后的测试集加载该单元数据
    test = load_processed_test(subset)
    unit_df = test[test["unit"] == int(unit)].sort_values("cycle")
    if len(unit_df) == 0:
        train = load_processed_train(subset)
        unit_df = train[train["unit"] == int(unit)].sort_values("cycle")
    if len(unit_df) == 0:
        raise ValueError(f"子集 {subset} 中不存在单元 {unit}")
    last_cycle = int(unit_df["cycle"].iloc[-1])
    true_rul = float(unit_df["RUL"].iloc[-1]) if "RUL" in unit_df.columns else None

    data = unit_df[feats].values.astype(np.float32)
    if len(data) < window:
        pad = window - len(data)
        data = np.concatenate([np.repeat(data[:1], pad, axis=0), data], axis=0)
    X = data[-window:][None, :, :].astype(np.float32)

    predictor = dl.RULPredictor(input_size=len(feats), window=window)
    predictor.load(model_path)
    pred = float(predictor.predict(X)[0])

    # 持久化预测记录
    db.add_predictions([{
        "dataset": subset, "model_name": model_name, "unit": int(unit),
        "last_cycle": last_cycle, "true_rul": true_rul,
        "pred_rul": round(pred, 2),
    }])
    return {"model_name": model_name, "subset": subset, "unit": int(unit),
            "last_cycle": last_cycle, "true_rul": true_rul,
            "pred_rul": round(pred, 2), "window_used": window}
def get_model_info(model_name):
    """
    返回模型元信息（特征列表、窗口长度、子集），
    并附带一行从原始训练数据抽取的示例传感器读数，
    供前端"手动输入数据测试"界面动态构建表单与填充示例。
    """
    meta_path = os.path.join(MODEL_DIR, f"{model_name}.pt.meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"模型 {model_name} 不存在，请先训练")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    subset = meta.get("subset", "FD001")
    sensor_feats = [c for c in meta.get("features", []) if c != "cycle"]
    sample = None
    raw_path = _raw_path(subset)
    if os.path.exists(raw_path):
        raw = stat.load_raw_train(raw_path)
        row = raw.iloc[len(raw) // 2]
        sample = [round(float(row[c]), 4) for c in sensor_feats]
    return {
        "name": model_name, "subset": subset,
        "features": meta.get("features", []),
        "sensor_features": sensor_feats,
        "window": meta.get("window", 30),
        "sample_values": sample,
    }
def predict_rul_manual(subset, model_name, rows):
    """
    手动输入数据预测（测试界面）：
    用户输入若干周期的原始传感器读数（每行一个周期），
    使用该子集的 StandardScaler 标准化后构造滑动窗口，预测 RUL。
    rows: list[list[float]]，每行依次为模型所用传感器特征（不含 cycle）的原始读数。
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    meta_path = model_path + ".meta.json"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型 {model_name} 不存在，请先训练")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    feats = meta.get("features", [])
    window = meta.get("window", 30)
    sensor_feats = [c for c in feats if c != "cycle"]
    if not rows or not rows[0]:
        raise ValueError("请输入至少一行传感器读数")
    if len(rows[0]) != len(sensor_feats):
        raise ValueError(
            f"每行需要 {len(sensor_feats)} 个特征值（顺序：{', '.join(sensor_feats)}），"
            f"实际输入 {len(rows[0])} 个")
    # 加载标准化参数
    scaler_path = os.path.join(PROCESSED_DIR, subset, "scaler.json")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"缺少 {subset} 标准化参数 scaler.json，请先执行预处理")
    with open(scaler_path, encoding="utf-8") as f:
        sc = json.load(f)
    mean = np.asarray(sc["mean"], dtype=np.float64)
    scale = np.asarray(sc["scale"], dtype=np.float64)
    sc_feats = list(sc.get("features", []))
    # 若 scaler 特征顺序与模型不一致，按模型特征顺序重排
    if sc_feats and sc_feats != feats:
        idx = [sc_feats.index(f) for f in feats]
        mean, scale = mean[idx], scale[idx]
    scale[scale == 0] = 1.0  # 防除零
    # 构造 DataFrame（cycle 自动递增）
    df = pd.DataFrame(rows, columns=sensor_feats).astype(float)
    df.insert(0, "cycle", range(1, len(df) + 1))
    # 标准化 → 构造窗口
    X = (df[feats].values - mean) / scale
    if len(X) < window:
        pad = window - len(X)
        X = np.concatenate([np.repeat(X[:1], pad, axis=0), X], axis=0)
    win = X[-window:][None, :, :].astype(np.float32)
    predictor = dl.RULPredictor(input_size=len(feats), window=window)
    predictor.load(model_path)
    pred = float(predictor.predict(win)[0])
    # 持久化预测记录（unit=0 表示手动输入）
    db.add_predictions([{
        "dataset": subset, "model_name": model_name, "unit": 0,
        "last_cycle": len(df), "true_rul": None,
        "pred_rul": round(pred, 2),
    }])
    return {
        "model_name": model_name, "subset": subset,
        "input_rows": len(df), "features": sensor_feats,
        "window_used": window, "pred_rul": round(pred, 2),
        "note": "手动输入数据测试（非数据集单元，RUL 仅供参考）",
    }


# ---------------- 优化服务 ----------------
def _load_predictor(subset, model_name):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型 {model_name} 不存在")
    meta_path = model_path + ".meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    feats = meta["features"]
    predictor = dl.RULPredictor(input_size=len(feats), window=meta["window"])
    predictor.load(model_path)
    return predictor, meta


def optimize_model(subset, model_name, prune_ratio=0.3, quant_bits=8):
    """
    对基准模型实施剪枝 + 量化，并对比压缩前后指标，持久化结果。
    """
    ensure_dirs()
    predictor, meta = _load_predictor(subset, model_name)
    X, y, units, last_cycles, feats = _prepare_windows(subset, 60)

    # 剪枝
    pruned_model, prune_stat = opt.prune_model_weight_magnitude(
        predictor.model, prune_ratio)
    pruned_predictor = dl.RULPredictor(input_size=X.shape[2],
                                       window=meta["window"])
    pruned_predictor.model = pruned_model

    # 量化
    quant_model, quant_stat = opt.quantize_model_sim(pruned_model, quant_bits)
    quant_predictor = dl.RULPredictor(input_size=X.shape[2], window=meta["window"])
    quant_predictor.model = quant_model

    # 对比（用测试样本）
    split = int(len(X) * 0.8)
    test_X, test_y = X[split:], y[split:]
    comp = opt.compare_models(predictor.model, quant_predictor.model,
                              test_X, test_y, quant_bits=quant_bits)

    opt_name = f"{model_name}_pruned{int(prune_ratio*100)}_q{quant_bits}"
    quant_predictor.save(os.path.join(MODEL_DIR, f"{opt_name}.pt"),
                         meta={**meta, "prune_ratio": prune_ratio,
                               "quant_bits": quant_bits, "optimized": True})
    db.add_model(
        name=opt_name, algo_type="optimized", dataset=subset,
        params_k=round(comp["optimized"]["params_k"], 2),
        model_bytes=int(comp["optimized"]["model_bytes"]),
        inference_ms=comp["optimized"]["inference_ms"],
        mae=comp["optimized"]["metric"]["mae"] if comp["optimized"]["metric"] else None,
        rmse=comp["optimized"]["metric"]["rmse"] if comp["optimized"]["metric"] else None,
        sparsity=prune_stat["sparsity"], quant_bits=quant_bits,
    )
    return {
        "model_name": opt_name, "subset": subset,
        "prune": prune_stat, "quant": quant_stat,
        "compare": comp,
    }


def get_compare_models(subset):
    """获取某子集下基准与优化后模型，构造对比数据。"""
    models = db.get_models_by_dataset(subset)
    baseline = [m for m in models if m["algo_type"] == "baseline"]
    optimized = [m for m in models if m["algo_type"] == "optimized"]
    return {"baseline": baseline, "optimized": optimized,
            "all": models}
