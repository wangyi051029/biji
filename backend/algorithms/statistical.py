# -*- coding: utf-8 -*-
"""
统计学习算法模块（对应《制造智能技术》技术方向之一：统计学习）
==============================================================
提供：
  - 数据清洗：缺失值检测、常值/近常值传感器识别
  - 噪声识别：IQR 准则 + Z-score 准则（统计异常检测）
  - 特征筛选：方差分析 + 相关性分析 + 单因素方差分析(ANOVA)
  - RUL 标签构造：分段线性剩余寿命
  - 数据集划分：按单元划分 train/val/test
  - k 折交叉验证：GroupKFold（按单元分组，防泄漏）
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

COLUMN_NAMES = ["unit", "cycle"] + [f"setting{i}" for i in range(1, 4)] + \
               [f"s{i}" for i in range(1, 22)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125
DEFAULT_N_FOLDS = 5


def load_raw_train(path):
    """加载训练原始数据并附加列名。"""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)


def detect_missing(df):
    """缺失值检测。"""
    miss = df.isna().sum()
    return {"total_missing": int(miss.sum()),
            "missing_by_col": {str(k): int(v) for k, v in miss[miss > 0].items()}}


def detect_constant_sensors(df):
    """常值 / 近常值传感器识别。"""
    var = df[SENSOR_COLS].var()
    return [c for c in SENSOR_COLS if var[c] < 1e-12]


def detect_noise(df, cols=None):
    """
    基于 IQR 与 Z-score 的统计噪声/异常点识别。
    返回 DataFrame：每传感器一个噪声标记列 + is_noise 汇总标记。
    """
    cols = cols or SENSOR_COLS
    noise_flags = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].replace([np.inf, -np.inf], np.nan)
        s = s.fillna(s.median())
        # 常值/近常值传感器无噪声可识别，跳过（避免 zscore 数值警告）
        if s.std() < 1e-8:
            noise_flags[f"{c}_noise"] = 0
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lb, ub = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.abs(stats.zscore(s, nan_policy="omit"))
        flag = ((s < lb) | (s > ub) | (z > 6)).astype(int)
        noise_flags[f"{c}_noise"] = flag
    noise_flags["noise_count"] = noise_flags.sum(axis=1)
    noise_flags["is_noise"] = (noise_flags["noise_count"] > 0).astype(int)
    return noise_flags


def make_rul_label(df, cap=RUL_CAP):
    """分段线性 RUL 标签：RUL = min(max_cycle - cycle, cap)。"""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return (max_cycle - df["cycle"]).clip(upper=cap)


def feature_selection_variance(train, threshold=1e-3):
    """方差分析：标准化后剔除低方差特征。"""
    scaler = StandardScaler().fit(train[SENSOR_COLS])
    std = pd.DataFrame(scaler.transform(train[SENSOR_COLS]), columns=SENSOR_COLS)
    var = std.var()
    keep = [c for c in SENSOR_COLS if var[c] > threshold]
    removed = [c for c in SENSOR_COLS if var[c] <= threshold]
    return keep, removed


def feature_selection_correlation(train, keep, threshold=0.95):
    """相关性分析：剔除高度相关冗余特征，保留与 RUL 相关性更高的。"""
    keep = list(keep)
    corr = train[keep].corr()
    removed = []
    cols = list(keep)
    i = 0
    while i < len(cols):
        c = cols[i]
        high = [o for o in cols[i + 1:] if abs(corr.loc[c, o]) > threshold]
        for o in high:
            c_rul = abs(train[c].corr(train["RUL"]))
            o_rul = abs(train[o].corr(train["RUL"]))
            if o_rul > c_rul:
                removed.append(c)
                cols.remove(c)
                break
            else:
                removed.append(o)
                cols.remove(o)
        i += 1
    keep_final = [c for c in keep if c not in removed]
    return keep_final, removed


def feature_selection_anova(train, keep, pvalue=0.05):
    """单因素方差分析：检验传感器随退化阶段均值差异是否显著。"""
    df = train.copy()
    df["life_bin"] = pd.qcut(df["cycle"], 10, labels=False)
    keep = list(keep)
    kept, removed = [], []
    for c in keep:
        groups = [df.loc[df["life_bin"] == b, c].values
                  for b in sorted(df["life_bin"].unique())]
        try:
            _, p = stats.f_oneway(*groups)
        except Exception:
            p = 1.0
        if np.isnan(p):
            p = 1.0
        if p < pvalue:
            kept.append(c)
        else:
            removed.append(c)
    return kept, removed


def feature_selection_pipeline(train, var_th=1e-3, corr_th=0.95, pv=0.05):
    """特征筛选完整流水线，返回 (保留特征, 明细)。"""
    keep1, r1 = feature_selection_variance(train, var_th)
    keep2, r2 = feature_selection_correlation(train, keep1, corr_th)
    keep3, r3 = feature_selection_anova(train, keep2, pv)
    detail = {
        "raw_sensors": len(SENSOR_COLS),
        "removed_by_variance": r1,
        "removed_by_correlation": r2,
        "removed_by_anova": r3,
        "selected_features": keep3,
    }
    return keep3, detail


def split_train_val_test(df, val_ratio=0.15, seed=42):
    """按发动机单元划分 train / val / test（test 取每台最后 20% 周期）。"""
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["max_cycle"] = df.groupby("unit")["cycle"].transform("max")
    test_mask = df["cycle"] > (df["max_cycle"] * 0.8)
    train_part = df[~test_mask].copy()
    test_part = df[test_mask].copy()
    tr_units = train_part["unit"].unique()
    val_units = set(rng.choice(tr_units, size=int(len(tr_units) * val_ratio),
                               replace=False))
    val_part = train_part[train_part["unit"].isin(val_units)].copy()
    tr_part = train_part[~train_part["unit"].isin(val_units)].copy()
    return {"train": tr_part, "val": val_part, "test": test_part}


def group_kfold_indices(df, n_folds=DEFAULT_N_FOLDS):
    """GroupKFold 交叉验证索引（按单元分组，同机不跨折）。"""
    gkf = GroupKFold(n_splits=n_folds)
    folds = []
    for tr_idx, va_idx in gkf.split(df, groups=df["unit"].values):
        folds.append({"train": tr_idx.tolist(), "val": va_idx.tolist()})
    return folds


def fit_scaler(df, features):
    """训练 StandardScaler 并返回实例。"""
    scaler = StandardScaler().fit(df[features])
    return scaler


def preprocess_pipeline(raw_df, var_th=1e-3, corr_th=0.95, pv=0.05,
                        val_ratio=0.15, n_folds=DEFAULT_N_FOLDS):
    """
    完整预处理流水线（与 code/preprocess.py 逻辑一致）：
    清洗 → 噪声识别 → 特征筛选 → RUL 标签 → 划分 → 交叉验证 → 标准化。
    返回 (标准化后的划分 dict, 元信息 dict)。
    """
    df = raw_df.copy()
    df["RUL"] = make_rul_label(df)

    # 清洗与噪声
    missing = detect_missing(df)
    const_sensors = detect_constant_sensors(df)
    noise = detect_noise(df, SENSOR_COLS)
    df["is_noise"] = noise["is_noise"].values
    noise_rate = float(df["is_noise"].mean())

    # 特征筛选
    selected, fs_detail = feature_selection_pipeline(df, var_th, corr_th, pv)
    feat = ["cycle"] + selected

    # 划分
    splits = split_train_val_test(df, val_ratio)

    # 交叉验证
    folds = group_kfold_indices(splits["train"], n_folds)

    # 标准化（仅在训练集拟合）
    scaler = StandardScaler().fit(splits["train"][feat])
    for name in ["train", "val", "test"]:
        part = splits[name].copy()
        part[feat] = scaler.transform(part[feat])
        splits[name] = part

    meta = {
        "missing": missing,
        "constant_sensors": const_sensors,
        "noise_rate": round(noise_rate, 4),
        "feature_selection": fs_detail,
        "n_train": int(len(splits["train"])),
        "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
        "n_train_units": int(splits["train"]["unit"].nunique()),
        "n_folds": n_folds,
        "features": feat,
    }
    return splits, meta, scaler
