# -*- coding: utf-8 -*-
"""
C-MAPSS 涡扇发动机退化数据集 —— 数据预处理程序
================================================
对应《方案设计》"数据层"：
  使用统计学习方法完成异常检测、特征筛选、数据集划分、交叉验证设置，
  降低噪声干扰，缓解小样本缺陷。

功能：
  1) 数据清洗   ：缺失值检测、常值/近常值传感器剔除
  2) 噪声识别   ：基于 IQR 与 Z-score 的统计异常点识别（仅标记，不删除）
  3) 特征筛选   ：方差分析(低方差剔除) + 相关性分析(高相关冗余剔除) + 单因素方差分析(ANOVA)
  4) 标签构造   ：训练集生成分段线性 RUL 标签
  5) 数据集划分 ：按发动机单元划分 train / val / test
  6) 交叉验证   ：GroupKFold 5 折（按单元分组）
  7) 标准化     ：StandardScaler 仅在训练集拟合，避免数据泄漏

运行方式：
  python3 code/preprocess.py
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")

COLUMN_NAMES = ["unit", "cycle"] + [f"setting{i}" for i in range(1, 4)] + \
               [f"s{i}" for i in range(1, 22)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
SETTING_COLS = [f"setting{i}" for i in range(1, 4)]

RUL_CAP = 125
VARIANCE_THRESHOLD = 1e-3
CORR_THRESHOLD = 0.95
ANOVA_PVALUE = 0.05
N_FOLDS = 5
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_raw(subset: str):
    train = pd.read_csv(os.path.join(RAW_DIR, f"train_{subset}.txt"),
                        sep=r"\s+", header=None, names=COLUMN_NAMES)
    test = pd.read_csv(os.path.join(RAW_DIR, f"test_{subset}.txt"),
                       sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul = pd.read_csv(os.path.join(RAW_DIR, f"RUL_{subset}.txt"),
                      sep=r"\s+", header=None, names=["RUL"])
    test_rul = rul["RUL"].values
    test["RUL"] = test["unit"].map(
        dict(zip(np.sort(test["unit"].unique()), test_rul)))
    return train, test


def make_rul_label(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.Series:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return (max_cycle - df["cycle"]).clip(upper=cap)


def detect_missing(df: pd.DataFrame) -> dict:
    miss = df.isna().sum()
    return {"total_missing": int(miss.sum()),
            "missing_by_col": {str(k): int(v) for k, v in miss[miss > 0].items()}}


def detect_constant_sensors(df: pd.DataFrame) -> list:
    var = df[SENSOR_COLS].var()
    return [c for c in SENSOR_COLS if var[c] < 1e-12]


def detect_noise(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    noise_flags = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lb, ub = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        z = np.abs(stats.zscore(s))
        flag = ((s < lb) | (s > ub) | (z > 6)).astype(int)
        noise_flags[f"{c}_noise"] = flag
    noise_flags["noise_count"] = noise_flags.sum(axis=1)
    noise_flags["is_noise"] = (noise_flags["noise_count"] > 0).astype(int)
    return noise_flags


def feature_selection_variance(train, threshold):
    scaler = StandardScaler().fit(train[SENSOR_COLS])
    std = pd.DataFrame(scaler.transform(train[SENSOR_COLS]), columns=SENSOR_COLS)
    var = std.var()
    keep = [c for c in SENSOR_COLS if var[c] > threshold]
    removed = [c for c in SENSOR_COLS if var[c] <= threshold]
    return keep, removed


def feature_selection_correlation(train, keep, threshold):
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


def feature_selection_anova(train, keep, pvalue):
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


def split_train_val_test(df, val_ratio=0.15):
    units = df["unit"].unique()
    rng = np.random.RandomState(42)
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


def kfold_indices(df, n_folds=N_FOLDS):
    gkf = GroupKFold(n_splits=n_folds)
    groups = df["unit"].values
    folds = []
    for tr_idx, va_idx in gkf.split(df, groups=groups):
        folds.append({"train": tr_idx.tolist(), "val": va_idx.tolist()})
    return folds


def process_subset(subset):
    print(f"\n===== 处理子集 {subset} =====")
    train, test = load_raw(subset)
    train["RUL"] = make_rul_label(train)

    missing_info = detect_missing(train)
    const_sensors = detect_constant_sensors(train)
    noise_flag = detect_noise(train, SENSOR_COLS)
    train["is_noise"] = noise_flag["is_noise"].values
    noise_rate = float(train["is_noise"].mean())

    keep1, removed_var = feature_selection_variance(train, VARIANCE_THRESHOLD)
    keep2, removed_corr = feature_selection_correlation(train, keep1, CORR_THRESHOLD)
    keep3, removed_anova = feature_selection_anova(train, keep2, ANOVA_PVALUE)
    selected_features = keep3

    splits = split_train_val_test(train)
    folds = kfold_indices(splits["train"])

    scaler = StandardScaler()
    feat = ["cycle"] + selected_features
    scaler.fit(splits["train"][feat])

    subset_out = os.path.join(OUT_DIR, subset)
    ensure_dir(subset_out)

    keep_cols = ["unit", "cycle"] + selected_features + \
                ["RUL", "is_noise", "max_cycle"]
    for name in ["train", "val", "test"]:
        part = splits[name].copy()
        scaled = scaler.transform(part[feat])
        part[feat] = scaled
        part[keep_cols].to_csv(os.path.join(subset_out, f"{name}_clean.csv"),
                               index=False)

    feat_info = {
        "raw_sensors": len(SENSOR_COLS),
        "removed_by_variance": removed_var,
        "removed_by_correlation": removed_corr,
        "removed_by_anova": removed_anova,
        "selected_features": selected_features,
        "n_selected": len(selected_features),
    }
    with open(os.path.join(subset_out, "feature_selection.json"), "w",
              encoding="utf-8") as f:
        json.dump(feat_info, f, ensure_ascii=False, indent=2)

    with open(os.path.join(subset_out, "cv_folds.json"), "w",
              encoding="utf-8") as f:
        json.dump({"n_folds": N_FOLDS, "folds": folds}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(subset_out, "scaler.json"), "w",
              encoding="utf-8") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                   "features": feat}, f, ensure_ascii=False, indent=2)

    summary = {
        "subset": subset,
        "n_train_units": int(train["unit"].nunique()),
        "n_test_units": int(test["unit"].nunique()),
        "train_samples": int(len(splits["train"])),
        "val_samples": int(len(splits["val"])),
        "test_samples": int(len(splits["test"])),
        "missing": missing_info,
        "constant_sensors": const_sensors,
        "noise_rate": round(noise_rate, 4),
        "selected_features": selected_features,
        "n_selected_features": len(selected_features),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    ensure_dir(OUT_DIR)
    all_summaries = []
    for subset in SUBSETS:
        all_summaries.append(process_subset(subset))
    with open(os.path.join(OUT_DIR, "preprocess_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print("\n===== 全部子集处理完成 =====")
    print("输出目录:", OUT_DIR)


if __name__ == "__main__":
    main()
