# -*- coding: utf-8 -*-
"""统计学习模块单元测试。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from backend.algorithms import statistical as stat

RAW = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data", "raw")


@pytest.fixture(scope="module")
def raw_df():
    return stat.load_raw_train(os.path.join(RAW, "train_FD001.txt"))


def test_load_raw_shape(raw_df):
    assert raw_df.shape[1] == 26
    assert "unit" in raw_df.columns and "cycle" in raw_df.columns
    assert "s21" in raw_df.columns


def test_missing_detection(raw_df):
    info = stat.detect_missing(raw_df)
    assert info["total_missing"] == 0


def test_constant_sensors(raw_df):
    const = stat.detect_constant_sensors(raw_df)
    # FD001 已知常值传感器
    assert "s1" in const


def test_rul_label(raw_df):
    rul = stat.make_rul_label(raw_df)
    assert rul.max() <= stat.RUL_CAP
    assert (rul >= 0).all()


def test_noise_detection(raw_df):
    noise = stat.detect_noise(raw_df)
    assert "is_noise" in noise.columns
    assert noise["is_noise"].dtype in (np.int64, np.int32)
    rate = noise["is_noise"].mean()
    assert 0.0 <= rate <= 0.5


def test_feature_selection(raw_df):
    raw_df["RUL"] = stat.make_rul_label(raw_df)
    keep, detail = stat.feature_selection_pipeline(raw_df)
    # FD001 标准保留 14 特征
    assert len(keep) == 14
    assert "s1" not in keep
    assert detail["raw_sensors"] == 21


def test_split_no_leakage(raw_df):
    raw_df["RUL"] = stat.make_rul_label(raw_df)
    splits = stat.split_train_val_test(raw_df)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    # 同单元不得同时出现在 train 和 val
    overlap = set(tr["unit"]) & set(va["unit"])
    assert len(overlap) == 0
    # 测试集为每台最后 20% 周期
    max_cycle = te.groupby("unit")["cycle"].transform("max")
    assert (te["cycle"] > max_cycle * 0.8).all()


def test_group_kfold(raw_df):
    raw_df["RUL"] = stat.make_rul_label(raw_df)
    splits = stat.split_train_val_test(raw_df)
    folds = stat.group_kfold_indices(splits["train"], 5)
    assert len(folds) == 5
    for f in folds:
        assert len(f["train"]) > 0 and len(f["val"]) > 0
        tr_units = set(splits["train"].iloc[f["train"]]["unit"])
        va_units = set(splits["train"].iloc[f["val"]]["unit"])
        assert len(tr_units & va_units) == 0


def test_preprocess_pipeline(raw_df):
    splits, meta, scaler = stat.preprocess_pipeline(raw_df)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert meta["n_train"] == len(splits["train"])
    assert "features" in meta
    # 标准化特征应约服从 N(0,1)
    s = splits["train"]["cycle"]
    assert abs(s.mean()) < 0.05 and abs(s.var() - 1) < 0.05
