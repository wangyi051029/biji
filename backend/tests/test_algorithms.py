# -*- coding: utf-8 -*-
"""深度学习与智能计算优化模块单元测试。"""
import os
import sys

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from backend.algorithms import deep_learning as dl
from backend.algorithms import optimization as opt


@pytest.fixture(scope="module")
def window_data():
    # 构造模拟时序：3 特征，20 台单元，每台 50 周期
    rows = []
    for u in range(1, 21):
        for c in range(1, 51):
            rows.append({"unit": u, "cycle": c,
                         "s2": c + np.random.randn() * 0.5,
                         "s3": 2 * c + np.random.randn(),
                         "s4": 0.5 * c, "RUL": 50 - c})
    return rows


def test_make_windows(window_data):
    import pandas as pd
    df = pd.DataFrame(window_data)
    feats = ["s2", "s3", "s4"]
    X, y, units, last = dl.make_windows(df, feats, window=10)
    assert X.shape[1] == 10 and X.shape[2] == 3
    assert X.shape[0] == 20 * (50 - 10 + 1)  # 20台 × 每台41个窗口
    assert len(y) == len(X)


def test_lstm_train_predict(window_data):
    import pandas as pd
    df = pd.DataFrame(window_data)
    feats = ["s2", "s3", "s4"]
    X, y, _, _ = dl.make_windows(df, feats, window=10)
    n = min(120, len(X))
    X, y = X[:n], y[:n]
    split = int(n * 0.8)
    pred = dl.RULPredictor(input_size=3, epochs=3, hidden_size=16)
    pred.fit(X[:split], y[:split], X[split:], y[split:], verbose=False)
    out = pred.predict(X[split:])
    assert out.shape == (n - split,)
    assert (out >= 0).all()
    m = pred.evaluate(X[split:], y[split:])
    assert m["mae"] >= 0 and m["rmse"] >= 0
    assert pred.params_size() > 0


def test_save_load_roundtrip(tmp_path, window_data):
    import pandas as pd
    df = pd.DataFrame(window_data)
    feats = ["s2", "s3", "s4"]
    X, y, _, _ = dl.make_windows(df, feats, window=10)
    X, y = X[:80], y[:80]
    pred = dl.RULPredictor(input_size=3, epochs=2, hidden_size=16)
    pred.fit(X, y, verbose=False)
    path = str(tmp_path / "model.pt")
    pred.save(path, meta={"subset": "FD001", "features": feats, "window": 10})
    pred2 = dl.RULPredictor(input_size=3, hidden_size=16)
    pred2.load(path)
    out1 = pred.predict(X[:5])
    out2 = pred2.predict(X[:5])
    assert np.allclose(out1, out2, atol=1e-4)


def test_pruning(window_data):
    import pandas as pd
    df = pd.DataFrame(window_data)
    feats = ["s2", "s3", "s4"]
    X, y, _, _ = dl.make_windows(df, feats, window=10)
    pred = dl.RULPredictor(input_size=3, epochs=2, hidden_size=16)
    pred.fit(X[:80], y[:80], verbose=False)
    pruned, stat = opt.prune_model_weight_magnitude(pred.model, 0.3)
    assert 0.2 <= stat["sparsity"] <= 0.4
    # 剪枝后输出仍可前向
    out = pruned(torch.from_numpy(X[:5])).detach().numpy()
    assert out.shape == (5,)


def test_quantization(window_data):
    import pandas as pd
    df = pd.DataFrame(window_data)
    feats = ["s2", "s3", "s4"]
    X, y, _, _ = dl.make_windows(df, feats, window=10)
    pred = dl.RULPredictor(input_size=3, epochs=2, hidden_size=16)
    pred.fit(X[:80], y[:80], verbose=False)
    qmodel, qstat = opt.quantize_model_sim(pred.model, 8)
    assert qstat["bits"] == 8
    # 量化后数值范围受限
    with torch.no_grad():
        out = qmodel(torch.from_numpy(X[:5]))
    assert out.shape == (5,)


def test_compare_models(window_data):
    import pandas as pd
    df = pd.DataFrame(window_data)
    feats = ["s2", "s3", "s4"]
    X, y, _, _ = dl.make_windows(df, feats, window=10)
    pred = dl.RULPredictor(input_size=3, epochs=2, hidden_size=16)
    pred.fit(X[:80], y[:80], verbose=False)
    pruned, _ = opt.prune_model_weight_magnitude(pred.model, 0.3)
    qmodel, _ = opt.quantize_model_sim(pruned, 8)
    comp = opt.compare_models(pred.model, qmodel, X[:20], y[:20], quant_bits=8)
    assert "base" in comp and "optimized" in comp
    assert comp["optimized"]["model_bytes"] < comp["base"]["model_bytes"]
    assert comp["size_reduction"] > 0
