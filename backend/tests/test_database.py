# -*- coding: utf-8 -*-
"""SQLite 数据库层单元测试。"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from backend import database as db


@pytest.fixture()
def tmp_db(tmp_path):
    old = db.DB_PATH
    db.DB_PATH = str(tmp_path / "test.db")
    db.init_db()
    yield db
    db.DB_PATH = old


def test_dataset_crud(tmp_db):
    tmp_db.upsert_dataset("FD001", "test", 100, 50, 14, 0.05)
    ds = tmp_db.list_datasets()
    assert len(ds) == 1
    assert ds[0]["name"] == "FD001"
    assert ds[0]["n_features"] == 14
    # 幂等更新
    tmp_db.upsert_dataset("FD001", "test2", 200, 60, 14, 0.06)
    ds = tmp_db.list_datasets()
    assert len(ds) == 1
    assert ds[0]["n_train"] == 200


def test_model_crud(tmp_db):
    tmp_db.add_model("FD001_LSTM", "baseline", "FD001", 121.0, 50000,
                     0.06, 12.0, 15.0)
    models = tmp_db.list_models()
    assert len(models) == 1
    assert models[0]["algo_type"] == "baseline"
    sub = tmp_db.get_models_by_dataset("FD001")
    assert len(sub) == 1


def test_predictions_crud(tmp_db):
    tmp_db.add_predictions([
        {"dataset": "FD001", "model_name": "m", "unit": 2, "last_cycle": 100,
         "true_rul": 20.0, "pred_rul": 22.0},
    ])
    rows = tmp_db.list_predictions()
    assert len(rows) == 1
    assert abs(rows[0]["abs_error"] - 2.0) < 1e-6


def test_experiment_crud(tmp_db):
    tmp_db.add_experiment("exp1", {"a": 1}, {"mae": 12.0})
    rows = tmp_db.list_experiments()
    assert len(rows) == 1
    assert rows[0]["name"] == "exp1"
