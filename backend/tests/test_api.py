# -*- coding: utf-8 -*-
"""API 集成测试（FastAPI TestClient）。"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from backend import database as db
from backend import services
from backend.main import app


@pytest.fixture(scope="module", autouse=True)
def _tmp_db(tmp_path_factory):
    """将 API 测试隔离到临时数据库与临时模型目录，避免污染正式数据。"""
    tmp = tmp_path_factory.mktemp("api_test")
    old_db, old_model = db.DB_PATH, services.MODEL_DIR
    db.DB_PATH = str(tmp / "test_app.db")
    services.MODEL_DIR = str(tmp / "models")
    db.init_db()  # 显式建表（新版 FastAPI 需 context manager 才触发 startup）
    yield
    db.DB_PATH, services.MODEL_DIR = old_db, old_model


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_datasets(client):
    r = client.get("/api/datasets")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_preprocess(client):
    r = client.post("/api/preprocess", json={"subset": "FD001"})
    assert r.status_code == 200
    meta = r.json()["meta"]
    assert "selected_features" in meta["feature_selection"]


def test_preprocess_bad_subset(client):
    r = client.post("/api/preprocess", json={"subset": "BAD"})
    assert r.status_code == 404


def test_features(client):
    r = client.get("/api/features/FD001")
    assert r.status_code == 200
    assert len(r.json()["selected_features"]) > 0


def test_train_and_predict(client):
    # 训练一个小模型
    r = client.post("/api/model/train", json={
        "subset": "FD001", "n_samples": 100, "epochs": 2, "hidden": 32})
    assert r.status_code == 200
    info = r.json()
    assert info["algo_type"] == "baseline"

    # 在线预测
    r2 = client.post("/api/model/predict", json={
        "subset": "FD001", "model_name": info["name"], "unit": 2})
    assert r2.status_code == 200
    assert "pred_rul" in r2.json()


def test_optimize(client):
    # 先确保有基准模型
    client.post("/api/model/train", json={
        "subset": "FD001", "n_samples": 100, "epochs": 2, "hidden": 32})
    models = client.get("/api/model/list").json()
    baseline = [m for m in models if m["algo_type"] == "baseline"]
    assert baseline
    r = client.post("/api/optimize", json={
        "subset": "FD001", "model_name": baseline[0]["name"],
        "prune_ratio": 0.3, "quant_bits": 8})
    assert r.status_code == 200
    assert "compare" in r.json()


def test_compare(client):
    r = client.get("/api/compare/FD001")
    assert r.status_code == 200
    assert "baseline" in r.json() and "optimized" in r.json()


def test_predictions(client):
    r = client.get("/api/predictions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
