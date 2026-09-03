# -*- coding: utf-8 -*-
"""
SQLite 数据库层
==============
管理课程设计系统的数据持久化：
  - datasets    数据集元信息（子集、规模、特征、预处理结果）
  - models      模型记录（参数、压缩状态、指标）
  - predictions 预测结果记录（输入、真值、预测值、误差）
  - experiments 实验记录（配置与指标 JSON）
"""

import os
import json
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,   -- 子集名，如 FD001
    description TEXT,
    n_train     INTEGER,
    n_test      INTEGER,
    n_features  INTEGER,
    noise_rate  REAL,
    created_at  TEXT
);
CREATE TABLE IF NOT EXISTS models (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    algo_type   TEXT NOT NULL,           -- baseline / pruned / quantized
    dataset     TEXT,
    params_k    REAL,
    model_bytes INTEGER,
    inference_ms REAL,
    mae         REAL,
    rmse        REAL,
    sparsity    REAL,
    quant_bits   INTEGER,
    created_at  TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset     TEXT,
    model_name  TEXT,
    unit        INTEGER,
    last_cycle  INTEGER,
    true_rul    REAL,
    pred_rul    REAL,
    abs_error   REAL,
    created_at  TEXT
);
CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    config      TEXT,                    -- JSON
    metrics     TEXT,                    -- JSON
    created_at  TEXT
);
"""


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------- datasets ----------------
def upsert_dataset(name, description="", n_train=0, n_test=0, n_features=0,
                   noise_rate=0.0):
    conn = get_conn()
    conn.execute("""
        INSERT INTO datasets(name, description, n_train, n_test, n_features,
                             noise_rate, created_at)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(name) DO UPDATE SET
          description=excluded.description, n_train=excluded.n_train,
          n_test=excluded.n_test, n_features=excluded.n_features,
          noise_rate=excluded.noise_rate
    """, (name, description, n_train, n_test, n_features, noise_rate, now_str()))
    conn.commit()
    conn.close()


def list_datasets():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM datasets ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------- models ----------------
def add_model(name, algo_type, dataset, params_k, model_bytes, inference_ms,
              mae=None, rmse=None, sparsity=None, quant_bits=None):
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO models(name, algo_type, dataset, params_k, model_bytes,
                           inference_ms, mae, rmse, sparsity, quant_bits, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (name, algo_type, dataset, params_k, model_bytes, inference_ms,
          mae, rmse, sparsity, quant_bits, now_str()))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def list_models():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM models ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_models_by_dataset(dataset):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM models WHERE dataset=? ORDER BY id", (dataset,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------- predictions ----------------
def add_predictions(records):
    """records: list of dict(dataset, model_name, unit, last_cycle,
                            true_rul, pred_rul)"""
    conn = get_conn()
    conn.executemany("""
        INSERT INTO predictions(dataset, model_name, unit, last_cycle,
                                true_rul, pred_rul, abs_error, created_at)
        VALUES (:dataset, :model_name, :unit, :last_cycle, :true_rul,
                :pred_rul, :abs_error, :created_at)
    """, [{
        **r, "abs_error": abs((r.get("true_rul") or 0) - (r.get("pred_rul") or 0)),
        "created_at": now_str(),
    } for r in records])
    conn.commit()
    conn.close()


def list_predictions(limit=100):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------- experiments ----------------
def add_experiment(name, config, metrics):
    conn = get_conn()
    conn.execute("""
        INSERT INTO experiments(name, config, metrics, created_at)
        VALUES (?,?,?,?)
    """, (name, json.dumps(config, ensure_ascii=False),
          json.dumps(metrics, ensure_ascii=False), now_str()))
    conn.commit()
    conn.close()


def list_experiments():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]
