# -*- coding: utf-8 -*-
"""
设备工况智能预测系统 —— 后端服务入口（FastAPI）
================================================
启动方式：
  cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
或
  python3 -m uvicorn main:app --port 8000

接口一览：
  GET  /                        前端静态页面
  GET  /api/health              健康检查
  GET  /api/datasets            数据集列表（数据库）
  POST /api/preprocess          预处理指定子集
  POST /api/preprocess/all      预处理全部子集
  GET  /api/features/{subset}   特征筛选结果
  POST /api/model/train         训练 LSTM 基准模型
  POST /api/model/predict       在线预测 RUL
  GET  /api/model/list          已训练模型列表
  POST /api/optimize            模型剪枝 + 量化优化
  GET  /api/compare/{subset}    基准 vs 优化对比
  GET  /api/predictions         预测记录
"""

import os
import sys
import json

# 保证 backend 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend import services
from backend import database as db

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    services.ensure_dirs()
    yield


app = FastAPI(title="设备工况智能预测系统", version="1.0.0",
              lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"])

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "frontend")


# ---------------- 请求模型 ----------------
class PreprocessReq(BaseModel):
    subset: str = "FD001"


class TrainReq(BaseModel):
    subset: str = "FD001"
    n_samples: int = Field(60, ge=30, le=2000)
    window: int = Field(30, ge=10, le=60)
    epochs: int = Field(10, ge=1, le=100)
    hidden: int = Field(64, ge=8, le=256)
    lr: float = Field(1e-3, gt=0)
    name: str | None = None


class PredictReq(BaseModel):
    subset: str = "FD001"
    model_name: str
    unit: int = Field(2, ge=1)
    n_cycles: int = Field(30, ge=10, le=60)


class OptimizeReq(BaseModel):
    subset: str = "FD001"
    model_name: str
    prune_ratio: float = Field(0.3, gt=0, lt=1)
    quant_bits: int = Field(8, ge=4, le=16)
class PredictManualReq(BaseModel):
    subset: str = "FD001"
    model_name: str
    rows: list[list[float]]


# ---------------- 基础 ----------------
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "设备工况智能预测系统"}


# ---------------- 数据服务 ----------------
@app.get("/api/datasets")
def list_datasets():
    return db.list_datasets()


@app.post("/api/preprocess")
def preprocess(req: PreprocessReq):
    try:
        result = services.preprocess_subset(req.subset)
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/preprocess/all")
def preprocess_all():
    return services.preprocess_all()


@app.get("/api/features/{subset}")
def get_features(subset: str):
    path = os.path.join(services.PROCESSED_DIR, subset, "feature_selection.json")
    if not os.path.exists(path):
        services.preprocess_subset(subset)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------- 模型服务 ----------------
@app.post("/api/model/train")
def train(req: TrainReq):
    try:
        return services.train_model(
            req.subset, n_samples=req.n_samples, window=req.window,
            epochs=req.epochs, hidden=req.hidden, lr=req.lr, name=req.name)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/model/predict")
def predict(req: PredictReq):
    try:
        return services.predict_rul(req.subset, req.model_name, req.unit, req.n_cycles)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/model/list")
def list_models():
    return db.list_models()
@app.get("/api/model/info/{model_name}")
def model_info(model_name: str):
    try:
        return services.get_model_info(model_name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
@app.post("/api/model/predict_manual")
def predict_manual(req: PredictManualReq):
    try:
        return services.predict_rul_manual(req.subset, req.model_name, req.rows)
    except Exception as e:
        raise HTTPException(400, str(e))


# ---------------- 优化服务 ----------------
@app.post("/api/optimize")
def optimize(req: OptimizeReq):
    try:
        return services.optimize_model(req.subset, req.model_name,
                                       req.prune_ratio, req.quant_bits)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/compare/{subset}")
def compare(subset: str):
    return services.get_compare_models(subset)


# ---------------- 记录 ----------------
@app.get("/api/predictions")
def predictions(limit: int = 100):
    return db.list_predictions(limit)


@app.get("/api/experiments")
def experiments():
    return db.list_experiments()


# ---------------- 前端静态资源 ----------------
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
