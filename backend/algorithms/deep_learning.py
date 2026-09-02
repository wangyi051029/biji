# -*- coding: utf-8 -*-
"""
深度学习算法模块（对应《制造智能技术》技术方向之一：深度学习）
==============================================================
基于 PyTorch 的 LSTM 时序深度网络，用于工业设备剩余寿命（RUL）预测。
  - 滑动窗口构造时序样本（window=30，stride=1）
  - LSTM 网络：输入窗口特征 → 双层 LSTM → 全连接回归输出 RUL
  - 训练：Adam + 权重衰减(正则化) + Dropout + 早停 + 学习率调度
  - 评估：MAE / RMSE / 得分函数
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 保证可复现
torch.manual_seed(42)
np.random.seed(42)


def make_windows(df, feature_cols, window=30):
    """
    按发动机单元构造滑动窗口样本。
    返回 X: (n_samples, window, n_features), y: (n_samples,)
    """
    xs, ys, units, last_cycles = [], [], [], []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        data = g[feature_cols].values.astype(np.float32)
        rul = g["RUL"].values.astype(np.float32)
        n = len(data)
        if n < window:
            continue
        for i in range(n - window + 1):
            xs.append(data[i:i + window])
            ys.append(rul[i + window - 1])
            units.append(unit)
            last_cycles.append(int(g["cycle"].iloc[i + window - 1]))
    X = np.stack(xs) if xs else np.zeros((0, window, len(feature_cols)),
                                         dtype=np.float32)
    y = np.array(ys, dtype=np.float32) if ys else np.zeros(0, dtype=np.float32)
    return X, y, np.array(units), np.array(last_cycles)


class LSTMRULModel(nn.Module):
    """双层 LSTM + 全连接回归头。"""

    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class RULPredictor:
    """LSTM RUL 预测器封装：训练 / 评估 / 预测。"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2,
                 lr=1e-3, weight_decay=1e-4, window=30, epochs=20,
                 batch_size=64, patience=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMRULModel(input_size, hidden_size, num_layers, dropout)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss，对噪声鲁棒
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2)
        self.history = {"train_loss": [], "val_loss": []}
        self.best_state = None

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, verbose=True):
        ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        best_val = float("inf")
        wait = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss, n = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item() * len(xb)
                n += len(xb)
            train_loss = total_loss / n
            self.history["train_loss"].append(train_loss)

            # 验证与早停
            val_loss = train_loss
            if X_va is not None and len(X_va) > 0:
                val_loss = self.evaluate(X_va, y_va)["loss"]
                self.scheduler.step(val_loss)
            self.history["val_loss"].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                self.best_state = {k: v.clone() for k, v in
                                   self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    if verbose:
                        print(f"早停 @ epoch {epoch}, 最优验证损失 {best_val:.4f}")
                    break
            if verbose and epoch % 5 == 0:
                print(f"epoch {epoch}: train_loss={train_loss:.4f} "
                      f"val_loss={val_loss:.4f} lr={self.optimizer.param_groups[0]['lr']:.2e}")
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self

    def predict(self, X):
        self.model.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            pred = self.model(Xt).cpu().numpy()
        return np.clip(pred, 0, None)

    def evaluate(self, X, y):
        self.model.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            pred = self.model(Xt)
            loss = self.criterion(pred, yt).item()
        pred_np = np.clip(pred.cpu().numpy(), 0, None)
        y_np = y
        mae = float(np.mean(np.abs(pred_np - y_np)))
        rmse = float(np.sqrt(np.mean((pred_np - y_np) ** 2)))
        return {"loss": loss, "mae": mae, "rmse": rmse}

    def params_size(self):
        """模型参数量（单位：K）。"""
        return sum(p.numel() for p in self.model.parameters()) / 1000.0

    def inference_time(self, X, repeat=5):
        """推理耗时（毫秒 / 样本）。"""
        self.model.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            t0 = time.perf_counter()
            for _ in range(repeat):
                self.model(Xt)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            total = time.perf_counter() - t0
        return total / repeat / len(X) * 1000.0

    def save(self, path, meta=None):
        torch.save(self.model.state_dict(), path)
        if meta is not None:
            meta = dict(meta)
            # 从模型实际架构推导，保证 meta 与权重一致
            lstm = self.model.lstm
            meta["arch"] = {
                "input_size": lstm.input_size,
                "hidden_size": lstm.hidden_size,
                "num_layers": lstm.num_layers,
                "dropout": self.dropout,
            }
            with open(path + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, path, input_size=None):
        meta_path = path + ".meta.json"
        arch = None
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            arch = meta.get("arch")
            if "window" in meta:
                self.window = meta["window"]
        if arch is not None:
            self.model = LSTMRULModel(
                arch["input_size"], arch["hidden_size"],
                arch["num_layers"], arch.get("dropout", 0.2))
        elif input_size is not None and self.model.lstm.input_size != input_size:
            self.model = LSTMRULModel(input_size,
                                      self.model.lstm.hidden_size,
                                      self.model.lstm.num_layers)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        return self
