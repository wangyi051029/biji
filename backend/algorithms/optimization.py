# -*- coding: utf-8 -*-
"""
智能计算优化算法模块（对应《制造智能技术》技术方向之一：优化智能计算）
================================================================
在深度学习模型基础上实施轻量化与训练策略优化：
  - 模型剪枝（Pruning）：按权重幅值剪枝，稀疏化冗余连接
  - 参数量化（Quantization）：float32 → int8 模拟量化，减小存储与计算开销
  - 学习率调度（Learning Rate Scheduling）：余弦退火 / 阶梯式衰减
  - 压缩对比：剪枝/量化前后 参数量、模型大小、推理耗时、精度对比
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn


def prune_model_weight_magnitude(model, prune_ratio=0.3):
    """
    基于权重幅值的非结构化剪枝：将 |weight| 最小的 prune_ratio 比例置零。
    返回 (剪枝后模型, 剪枝统计)。
    """
    model = copy.deepcopy(model)
    total = 0
    zeroed = 0
    pruned_spec = {}
    for name, param in model.named_parameters():
        if param.dim() < 2:  # 不剪偏置
            continue
        w = param.data
        total += w.numel()
        threshold = np.percentile(w.abs().cpu().numpy(), prune_ratio * 100)
        mask = w.abs() > threshold
        zeroed += int((~mask).sum())
        param.data.mul_(mask)
        pruned_spec[name] = {
            "params": int(w.numel()),
            "zeroed": int((~mask).sum()),
        }
    return model, {
        "method": "weight_magnitude",
        "prune_ratio": prune_ratio,
        "total_params": int(total),
        "zeroed_params": int(zeroed),
        "sparsity": round(zeroed / total, 4),
    }


def quantize_model_sim(model, bits=8):
    """
    参数量化模拟：将权重线性量化为 bits 位整数再反量化。
    返回 (量化后模型, 量化统计)。
    """
    model = copy.deepcopy(model)
    qmin, qmax = 0, 2 ** bits - 1
    quantized = {}
    for name, param in model.named_parameters():
        w = param.data
        zero_mask = (w == 0)
        wmin, wmax = w.min(), w.max()
        scale = (wmax - wmin) / (qmax - qmin) if wmax > wmin else 1.0
        zero_point = qmin - wmin / scale if scale else 0.0
        q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax)
        wq = (q - zero_point) * scale
        wq = wq.masked_fill(zero_mask, 0.0)  # 保留剪枝的精确零值
        param.data.copy_(wq)
        quantized[name] = {"scale": float(scale), "zero_point": float(zero_point)}
    return model, {"method": "sim_quantization", "bits": bits,
                   "stat": quantized}


def get_learning_rate_scheduler(optimizer, mode="cosine", total_epochs=20,
                                step_size=5, gamma=0.5):
    """返回学习率调度器。mode: 'cosine'(余弦退火) / 'step'(阶梯衰减) / 'none'。"""
    if mode == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs)
    if mode == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
    return None


def compare_models(base_model, opt_model, X, y=None, repeat=5,
                   quant_bits=None):
    """
    压缩前后模型对比：参数量、模型文件大小、推理耗时、精度（可选）。
    - 参数量：剪枝后按非零参数计（剪枝生效）
    - 模型大小：量化后按 quant_bits/8 字节每参数计（量化生效）
    返回对比结果 dict。
    """
    def effective_params(m):
        total = 0
        for p in m.parameters():
            total += int((p != 0).sum().item())
        return total

    def model_size_bytes(m, bits=None):
        total = 0
        for p in m.parameters():
            nonzero = int((p != 0).sum().item())
            elem = (bits / 8.0) if bits else p.element_size()
            total += int(nonzero * elem)
        return total

    def infer_ms(m):
        m.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(repeat):
                m(Xt)
            total = time.perf_counter() - t0
        return total / repeat / len(X) * 1000.0

    def metric(m):
        if y is None:
            return None
        m.eval()
        with torch.no_grad():
            pred = m(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy()
        pred = np.clip(pred, 0, None)
        return {"mae": float(np.mean(np.abs(pred - y))),
                "rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}

    base_size = model_size_bytes(base_model)
    opt_size = model_size_bytes(opt_model, quant_bits)
    result = {
        "base": {
            "params_k": round(effective_params(base_model) / 1000.0, 3),
            "model_bytes": int(base_size),
            "inference_ms": round(infer_ms(base_model), 4),
            "metric": metric(base_model),
        },
        "optimized": {
            "params_k": round(effective_params(opt_model) / 1000.0, 3),
            "model_bytes": int(opt_size),
            "inference_ms": round(infer_ms(opt_model), 4),
            "metric": metric(opt_model),
        },
    }
    result["speedup"] = round(result["base"]["inference_ms"] /
                              max(result["optimized"]["inference_ms"], 1e-9), 2)
    result["size_reduction"] = round(1 - opt_size / max(base_size, 1e-9), 4)
    result["param_reduction"] = round(
        1 - result["optimized"]["params_k"] / max(result["base"]["params_k"], 1e-9), 4)
    return result
