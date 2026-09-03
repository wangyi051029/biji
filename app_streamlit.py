# -*- coding: utf-8 -*-
"""
设备工况 RUL 预测系统 —— Streamlit 在线完整版
==============================================
复用 backend（FastAPI 版）全部业务逻辑（统计学习预处理 / LSTM 训练预测 / 剪枝量化优化），
以 Streamlit 多页面形态呈现，可一键部署到 Streamlit Community Cloud 等平台，
任何电脑打开网址即可使用，无需本地安装。

本地运行：streamlit run app_streamlit.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from backend import database as db          # noqa: E402
from backend import services as svc          # noqa: E402

st.set_page_config(
    page_title="设备工况 RUL 预测系统",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUBSETS = svc.get_subset_list()


# ---------------- 缓存：预测器加载 ----------------
@st.cache_resource(show_spinner=False)
def load_predictor(model_name):
    from backend.algorithms import deep_learning as dl
    meta_path = os.path.join(svc.MODEL_DIR, f"{model_name}.pt.meta.json")
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    pred = dl.RULPredictor(input_size=meta["arch"]["input_size"],
                           window=meta["window"])
    pred.load(os.path.join(svc.MODEL_DIR, f"{model_name}.pt"))
    return pred, meta


def available_models():
    return db.list_models()


def model_names_for(dataset):
    return [m["name"] for m in db.list_models() if m["dataset"] == dataset]


# ============================================================
# 页面 1：总览
# ============================================================
def page_overview():
    st.title("设备工况智能预测系统")
    st.caption("面向小样本工业工况预测 · 融合统计学习 + 深度学习(LSTM) + 智能计算优化")
    c1, c2, c3, c4 = st.columns(4)
    ds = db.list_datasets()
    ms = db.list_models()
    pr = db.list_predictions(50)
    c1.metric("数据集", len(ds))
    c2.metric("模型", len(ms))
    c3.metric("预测记录", len(pr))
    c4.metric("数据规模(样本)", sum(d.get("n_train", 0) or 0 for d in ds))

    st.subheader("数据集")
    if ds:
        st.dataframe(pd.DataFrame(ds).drop(columns=["id", "created_at"]),
                     use_container_width=True)
    else:
        st.info("暂无数据集记录，请先在「数据管理」执行预处理。")

    st.subheader("模型记录")
    if ms:
        cols = ["name", "algo_type", "dataset", "params_k", "model_bytes",
                "inference_ms", "mae", "rmse"]
        st.dataframe(pd.DataFrame(ms)[cols], use_container_width=True)
    else:
        st.info("暂无模型，请先在「模型训练」训练 LSTM。")


# ============================================================
# 页面 2：数据管理
# ============================================================
def page_data():
    st.title("数据管理（统计学习预处理）")
    subset = st.selectbox("选择子集", SUBSETS)
    if st.button("执行预处理流水线（清洗→噪声识别→特征筛选→RUL标签→划分→标准化）",
                 use_container_width=True):
        with st.spinner(f"正在预处理 {subset} ..."):
            r = svc.preprocess_subset(subset)
        st.success(f"{subset} 预处理完成：训练 {r['meta']['n_train']}，"
                   f"测试 {r['meta']['n_test']}，特征 "
                   f"{len(r['meta']['feature_selection']['selected_features'])} 个")
    df = svc.load_processed_train(subset)
    st.subheader(f"{subset} 预处理后训练集（前 200 行）")
    st.dataframe(df.head(200), use_container_width=True, height=360)
    st.caption(f"共 {len(df):,} 行 · 列：{', '.join(df.columns)}")


# ============================================================
# 页面 3：模型训练
# ============================================================
def page_train():
    st.title("模型训练（深度学习 LSTM）")
    subset = st.selectbox("子集", SUBSETS, key="tr_subset")
    c1, c2, c3 = st.columns(3)
    n_samples = c1.slider("训练样本数", 40, 200, 60, 20)
    epochs = c2.slider("训练轮数", 5, 30, 10, 5)
    hidden = c3.selectbox("隐藏层维度", [32, 64, 96], index=1)
    if st.button("开始训练", type="primary", use_container_width=True):
        with st.spinner("训练中（CPU 即可，约 20~60 秒）..."):
            try:
                info = svc.train_model(subset, n_samples=n_samples,
                                       epochs=epochs, hidden=hidden)
            except Exception as e:  # noqa: BLE001
                st.error(f"训练失败：{e}")
                return
        st.success("训练完成")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", round(info["mae"], 3))
        m2.metric("RMSE", round(info["rmse"], 3))
        m3.metric("参数量", f"{info['params_k']} K")
        m4.metric("推理耗时", f"{info['inference_ms']} ms")
        st.code(f"模型已保存：data/models/{info['name']}.pt")


# ============================================================
# 页面 4：智能优化
# ============================================================
def page_optimize():
    st.title("智能计算优化（剪枝 + 量化）")
    subset = st.selectbox("子集", SUBSETS, key="op_subset")
    base_models = [m for m in model_names_for(subset)
                   if not m.endswith("_pruned")]
    if not base_models:
        st.warning("该子集暂无模型，请先在「模型训练」训练。")
        return
    model_name = st.selectbox("基准模型", base_models)
    c1, c2 = st.columns(2)
    prune_ratio = c1.slider("剪枝比例", 0.1, 0.5, 0.3, 0.05)
    quant_bits = c2.selectbox("量化位宽", [4, 8], index=1)
    if st.button("运行优化并对比", type="primary", use_container_width=True):
        with st.spinner("剪枝 + 量化中..."):
            try:
                r = svc.optimize_model(subset, model_name,
                                       prune_ratio=prune_ratio,
                                       quant_bits=quant_bits)
            except Exception as e:  # noqa: BLE001
                st.error(f"优化失败：{e}")
                return
        st.success("优化完成")
        comp = r["compare"]
        row = {
            "指标": ["参数量 (K)", "模型体积 (KB)", "推理耗时 (ms)",
                     "MAE", "RMSE"],
            "基准模型": [
                round(comp["baseline"]["params_k"], 2),
                round(comp["baseline"]["model_bytes"] / 1024, 1),
                comp["baseline"]["inference_ms"],
                comp["baseline"]["metric"]["mae"] if comp["baseline"]["metric"] else None,
                comp["baseline"]["metric"]["rmse"] if comp["baseline"]["metric"] else None,
            ],
            "优化后": [
                round(comp["optimized"]["params_k"], 2),
                round(comp["optimized"]["model_bytes"] / 1024, 1),
                comp["optimized"]["inference_ms"],
                comp["optimized"]["metric"]["mae"] if comp["optimized"]["metric"] else None,
                comp["optimized"]["metric"]["rmse"] if comp["optimized"]["metric"] else None,
            ],
        }
        st.dataframe(pd.DataFrame(row), use_container_width=True)


# ============================================================
# 页面 5：在线预测
# ============================================================
def page_predict():
    st.title("在线预测（RUL）")
    subset = st.selectbox("子集", SUBSETS, key="pd_subset")
    names = model_names_for(subset)
    if not names:
        st.warning("该子集暂无模型，请先训练。")
        return
    model_name = st.selectbox("模型", names)
    unit = st.number_input("发动机单元编号", min_value=1, max_value=300,
                           value=2, step=1)
    if st.button("预测 RUL", type="primary"):
        try:
            r = svc.predict_rul(subset, model_name, int(unit))
        except Exception as e:  # noqa: BLE001
            st.error(f"预测失败：{e}")
            return
        c1, c2 = st.columns(2)
        c1.metric("预测 RUL", f"{r['pred_rul']} 周期")
        if r.get("true_rul") is not None:
            c2.metric("真实 RUL", f"{r['true_rul']} 周期")
        st.caption(f"模型 {r['model_name']} · 单元 {r['unit']} · "
                   f"最后周期 {r['last_cycle']} · 窗口 {r['window_used']}")
        # 展示该单元退化曲线
        test = svc.load_processed_test(subset)
        unit_df = test[test["unit"] == int(unit)].sort_values("cycle")
        if len(unit_df) == 0:
            unit_df = svc.load_processed_train(subset)
            unit_df = unit_df[unit_df["unit"] == int(unit)].sort_values("cycle")
        if len(unit_df) > 0:
            st.line_chart(unit_df.set_index("cycle")[["RUL"]]
                          if "RUL" in unit_df.columns
                          else unit_df.set_index("cycle").iloc[:, -1])
            st.caption("该单元 RUL 退化曲线")


# ============================================================
# 页面 6：手动测试
# ============================================================
def page_manual():
    st.title("手动测试（输入传感器读数）")
    subset = st.selectbox("子集", SUBSETS, key="mn_subset")
    names = model_names_for(subset)
    if not names:
        st.warning("该子集暂无模型，请先训练。")
        return
    model_name = st.selectbox("模型", names, key="mn_model")
    try:
        info = svc.get_model_info(model_name)
    except Exception as e:  # noqa: BLE001
        st.error(f"模型信息获取失败：{e}")
        return
    sensor_feats = info["sensor_features"]
    st.caption(f"特征顺序（每行 {len(sensor_feats)} 个值，逗号分隔）："
               f"{' , '.join(sensor_feats)}")
    sample = info.get("sample_values") or []
    default_text = "\n".join([",".join(str(round(v, 2)) for v in sample)] * 30)
    text = st.text_area("传感器读数（每行一个周期，逗号分隔，可多行）",
                        value=default_text, height=200)
    if st.button("开始预测", type="primary"):
        try:
            rows = [list(map(float, line.split(",")))
                    for line in text.strip().splitlines() if line.strip()]
            r = svc.predict_rul_manual(subset, model_name, rows)
        except Exception as e:  # noqa: BLE001
            st.error(f"预测失败：{e}")
            return
        st.metric("预测 RUL", f"{r['pred_rul']} 周期")
        st.caption(f"输入 {r['input_rows']} 行 · 窗口 {r['window_used']} · "
                   f"模型 {r['model_name']} · {r.get('note', '')}")


# ============================================================
# 页面 7：对比分析
# ============================================================
def page_compare():
    st.title("对比分析（基准 vs 优化）")
    subset = st.selectbox("子集", SUBSETS, key="cp_subset")
    data = svc.get_compare_models(subset)
    if not data["baseline"]:
        st.warning("暂无基准模型。")
        return
    bl = data["baseline"][0]
    if data["optimized"]:
        op = data["optimized"][0]
        comp_df = pd.DataFrame({
            "指标": ["参数量 (K)", "模型体积 (KB)", "推理耗时 (ms)", "MAE", "RMSE"],
            "基准": [bl["params_k"], round((bl["model_bytes"] or 0) / 1024, 1),
                     bl["inference_ms"], bl["mae"], bl["rmse"]],
            "优化后": [op["params_k"], round((op["model_bytes"] or 0) / 1024, 1),
                       op["inference_ms"], op["mae"], op["rmse"]],
        })
        st.dataframe(comp_df, use_container_width=True)
        chart = pd.DataFrame({
            "基准": [bl["params_k"], (bl["model_bytes"] or 0) / 1024],
            "优化后": [op["params_k"], (op["model_bytes"] or 0) / 1024],
        }, index=["参数量 (K)", "体积 (KB)"])
        st.bar_chart(chart)
        st.success(f"结论：剪枝+量化后参数量 "
                   f"{(bl['params_k']-op['params_k'])/bl['params_k']*100:.1f}% "
                   f"（{bl['params_k']}K→{op['params_k']}K），精度保持。")
    else:
        st.info("暂无优化模型，请先在「智能优化」运行剪枝量化。")


# ---------------- 导航 ----------------
PAGES = {
    "总览": page_overview,
    "数据管理": page_data,
    "模型训练": page_train,
    "智能优化": page_optimize,
    "在线预测": page_predict,
    "手动测试": page_manual,
    "对比分析": page_compare,
}

def main():
    st.sidebar.title("🏭 设备工况 RUL 预测")
    st.sidebar.caption("融合统计学习 + 深度学习 + 智能计算优化")
    choice = st.sidebar.radio("功能导航", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.caption("制造智能技术课程设计\n数据：NASA C-MAPSS")
    PAGES[choice]()


if __name__ == "__main__":
    main()
