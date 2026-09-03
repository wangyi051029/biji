# -*- coding: utf-8 -*-
"""
离线单文件 RUL 预测测试站构建脚本
=================================
把训练好的 LSTM 模型权重 + 标准化参数内嵌进一个独立 HTML 文件，
任何电脑只需浏览器打开即可使用（无需安装 Python / 无需后端 / 无需联网）。

用法：
    python3 build_offline_site.py [subset] [model_name]
默认 subset=FD001 model_name=FD001_LSTM
输出：offline_rul_site.html
"""
import os
import sys
import json
import base64
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def main():
    subset = sys.argv[1] if len(sys.argv) > 1 else "FD001"
    model_name = sys.argv[2] if len(sys.argv) > 2 else f"{subset}_LSTM"

    # 1. 读取模型与元信息
    state = torch.load(os.path.join(MODEL_DIR, f"{model_name}.pt"),
                       map_location="cpu")
    with open(os.path.join(MODEL_DIR, f"{model_name}.pt.meta.json"),
              encoding="utf-8") as f:
        meta = json.load(f)
    arch = meta["arch"]
    input_size = arch["input_size"]
    hidden = arch["hidden_size"]
    num_layers = arch["num_layers"]
    window = meta.get("window", 30)
    feats = meta["features"]
    sensor_feats = [c for c in feats if c != "cycle"]

    # 2. 读取标准化参数
    with open(os.path.join(PROCESSED_DIR, subset, "scaler.json"),
              encoding="utf-8") as f:
        sc = json.load(f)
    mean = np.asarray(sc["mean"], dtype=np.float64)
    scale = np.asarray(sc["scale"], dtype=np.float64)
    sc_feats = list(sc["features"])
    if sc_feats != feats:  # 按模型特征顺序重排
        idx = [sc_feats.index(f) for f in feats]
        mean, scale = mean[idx], scale[idx]

    # 2.5 读取一行真实示例读数（原始数据中间行）
    import re
    raw_path = os.path.join(BASE_DIR, "data", "raw", f"train_{subset}.txt")
    sample_base = None
    if os.path.exists(raw_path):
        # 按空格分隔读取，跳过表头（若有）
        with open(raw_path, encoding="utf-8", errors="replace") as f:
            lines = [l.split() for l in f if l.strip()]
        if lines:
            mid = lines[len(lines) // 2]
            # 列顺序: unit cycle setting1-3 s1-s21
            s_index = {f"s{i}": 4 + i for i in range(1, 22)}
            sample_base = [float(mid[s_index[f]]) for f in sensor_feats]
            sample_base = [round(v, 2) for v in sample_base]

    # 3. 序列化权重为 base64
    def b64(name):
        arr = state[name].numpy().astype(np.float32).ravel()
        return base64.b64encode(arr.tobytes()).decode("ascii")

    weights = {k: b64(k) for k in state.keys()}
    shapes = {k: list(state[k].shape) for k in state.keys()}

    data = {
        "subset": subset, "model_name": model_name,
        "input_size": input_size, "hidden": hidden,
        "num_layers": num_layers, "window": window,
        "sensor_features": sensor_feats,
        "mean": mean.tolist(), "scale": scale.tolist(),
        "sample_base": sample_base,
        "weights": weights, "shapes": shapes,
    }
    data_json = json.dumps(data, ensure_ascii=False)

    # 4. 组装 HTML
    html = TEMPLATE.replace("__MODEL_DATA__", data_json)
    out = os.path.join(BASE_DIR, "offline_rul_site.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    size = os.path.getsize(out) / 1024 / 1024
    print(f"生成完成: {out} ({size:.2f} MB)")
    print(f"模型: {model_name} | 特征 {len(sensor_feats)} 个 | 窗口 {window} "
          f"| hidden {hidden} 层数 {num_layers}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>设备工况 RUL 预测测试站（离线单文件版）</title>
<style>
  :root{--blue:#002FA7;--bg:#F5F3EE;--ink:#111;--gray:#5B5A55;--line:#E0DDD5;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",sans-serif;
       background:var(--bg);color:var(--ink);line-height:1.6;}
  header{background:var(--blue);color:#fff;padding:20px 28px;}
  header h1{font-size:22px;letter-spacing:.5px;}
  header p{font-size:13px;opacity:.85;margin-top:4px;}
  .wrap{max-width:960px;margin:24px auto;padding:0 20px;}
  .card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:20px 24px;margin-bottom:20px;}
  .card h2{font-size:17px;border-left:4px solid var(--blue);padding-left:10px;margin-bottom:14px;}
  .desc{font-size:13px;color:var(--gray);margin-bottom:14px;}
  .row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px;}
  label{font-size:13px;color:var(--gray);}
  select,input[type=number]{border:1px solid #ccc;border-radius:6px;padding:7px 10px;font-size:14px;background:#fff;}
  textarea{width:100%;border:1px solid #ccc;border-radius:6px;padding:10px;font-size:13px;
           font-family:Consolas,Menlo,monospace;resize:vertical;min-height:110px;}
  button{border:none;border-radius:6px;padding:9px 18px;font-size:14px;cursor:pointer;}
  .primary{background:var(--blue);color:#fff;}
  .ghost{background:#eee;color:#333;border:1px solid #ccc;}
  .feature-hint{font-size:12px;color:#777;background:#f7f7f5;border:1px dashed #ccc;
                border-radius:6px;padding:8px 12px;margin:10px 0;}
  #result{white-space:pre-wrap;font-size:15px;background:#f4f7ff;border:1px solid #cfdcff;
          border-radius:6px;padding:14px;display:none;}
  .big{font-size:30px;font-weight:bold;color:var(--blue);}
  footer{margin:30px auto 40px;text-align:center;font-size:12px;color:#999;max-width:960px;padding:0 20px;}
  .note{font-size:12px;color:#888;margin-top:8px;}
</style>
</head>
<body>
<header>
  <h1>设备工况 RUL 预测测试站</h1>
  <p>离线单文件版 · 内嵌 FD001 预训练 LSTM 模型 · 任何电脑双击本文件即可使用（无需安装、无需联网）</p>
</header>
<div class="wrap">
  <div class="card">
    <h2>模型信息</h2>
    <div class="desc" id="model-info">加载中…</div>
  </div>
  <div class="card">
    <h2>输入传感器数据</h2>
    <p class="desc">粘贴若干周期的传感器读数：<b>每行一个周期，用英文逗号分隔</b>，顺序与下方特征一致。
      可只填 1 行（系统自动补齐为完整窗口），建议填 30 行以上模拟历史序列。</p>
    <div class="row">
      <button class="ghost" onclick="fillSample()">填入示例数据</button>
      <button class="ghost" onclick="clearInput()">清空</button>
      <button class="primary" onclick="predict()">开始预测</button>
    </div>
    <textarea id="input" placeholder="例如：&#10;643.20,1597.44,1416.68,21.61,551.88,2380.21,9046.19,519.04,240.68,5215.57,2388.06,9048.07,518.48,2453.80,5191.04&#10;…（每行一个周期）"></textarea>
    <div class="feature-hint" id="feat-hint"></div>
    <div id="result"></div>
  </div>
  <div class="card">
    <h2>说明</h2>
    <ul class="desc" style="padding-left:18px;">
      <li>本页面内置课程设计 FD001 子集训练的双层 LSTM 模型（窗口 30），预测剩余使用寿命 RUL（周期数）。</li>
      <li>全部计算在浏览器本地完成，数据不上传，离线可用。</li>
      <li>传感器读数单位与 C-MAPSS 原始数据一致；预测结果仅供课程设计演示参考。</li>
    </ul>
  </div>
</div>
<footer>制造智能技术课程设计 · 面向小样本工业工况预测：融合统计学习的深度学习智能计算优化方案 · 离线单文件测试站</footer>

<script>
"use strict";
// ---------------- 内嵌模型数据 ----------------
const MD = __MODEL_DATA__;

// 解码 base64 -> Float32Array
function b64ToF32(b64){
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for(let i=0;i<bin.length;i++) view[i]=bin.charCodeAt(i);
  return new Float32Array(buf);
}
// 按 shape 还原权重：1D 返回一维数组，2D 返回二维矩阵（row-major）
function mat(name){
  const sh = MD.shapes[name];
  const f = b64ToF32(MD.weights[name]);
  if(sh.length===1){
    return Array.from(f); // 一维数组（bias 等）
  }
  const rows=sh[0], cols=sh[1];
  const m=[];
  for(let r=0;r<rows;r++){ const row=[]; for(let c=0;c<cols;c++) row.push(f[r*cols+c]); m.push(row); }
  return m;
}
const W = {};
Object.keys(MD.weights).forEach(k=>{ W[k]=mat(k); });

// 矩阵乘 / 加 bias
function matMul(A,B){ // A: rA x k, B: k x cB
  const rA=A.length, k=A[0].length, cB=B[0].length;
  const C=Array.from({length:rA},()=>new Array(cB).fill(0));
  for(let i=0;i<rA;i++)for(let j=0;j<cB;j++){let s=0;for(let t=0;t<k;t++)s+=A[i][t]*B[t][j];C[i][j]=s;}
  return C;
}
function addVec(M,v){ const r=M.length,c=M[0].length; for(let i=0;i<r;i++)for(let j=0;j<c;j++)M[i][j]+=v[j]; return M; }
function sig(x){ return 1/(1+Math.exp(-x)); }
function tanh(x){ return Math.tanh(x); }

// 单层 LSTM 前向：返回 (h_seq, last_h)
function lstmLayer(X, wih, whh, bih, bhh, hid){
  // gates 顺序: i,f,g,o
  const T=X.length, inSz=X[0].length;
  let h=new Array(hid).fill(0), c=new Array(hid).fill(0);
  const H=[];
  for(let t=0;t<T;t++){
    const g=new Array(4*hid).fill(0);
    // Wih*x + bih
    for(let r=0;r<4*hid;r++){ let s=bih[r]; for(let k=0;k<inSz;k++) s+=wih[r][k]*X[t][k]; g[r]=s; }
    // Whh*h + bhh
    for(let r=0;r<4*hid;r++){ let s=bhh[r]; for(let k=0;k<hid;k++) s+=whh[r][k]*h[k]; g[r]+=s; }
    const slice=k=>g.slice(k*hid,(k+1)*hid);
    const ig=slice(0).map(sig), fg=slice(1).map(sig), gg=slice(2).map(tanh), og=slice(3).map(sig);
    const cNew=[], hNew=[];
    for(let j=0;j<hid;j++){
      cNew[j]=fg[j]*c[j]+ig[j]*gg[j];
      hNew[j]=og[j]*tanh(cNew[j]);
    }
    c=cNew; h=hNew; H.push(h.slice());
  }
  return {H:H, last:h};
}

// 模型前向：窗口 x 为 (window x inputSize) 原始标准化特征
function forward(X){
  const L=MD.num_layers, hid=MD.hidden;
  let inp=X;
  let lastH=null;
  for(let l=0;l<L;l++){
    const res=lstmLayer(inp, W[`lstm.weight_ih_l${l}`], W[`lstm.weight_hh_l${l}`],
                          W[`lstm.bias_ih_l${l}`], W[`lstm.bias_hh_l${l}`], hid);
    inp=res.H; lastH=res.last;
  }
  // head: Linear(hid->32) ReLU Linear(32->1)
  let z=new Array(W['head.0.weight'].length).fill(0);
  const w0=W['head.0.weight'], b0=W['head.0.bias'];
  for(let i=0;i<w0.length;i++){ let s=b0[i]; for(let j=0;j<hid;j++) s+=w0[i][j]*lastH[j]; z[i]=Math.max(0,s); }
  let out=W['head.3.bias'][0];
  const w1=W['head.3.weight'][0];
  for(let j=0;j<32;j++) out+=w1[j]*z[j];
  return Math.max(0,out); // clip(0,None)
}

// ---------------- 页面逻辑 ----------------
const feats=MD.sensor_features;
function init(){
  document.getElementById('model-info').textContent =
    `子集 ${MD.subset} · 模型 ${MD.model_name} · 特征 ${feats.length} 个 · 窗口 ${MD.window} 周期 · 双层 LSTM(hidden=${MD.hidden})`;
  document.getElementById('feat-hint').innerHTML =
    `<b>特征顺序（每行 ${feats.length} 个值，逗号分隔）：</b><br>${feats.join(' , ')}`;
}
function parseRows(){
  const raw=document.getElementById('input').value.trim();
  if(!raw) throw new Error('请先输入传感器读数');
  const rows=raw.split(/\n+/).map(l=>l.trim()).filter(Boolean)
    .map(l=>l.split(/[,，\s]+/).map(Number));
  if(rows.length===0) throw new Error('请输入至少一行数据');
  const need=feats.length;
  if(rows.some(r=>r.length!==need)) throw new Error(`每行需要 ${need} 个数值，请检查格式`);
  if(rows.some(r=>r.some(v=>isNaN(v)))) throw new Error('输入包含非数字内容');
  return rows;
}
function standardize(rows){
  // 构造 cycle 列 + 传感器列，顺序与 feats 一致（cycle + sensor_features）
  const cols=rows[0].length;
  const X=[];
  for(let t=0;t<rows.length;t++){
    const row=[t+1].concat(rows[t]); // [cycle, s...]
    const std=[];
    for(let j=0;j<row.length;j++) std.push((row[j]-MD.mean[j])/MD.scale[j]);
    X.push(std);
  }
  return X;
}
function buildWindow(X){
  const win=MD.window;
  if(X.length>=win) return X.slice(X.length-win);
  const pad=[];
  while(pad.length<win-X.length) pad.push(X[0]);
  return pad.concat(X);
}
function predict(){
  const res=document.getElementById('result');
  try{
    const rows=parseRows();
    const X=buildWindow(standardize(rows));
    const rul=forward(X);
    res.style.display='block';
    res.innerHTML=`<div>预测结果：剩余使用寿命 RUL ≈ <span class="big">${rul.toFixed(2)}</span> 个周期</div>
      <div class="note">输入 ${rows.length} 个周期 · 取最近 ${MD.window} 周期构成窗口 · 全程浏览器本地计算</div>`;
  }catch(e){
    res.style.display='block';
    res.innerHTML=`<span style="color:#c0392b">${e.message}</span>`;
  }
}
function fillSample(){
  // 从原始数据取的真实一行读数，加入轻度退化漂移生成 30 行示例序列
  const base=MD.sample_base;
  if(!base){ document.getElementById('input').value=''; return; }
  const lines=[];
  for(let t=0;t<30;t++){
    const drift=Math.min(t*0.3,8); // 轻度退化漂移
    lines.push(base.map(v=>(v*(1+drift/1000)+(Math.random()*2-1)*2)).map(v=>v.toFixed(2)).join(','));
  }
  document.getElementById('input').value=lines.join('\n');
}
function clearInput(){ document.getElementById('input').value=''; document.getElementById('result').style.display='none'; }
init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
