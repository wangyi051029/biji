/* 设备工况智能预测系统 前端逻辑 */
const API = '';
const $ = (id) => document.getElementById(id);
const charts = {};

function initCharts() {
  charts.features = echarts.init($('chart-features'));
  charts.model = echarts.init($('chart-model'));
  charts.optCompare = echarts.init($('chart-opt-compare'));
  // 隐藏面板中的图表初始化为 0 尺寸，切换到可见面板后再 resize
  window.addEventListener('resize', () => {
    Object.values(charts).forEach(c => c && c.resize());
  });
}

window.addEventListener('error', function (e) {
  setStatus('JS错误: ' + (e.message || 'unknown'), 'err');
});

/* ---------- 通用 ---------- */
async function api(url, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API + url, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function setStatus(text, cls = '') {
  const el = $('api-status');
  el.innerHTML = `<span class="dot ${cls}"></span>${text}`;
}

function showResult(id, data, ok = true) {
  const el = $(id);
  if (typeof data === 'string') { el.textContent = data; }
  else { el.textContent = JSON.stringify(data, null, 2); }
  el.className = 'result ' + (ok ? 'ok' : 'err');
}

/* ---------- 标签页 ---------- */
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    $('tab-' + t.dataset.tab).classList.add('active');
    if (t.dataset.tab === 'overview') loadOverview();
    if (t.dataset.tab === 'data') loadDataPage();
    if (t.dataset.tab === 'train') loadModels();
    if (t.dataset.tab === 'optimize') refreshOptModelSelect();
    if (t.dataset.tab === 'predict') refreshPredModelSelect();
    if (t.dataset.tab === 'compare') loadCompare();
    setTimeout(() => { Object.values(charts).forEach(c => c && c.resize()); }, 50);
  });
});

/* ---------- 总览 ---------- */
async function loadOverview() {
  try {
    const [datasets, models] = await Promise.all([api('/api/datasets'), api('/api/model/list')]);
    // 卡片
    const nFeature = datasets.reduce((s, d) => s + (d.n_features || 0), 0);
    const nModel = models.length;
    const best = models.filter(m => m.mae).sort((a, b) => a.mae - b.mae)[0];
    const avgInfer = models.length ? (models.reduce((s, m) => s + (m.inference_ms || 0), 0) / models.length).toFixed(3) : '—';
    $('overview-cards').innerHTML = `
      <div class="card"><div class="label">已入库数据集</div><div class="num">${datasets.length}</div><div class="label">FD001~FD004 工业时序数据</div></div>
      <div class="card"><div class="label">筛选特征总数</div><div class="num">${nFeature}</div><div class="label">方差+相关+ANOVA 筛选</div></div>
      <div class="card"><div class="label">已训练模型</div><div class="num">${nModel}</div><div class="label">最优 MAE：${best ? best.mae : '—'}</div></div>
      <div class="card"><div class="label">平均推理耗时</div><div class="num">${avgInfer}ms</div><div class="label">普通 PC CPU 推理</div></div>`;

    // 特征筛选图
    const featNames = datasets.map(d => d.name);
    const featNums = datasets.map(d => d.n_features || 0);
    charts.features.setOption({
      tooltip: {}, legend: { data: ['筛选后特征数'] },
      xAxis: { type: 'category', data: featNames },
      yAxis: { type: 'value', name: '特征数' },
      series: [{ name: '筛选后特征数', type: 'bar', barWidth: '45%',
        itemStyle: { color: '#24527e', borderRadius: [4,4,0,0] },
        label: { show: true, position: 'top' }, data: featNums }]
    }, true);

    // 模型对比图（推理耗时）
    const mNames = models.map(m => m.name);
    const mTime = models.map(m => m.inference_ms || 0);
    charts.model.setOption({
      tooltip: {}, legend: { data: ['推理耗时(ms)'] },
      xAxis: { type: 'category', data: mNames, axisLabel: { rotate: 30 } },
      yAxis: { type: 'value', name: 'ms' },
      series: [{ name: '推理耗时(ms)', type: 'bar', barWidth: '50%',
        itemStyle: { color: '#e67e22', borderRadius: [4,4,0,0] },
        label: { show: true, position: 'top' }, data: mTime }]
    }, true);
  } catch (e) { setStatus('总览加载失败', 'err'); }
}

/* ---------- 数据管理 ---------- */
async function preprocess() {
  const subset = $('data-subset').value;
  showResult('data-result', '预处理中...');
  try {
    const r = await api('/api/preprocess', 'POST', { subset });
    showResult('data-result', r, true);
    await Promise.all([loadDatasetTable(), loadFeatureDetail(subset)]);
    setStatus('预处理完成', 'ok');
  } catch (e) { showResult('data-result', e.message, false); }
}
async function preprocessAll() {
  showResult('data-result', '预处理全部子集中...');
  try {
    const r = await api('/api/preprocess/all', 'POST', {});
    showResult('data-result', '全部完成：' + r.length + ' 个子集', true);
    await loadDatasetTable();
  } catch (e) { showResult('data-result', e.message, false); }
}
async function loadDatasetTable() {
  const ds = await api('/api/datasets');
  $('dataset-table').querySelector('tbody').innerHTML = ds.map(d => `
    <tr><td>${d.name}</td><td>${d.n_train}</td><td>${d.n_test}</td>
    <td>${d.n_features}</td><td>${d.noise_rate != null ? (d.noise_rate * 100).toFixed(1) + '%' : '—'}</td></tr>`).join('');
}
async function loadFeatureDetail(subset) {
  try {
    const f = await api('/api/features/' + subset);
    showResult('feature-detail', {
      原始传感器: f.raw_sensors,
      方差剔除: f.removed_by_variance,
      相关剔除: f.removed_by_correlation,
      ANOVA剔除: f.removed_by_anova,
      保留特征: f.selected_features,
    });
  } catch (e) { showResult('feature-detail', e.message, false); }
}
async function loadDataPage() {
  await loadDatasetTable();
  loadFeatureDetail($('data-subset').value);
}

/* ---------- 模型训练 ---------- */
async function trainModel() {
  const body = {
    subset: $('train-subset').value,
    n_samples: +$('train-samples').value,
    window: +$('train-window').value,
    epochs: +$('train-epochs').value,
    hidden: +$('train-hidden').value,
    lr: +$('train-lr').value,
  };
  showResult('train-result', '模型训练中，请稍候...（普通 PC CPU 训练）');
  $('train-result').className = 'result';
  try {
    const r = await api('/api/model/train', 'POST', body);
    showResult('train-result', r, true);
    await loadModels();
    refreshOptModelSelect(); refreshPredModelSelect();
    setStatus('模型训练完成', 'ok');
  } catch (e) { showResult('train-result', e.message, false); }
}
async function loadModels() {
  const models = await api('/api/model/list');
  $('model-table').querySelector('tbody').innerHTML = models.map(m => `
    <tr><td>${m.id}</td><td>${m.name}</td><td>${m.algo_type}</td><td>${m.dataset}</td>
    <td>${m.params_k}</td><td>${m.inference_ms}</td>
    <td>${m.mae ?? '—'}</td><td>${m.rmse ?? '—'}</td></tr>`).join('');
}

/* ---------- 智能优化 ---------- */
async function refreshOptModelSelect() {
  const models = await api('/api/model/list');
  const base = models.filter(m => m.algo_type === 'baseline');
  $('opt-model').innerHTML = base.map(m => `<option value="${m.name}">${m.name} (${m.dataset})</option>`).join('');
}
async function optimize() {
  const body = {
    subset: $('opt-subset').value,
    model_name: $('opt-model').value,
    prune_ratio: +$('opt-prune').value,
    quant_bits: +$('opt-bits').value,
  };
  showResult('opt-result', '优化执行中...');
  try {
    const r = await api('/api/optimize', 'POST', body);
    showResult('opt-result', r, true);
    drawOptCompare(r.compare);
    refreshPredModelSelect();
    setStatus('智能优化完成', 'ok');
  } catch (e) { showResult('opt-result', e.message, false); }
}
function drawOptCompare(c) {
  charts.optCompare.setOption({
    tooltip: {}, legend: { data: ['基准', '优化后'] },
    radar: { indicator: [
      { name: '参数量', max: Math.max(c.base.params_k, c.optimized.params_k) * 1.2 },
      { name: '模型大小', max: Math.max(c.base.model_bytes, c.optimized.model_bytes) * 1.2 },
      { name: '推理耗时', max: Math.max(c.base.inference_ms, c.optimized.inference_ms) * 1.2 },
      { name: '精度(反向MAE)', max: 100 },
    ]},
    series: [{
      type: 'radar', data: [
        { name: '基准', value: [c.base.params_k, c.base.model_bytes, c.base.inference_ms,
            c.base.metric ? 100 - Math.min(c.base.metric.mae, 100) : 50],
          areaStyle: { opacity: .2 } },
        { name: '优化后', value: [c.optimized.params_k, c.optimized.model_bytes,
            c.optimized.inference_ms, c.optimized.metric ? 100 - Math.min(c.optimized.metric.mae, 100) : 50],
          areaStyle: { opacity: .2 } },
      ]
    }]
  }, true);
}

/* ---------- 在线预测 ---------- */
async function refreshPredModelSelect() {
  const models = await api('/api/model/list');
  $('pred-model').innerHTML = models.map(m => `<option value="${m.name}">${m.name} (${m.dataset})</option>`).join('');
}
async function predictRul() {
  const subset = $('pred-subset').value;
  const model = $('pred-model').value;
  const unit = +$('pred-unit').value;
  try {
    const r = await api('/api/model/predict', 'POST', {
      subset, model_name: model, unit, n_cycles: 30
    });
    const txt = `预测结果：该单元剩余使用寿命 RUL ≈ ${r.pred_rul} 个周期\n（模型：${r.model_name} · 最后观测周期 ${r.last_cycle}` +
      (r.true_rul != null ? ` · 真值 ${r.true_rul}` : '') + '）';
    showResult('pred-result', txt, true);
    await loadPredTable();
  } catch (e) { showResult('pred-result', e.message, false); }
}
async function loadPredTable() {
  const rows = await api('/api/predictions');
  $('pred-table').querySelector('tbody').innerHTML = rows.map(p => `
    <tr><td>${p.dataset}</td><td>${p.model_name}</td><td>${p.unit}</td>
    <td>${p.last_cycle}</td><td>${p.true_rul}</td><td>${p.pred_rul}</td>
    <td>${p.abs_error}</td></tr>`).join('');
}

/* ---------- 对比分析 ---------- */
async function loadCompare() {
  const subset = $('cmp-subset').value;
  try {
    const r = await api('/api/compare/' + subset);
    const base = r.baseline[0], opt = r.optimized[0];
    if (!base) { showResult('compare-result', '该子集暂无基准模型，请先训练', false); return; }
    const lines = [];
    lines.push(`=== ${subset} 基准模型 vs 优化模型 ===`);
    lines.push(`基准模型: ${base.name} | 参数量 ${base.params_k}K | 推理 ${base.inference_ms}ms | MAE ${base.mae}`);
    if (opt) {
      lines.push(`优化模型: ${opt.name} | 参数量 ${opt.params_k}K | 推理 ${opt.inference_ms}ms | MAE ${opt.mae}`);
      lines.push(`参数量减少: ${((1 - opt.params_k / base.params_k) * 100).toFixed(1)}%`);
      lines.push(`模型大小减少: ${((1 - opt.model_bytes / base.model_bytes) * 100).toFixed(1)}%`);
      lines.push(`推理耗时: ${base.inference_ms}ms → ${opt.inference_ms}ms`);
    } else {
      lines.push('尚未执行优化，请到"智能优化"页面执行剪枝+量化。');
    }
    showResult('compare-result', lines.join('\n'), true);
  } catch (e) { showResult('compare-result', e.message, false); }
}

/* ---------- 启动 ---------- */
(async function init() {
  initCharts();
  try {
    const h = await api('/api/health');
    setStatus(h.service || '服务正常', 'ok');
  } catch (e) { setStatus('后端未连接', 'err'); }
  loadOverview();
})();
