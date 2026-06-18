"""Local desktop application for regression-only bochan workflows."""

from __future__ import annotations

import threading
import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from .services import (
    DatasetStore,
    build_dataset_record,
    dataframe_preview,
    load_dataframe_from_payload,
    run_regression_workflow,
)


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetLoadRequest(_Schema):
    source_type: Literal["csv", "excel", "sqlite", "duckdb"] = "csv"
    name: str | None = None
    content_base64: str | None = None
    encoding: str = "utf-8-sig"
    sep: str | None = None
    sheet_name: str | int | None = 0
    sql: str | None = None
    database_path: str | None = None


class SearchVariableSchema(_Schema):
    name: str
    type: Literal["auto", "numeric", "categorical"] = "auto"
    lower: float | None = None
    upper: float | None = None
    step: float | None = None
    fixed: bool = False
    fixed_value: Any | None = None
    categories: list[Any] | None = None


class LinearConstraintTermSchema(_Schema):
    column: str
    coefficient: float = 1.0


class LinearConstraintSchema(_Schema):
    name: str = "constraint"
    terms: list[LinearConstraintTermSchema]
    sense: Literal["le", "ge", "eq"] = "le"
    rhs: float
    enabled: bool = True


class KSparseSchema(_Schema):
    enabled: bool = False
    columns: list[str] = Field(default_factory=list)
    k: int = 0
    score: Literal["abs", "value"] = "abs"
    support_selection: Literal["topk", "sample"] = "topk"
    final_priority: Literal["grid", "constraints"] = "grid"


class AcquisitionSettingsSchema(_Schema):
    name: str = "EI"
    beta: float = 2.0
    acqf_kwargs: dict[str, Any] = Field(default_factory=dict)


class OptimizerSettingsSchema(_Schema):
    name: str = "optimize_acqf"
    q: int = 1
    num_restarts: int = 10
    raw_samples: int = 256
    sequential: bool = True


class RegressionRunRequest(_Schema):
    dataset_id: str
    feature_columns: list[str]
    target_column: str
    direction: Literal["maximize", "minimize"] = "maximize"
    model_type: str = "base"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    fit_maxiter: int = 128
    normalize: bool = True
    outcome_transform: bool = True
    input_perturbation: bool = False
    n_w: int = 16
    perturbation_std: float = 0.1
    search_space: list[SearchVariableSchema] = Field(default_factory=list)
    constraints: list[LinearConstraintSchema] = Field(default_factory=list)
    k_sparse: KSparseSchema | None = None
    acquisition: AcquisitionSettingsSchema = Field(default_factory=AcquisitionSettingsSchema)
    optimizer: OptimizerSettingsSchema = Field(default_factory=OptimizerSettingsSchema)
    drop_missing: bool = True


def create_app(*, title: str = "bochan Desktop Regression", version: str = "0.1.0") -> FastAPI:
    """Create the local FastAPI app used by the desktop shell."""

    app = FastAPI(title=title, version=version)
    store = DatasetStore()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "task_type": "regression"}

    @app.get("/api/datasets")
    def list_datasets() -> dict[str, Any]:
        return {"datasets": store.list()}

    @app.post("/api/datasets/load")
    def load_dataset(request: DatasetLoadRequest) -> dict[str, Any]:
        try:
            data, metadata = load_dataframe_from_payload(
                source_type=request.source_type,
                content_base64=request.content_base64,
                name=request.name,
                encoding=request.encoding,
                sep=request.sep,
                sheet_name=request.sheet_name,
                sql=request.sql,
                database_path=request.database_path,
            )
            record = build_dataset_record(
                data=data,
                name=request.name or "dataset",
                source_type=request.source_type,
                metadata=metadata,
            )
            store.add(record)
            return {
                "dataset_id": record.dataset_id,
                "name": record.name,
                "source_type": record.source_type,
                "profile": record.profile,
                "preview": dataframe_preview(record.data, limit=50),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/datasets/{dataset_id}/preview")
    def get_preview(dataset_id: str, limit: int = 100) -> dict[str, Any]:
        try:
            record = store.get(dataset_id)
            return {
                "dataset_id": dataset_id,
                "profile": record.profile,
                "preview": dataframe_preview(record.data, limit=limit),
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/regression/run")
    def run_regression(request: RegressionRunRequest) -> dict[str, Any]:
        try:
            return run_regression_workflow(request, store)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def run_server(*, host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    """Run the local desktop API with uvicorn."""

    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, reload=reload)


def run_desktop(*, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Launch the local API and open it in a pywebview desktop window."""

    try:
        import webview
    except ImportError as exc:
        raise RuntimeError("Install desktop dependencies first: pip install -e '.[desktop]'") from exc

    import uvicorn

    app = create_app()

    def _serve() -> None:
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    time.sleep(0.8)
    webview.create_window("bochan Regression Desktop", f"http://{host}:{port}", width=1280, height=860)
    webview.start()


app = create_app()


_INDEX_HTML = r"""
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>bochan Regression Desktop</title>
  <style>
    :root { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; }
    body { margin: 0; background: #f4f6f8; }
    header { background: #172033; color: white; padding: 16px 24px; }
    header h1 { margin: 0; font-size: 20px; }
    main { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 58px); }
    nav { background: #ffffff; border-right: 1px solid #d9dee7; padding: 16px; }
    nav button { display: block; width: 100%; margin: 0 0 8px; padding: 10px 12px; text-align: left; border: 1px solid #d9dee7; border-radius: 8px; background: #fff; cursor: pointer; }
    nav button.active { background: #e9f0ff; border-color: #8eafff; }
    section { display: none; padding: 20px 24px 48px; }
    section.active { display: block; }
    .card { background: white; border: 1px solid #d9dee7; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.03); }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; }
    label { display: block; font-size: 12px; font-weight: 700; margin: 0 0 4px; color: #4c5870; }
    input, select, textarea { box-sizing: border-box; width: 100%; border: 1px solid #cdd4df; border-radius: 8px; padding: 8px 10px; background: #fff; color: #172033; }
    textarea { min-height: 110px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
    .primary { background: #2454ff; color: white; border: 0; border-radius: 8px; padding: 10px 14px; cursor: pointer; font-weight: 700; }
    .secondary { background: white; color: #172033; border: 1px solid #cdd4df; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
    table { width: 100%; border-collapse: collapse; background: white; }
    th, td { border-bottom: 1px solid #e8ecf2; padding: 8px; text-align: left; font-size: 13px; vertical-align: top; }
    th { background: #f8fafc; font-size: 12px; color: #4c5870; }
    .scroll { overflow: auto; max-height: 420px; border: 1px solid #e8ecf2; border-radius: 8px; }
    .muted { color: #667085; font-size: 13px; }
    .ok { color: #087443; font-weight: 700; }
    .bad { color: #b42318; font-weight: 700; }
    pre { white-space: pre-wrap; overflow: auto; background: #0b1020; color: #d8e3ff; padding: 12px; border-radius: 8px; }
    canvas { width: 100%; max-width: 720px; height: 320px; background: #fff; border: 1px solid #d9dee7; border-radius: 8px; }
  </style>
</head>
<body>
<header><h1>bochan Regression Desktop MVP</h1></header>
<main>
  <nav>
    <button class="active" data-page="data">1. データ</button>
    <button data-page="variables">2. 変数選択</button>
    <button data-page="settings">3. 探索・モデル設定</button>
    <button data-page="run">4. 実行</button>
    <button data-page="results">5. 結果</button>
    <p class="muted">regression 単一目的のみ対応。CSV / Excel / SQLite / DuckDB の入口を用意しています。</p>
  </nav>

  <div>
    <section id="data" class="active">
      <div class="card">
        <h2>データ読込</h2>
        <div class="grid">
          <div><label>source_type</label><select id="sourceType"><option value="csv">CSV</option><option value="excel">Excel</option><option value="sqlite">SQLite</option><option value="duckdb">DuckDB</option></select></div>
          <div><label>CSV encoding</label><input id="encoding" value="utf-8-sig" /></div>
          <div><label>CSV sep（空欄で自動推定）</label><input id="sep" placeholder="," /></div>
          <div><label>Excel sheet</label><input id="sheetName" value="0" /></div>
        </div>
        <p><input id="fileInput" type="file" /></p>
        <div class="grid">
          <div><label>DB path</label><input id="databasePath" placeholder="C:/path/to/data.sqlite" /></div>
          <div style="grid-column: span 3"><label>SQL</label><input id="sql" placeholder="select * from table_name" /></div>
        </div>
        <p><button class="primary" onclick="loadDataset()">読込</button></p>
        <pre id="dataStatus">未読込</pre>
      </div>
      <div class="card">
        <h2>プレビュー</h2>
        <div id="preview" class="scroll"></div>
      </div>
    </section>

    <section id="variables">
      <div class="card">
        <h2>説明変数・目的変数</h2>
        <p class="muted">目的変数は numeric の regression として扱います。カテゴリ説明変数は整数エンコードし、mixed model 用の cat_dims に渡します。</p>
        <div id="columnsTable" class="scroll"></div>
      </div>
    </section>

    <section id="settings">
      <div class="card">
        <h2>探索範囲・step・固定値</h2>
        <p class="muted">数値列は min/max/step/fixed を指定できます。カテゴリ列は fixed_value にカテゴリ名を入れると固定されます。</p>
        <div id="searchSpaceTable" class="scroll"></div>
      </div>
      <div class="card">
        <h2>制約・k-sparse</h2>
        <div class="grid">
          <div style="grid-column: span 2"><label>線形制約 JSON</label><textarea id="constraintsJson">[]</textarea></div>
          <div style="grid-column: span 2"><label>k-sparse JSON</label><textarea id="kSparseJson">{"enabled": false, "columns": [], "k": 0, "score": "abs", "support_selection": "topk", "final_priority": "grid"}</textarea></div>
        </div>
      </div>
      <div class="card">
        <h2>モデル・獲得関数</h2>
        <div class="grid">
          <div><label>model_type</label><select id="modelType"><option>base</option><option>saas</option><option>pca</option><option>rembo</option><option>hetero</option></select></div>
          <div><label>direction</label><select id="direction"><option value="maximize">maximize</option><option value="minimize">minimize</option></select></div>
          <div><label>fit_maxiter</label><input id="fitMaxiter" type="number" value="128" /></div>
          <div><label>acquisition</label><select id="acqName"><option>EI</option><option>LogEI</option><option>PI</option><option>UCB</option><option>NEI</option><option>PosteriorVariance</option><option>NIPV</option><option>Straddle</option></select></div>
          <div><label>beta（UCB用）</label><input id="beta" type="number" value="2.0" step="0.1" /></div>
          <div><label>q</label><input id="q" type="number" value="1" /></div>
          <div><label>num_restarts</label><input id="numRestarts" type="number" value="10" /></div>
          <div><label>raw_samples</label><input id="rawSamples" type="number" value="256" /></div>
        </div>
      </div>
    </section>

    <section id="run">
      <div class="card">
        <h2>実行</h2>
        <p><button class="primary" onclick="runRegression()">regression候補点を生成</button></p>
        <pre id="runStatus">未実行</pre>
      </div>
    </section>

    <section id="results">
      <div class="card">
        <h2>候補点</h2>
        <p><button class="secondary" onclick="downloadCandidates()">候補点JSONを保存</button></p>
        <div id="candidateTable" class="scroll"></div>
      </div>
      <div class="card">
        <h2>グラフ</h2>
        <p class="muted">候補rankごとの predicted_target_mean を表示します。</p>
        <canvas id="resultChart" width="720" height="320"></canvas>
      </div>
    </section>
  </div>
</main>

<script>
let currentDataset = null;
let currentResult = null;
let selectedFeatures = new Set();
let selectedTarget = null;
let profileColumns = [];

for (const button of document.querySelectorAll('nav button')) {
  button.addEventListener('click', () => {
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
    button.classList.add('active');
    document.getElementById(button.dataset.page).classList.add('active');
  });
}

function setStatus(id, value) { document.getElementById(id).textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2); }
function parseJson(id) {
  const text = document.getElementById(id).value.trim();
  return text ? JSON.parse(text) : null;
}

async function readFileAsDataUrl(file) {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function loadDataset() {
  try {
    const sourceType = document.getElementById('sourceType').value;
    const file = document.getElementById('fileInput').files[0];
    let contentBase64 = null;
    if ((sourceType === 'csv' || sourceType === 'excel') && file) {
      contentBase64 = await readFileAsDataUrl(file);
    }
    const sheetRaw = document.getElementById('sheetName').value;
    const payload = {
      source_type: sourceType,
      name: file ? file.name : 'dataset',
      content_base64: contentBase64,
      encoding: document.getElementById('encoding').value,
      sep: document.getElementById('sep').value || null,
      sheet_name: /^\d+$/.test(sheetRaw) ? Number(sheetRaw) : sheetRaw,
      database_path: document.getElementById('databasePath').value || null,
      sql: document.getElementById('sql').value || null
    };
    const res = await fetch('/api/datasets/load', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || 'load failed');
    currentDataset = json;
    profileColumns = json.profile.columns;
    const numericColumns = profileColumns.filter(c => c.kind === 'numeric');
    selectedTarget = numericColumns[0]?.name || null;
    selectedFeatures = new Set(numericColumns.filter(c => c.name !== selectedTarget).slice(0, 2).map(c => c.name));
    setStatus('dataStatus', {dataset_id: json.dataset_id, profile: json.profile});
    renderPreview(json.preview);
    renderColumns();
    renderSearchSpace();
  } catch (err) {
    setStatus('dataStatus', 'ERROR: ' + err.message);
  }
}

function renderPreview(rows) {
  if (!rows || rows.length === 0) { document.getElementById('preview').innerHTML = '<p class="muted">No rows</p>'; return; }
  const cols = Object.keys(rows[0]);
  document.getElementById('preview').innerHTML = '<table><thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => '<tr>' + cols.map(c => `<td>${r[c] ?? ''}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
}

function renderColumns() {
  const html = '<table><thead><tr><th>feature</th><th>target</th><th>column</th><th>kind</th><th>dtype</th><th>missing</th><th>unique</th></tr></thead><tbody>' +
    profileColumns.map(c => {
      const nameArg = JSON.stringify(c.name);
      const isTarget = selectedTarget === c.name;
      return `<tr>
        <td><input type="checkbox" ${selectedFeatures.has(c.name) ? 'checked' : ''} ${isTarget ? 'disabled' : ''} onchange='toggleFeature(${nameArg}, this.checked)' /></td>
        <td><input type="radio" name="target" ${isTarget ? 'checked' : ''} onchange='setTarget(${nameArg})' /></td>
        <td>${c.name}</td><td>${c.kind}</td><td>${c.dtype}</td><td>${c.missing_count}</td><td>${c.unique_count}</td>
      </tr>`;
    }).join('') + '</tbody></table>';
  document.getElementById('columnsTable').innerHTML = html;
}

function toggleFeature(name, checked) {
  if (name === selectedTarget) return;
  checked ? selectedFeatures.add(name) : selectedFeatures.delete(name);
  renderSearchSpace();
}
function setTarget(name) {
  selectedTarget = name;
  selectedFeatures.delete(name);
  renderColumns();
  renderSearchSpace();
}

function renderSearchSpace() {
  const cols = profileColumns.filter(c => selectedFeatures.has(c.name) && c.name !== selectedTarget);
  const html = '<table><thead><tr><th>column</th><th>type</th><th>lower</th><th>upper</th><th>step</th><th>fixed</th><th>fixed_value</th></tr></thead><tbody>' +
    cols.map(c => `<tr data-var="${c.name}">
      <td>${c.name}</td>
      <td><select class="varType"><option value="auto">auto</option><option value="numeric" ${c.kind === 'numeric' ? 'selected' : ''}>numeric</option><option value="categorical" ${c.kind !== 'numeric' ? 'selected' : ''}>categorical</option></select></td>
      <td><input class="lower" type="number" value="${c.min ?? ''}" /></td>
      <td><input class="upper" type="number" value="${c.max ?? ''}" /></td>
      <td><input class="step" type="number" step="any" /></td>
      <td><input class="fixed" type="checkbox" /></td>
      <td><input class="fixedValue" /></td>
    </tr>`).join('') + '</tbody></table>';
  document.getElementById('searchSpaceTable').innerHTML = html;
}

function collectSearchSpace() {
  return [...document.querySelectorAll('#searchSpaceTable tr[data-var]')].map(row => {
    const val = cls => row.querySelector(cls).value;
    const numOrNull = x => x === '' ? null : Number(x);
    return {
      name: row.dataset.var,
      type: val('.varType'),
      lower: numOrNull(val('.lower')),
      upper: numOrNull(val('.upper')),
      step: numOrNull(val('.step')),
      fixed: row.querySelector('.fixed').checked,
      fixed_value: val('.fixedValue') === '' ? null : val('.fixedValue')
    };
  });
}

async function runRegression() {
  try {
    if (!currentDataset) throw new Error('データを先に読込んでください');
    if (!selectedTarget) throw new Error('目的変数を選択してください');
    const featureColumns = [...selectedFeatures].filter(name => name !== selectedTarget);
    if (featureColumns.length === 0) throw new Error('説明変数を1つ以上選択してください');
    const payload = {
      dataset_id: currentDataset.dataset_id,
      feature_columns: featureColumns,
      target_column: selectedTarget,
      direction: document.getElementById('direction').value,
      model_type: document.getElementById('modelType').value,
      fit_maxiter: Number(document.getElementById('fitMaxiter').value),
      search_space: collectSearchSpace(),
      constraints: parseJson('constraintsJson') || [],
      k_sparse: parseJson('kSparseJson'),
      acquisition: {name: document.getElementById('acqName').value, beta: Number(document.getElementById('beta').value), acqf_kwargs: {}},
      optimizer: {name: 'optimize_acqf', q: Number(document.getElementById('q').value), num_restarts: Number(document.getElementById('numRestarts').value), raw_samples: Number(document.getElementById('rawSamples').value), sequential: true}
    };
    setStatus('runStatus', '実行中...');
    const res = await fetch('/api/regression/run', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || 'run failed');
    currentResult = json;
    setStatus('runStatus', {summary: json.metadata, best_observed: json.best_observed});
    renderCandidates(json.candidates);
    renderChart(json.candidates);
    document.querySelector('nav button[data-page="results"]').click();
  } catch (err) {
    setStatus('runStatus', 'ERROR: ' + err.message);
  }
}

function renderCandidates(rows) {
  if (!rows || rows.length === 0) { document.getElementById('candidateTable').innerHTML = '<p class="muted">No candidates</p>'; return; }
  const featureCols = Object.keys(rows[0].values);
  const html = '<table><thead><tr><th>rank</th>' + featureCols.map(c => `<th>${c}</th>`).join('') + '<th>pred_mean</th><th>pred_std</th><th>acq</th><th>constraints</th></tr></thead><tbody>' +
    rows.map(r => '<tr>' +
      `<td>${r.rank}</td>` +
      featureCols.map(c => `<td>${r.values[c]}</td>`).join('') +
      `<td>${Number(r.predicted_target_mean).toPrecision(5)}</td><td>${Number(r.predicted_target_std).toPrecision(5)}</td><td>${r.acq_value == null ? '' : Number(r.acq_value).toPrecision(5)}</td><td class="${r.constraints_ok ? 'ok' : 'bad'}">${r.constraints_ok ? 'OK' : 'NG'}</td>` +
    '</tr>').join('') + '</tbody></table>';
  document.getElementById('candidateTable').innerHTML = html;
}

function renderChart(rows) {
  const canvas = document.getElementById('resultChart');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!rows || rows.length === 0) return;
  const values = rows.map(r => Number(r.predicted_target_mean));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = 40;
  ctx.strokeStyle = '#cdd4df'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, 20); ctx.lineTo(pad, canvas.height - pad); ctx.lineTo(canvas.width - 20, canvas.height - pad); ctx.stroke();
  ctx.fillStyle = '#172033'; ctx.font = '12px sans-serif'; ctx.fillText('predicted target mean', pad, 14);
  values.forEach((v, i) => {
    const x = pad + (i + 1) * ((canvas.width - pad - 40) / (values.length + 1));
    const y = canvas.height - pad - ((v - min) / Math.max(max - min, 1e-12)) * (canvas.height - 70);
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillText(String(i + 1), x - 4, canvas.height - 20);
  });
}

function downloadCandidates() {
  if (!currentResult) return;
  const blob = new Blob([JSON.stringify(currentResult, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'bochan_regression_candidates.json';
  a.click();
  URL.revokeObjectURL(url);
}
</script>
</body>
</html>
"""
