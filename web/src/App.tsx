import { useMemo, useState } from "react";
import { runRegression, uploadDataset } from "./api";
import ResultVisualizations from "./ResultVisualizations";
import type { ColumnProfile, DatasetResponse, RegressionResult, SearchVariable } from "./types";

const STEPS = ["データ", "変数", "探索設定", "結果"];

function numberOrUndefined(value: string): number | undefined {
  if (value.trim() === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  return Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)
    ? value.toExponential(4)
    : value.toFixed(4).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

function createVariable(column: ColumnProfile): SearchVariable {
  if (column.kind === "categorical") {
    return {
      name: column.name,
      type: "categorical",
      fixed: false,
      categories: column.values ?? []
    };
  }
  return {
    name: column.name,
    type: "numeric",
    lower: column.min ?? undefined,
    upper: column.max ?? undefined,
    fixed: false
  };
}

export default function App() {
  const [step, setStep] = useState(0);
  const [dataset, setDataset] = useState<DatasetResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState("");
  const [variables, setVariables] = useState<Record<string, SearchVariable>>({});
  const [direction, setDirection] = useState<"maximize" | "minimize">("maximize");
  const [modelType, setModelType] = useState("base");
  const [acquisition, setAcquisition] = useState("EI");
  const [beta, setBeta] = useState(2);
  const [fitMaxiter, setFitMaxiter] = useState(128);
  const [q, setQ] = useState(3);
  const [numRestarts, setNumRestarts] = useState(10);
  const [rawSamples, setRawSamples] = useState(256);
  const [result, setResult] = useState<RegressionResult | null>(null);

  const columns = dataset?.profile.columns ?? [];
  const selectableColumns = columns.filter((column) => column.kind === "numeric" || column.kind === "categorical");
  const targetCandidates = columns.filter((column) => column.kind === "numeric");

  const selectedVariables = useMemo(
    () => featureColumns.map((name) => variables[name]).filter((value): value is SearchVariable => Boolean(value)),
    [featureColumns, variables]
  );

  async function handleFile(file: File | null) {
    if (!file) return;
    setUploading(true);
    setError(null);
    setResult(null);
    try {
      const loaded = await uploadDataset(file);
      setDataset(loaded);
      const numeric = loaded.profile.columns.filter((column) => column.kind === "numeric");
      const initialTarget = numeric.at(-1)?.name ?? "";
      const initialFeatures = loaded.profile.columns
        .filter((column) => (column.kind === "numeric" || column.kind === "categorical") && column.name !== initialTarget)
        .map((column) => column.name);
      setTargetColumn(initialTarget);
      setFeatureColumns(initialFeatures);
      setVariables(
        Object.fromEntries(
          loaded.profile.columns
            .filter((column) => column.kind === "numeric" || column.kind === "categorical")
            .map((column) => [column.name, createVariable(column)])
        )
      );
      setStep(1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setUploading(false);
    }
  }

  function toggleFeature(name: string) {
    setFeatureColumns((current) =>
      current.includes(name) ? current.filter((column) => column !== name) : [...current, name]
    );
  }

  function changeTarget(name: string) {
    setTargetColumn(name);
    setFeatureColumns((current) => current.filter((column) => column !== name));
  }

  function patchVariable(name: string, patch: Partial<SearchVariable>) {
    setVariables((current) => ({
      ...current,
      [name]: { ...current[name], ...patch }
    }));
  }

  async function execute() {
    if (!dataset) return;
    setRunning(true);
    setError(null);
    try {
      const response = await runRegression({
        datasetId: dataset.dataset_id,
        featureColumns,
        targetColumn,
        direction,
        modelType,
        fitMaxiter,
        acquisition,
        beta,
        q,
        numRestarts,
        rawSamples,
        searchSpace: selectedVariables
      });
      setResult(response);
      setStep(3);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
    }
  }

  const canMoveToSettings = Boolean(dataset && targetColumn && featureColumns.length > 0);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Bayesian optimization workspace</p>
          <h1>bochan Web</h1>
        </div>
        <span className="badge">Regression MVP</span>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <ol className="step-list">
            {STEPS.map((label, index) => (
              <li key={label}>
                <button
                  className={index === step ? "step-button active" : "step-button"}
                  onClick={() => index <= step && setStep(index)}
                  disabled={index > step}
                >
                  <span>{index + 1}</span>
                  {label}
                </button>
              </li>
            ))}
          </ol>
          <div className="sidebar-note">
            CSV / Excelから単一目的回帰モデルを学習し、次の実験候補を生成します。
          </div>
        </aside>

        <main className="content">
          {error && <div className="alert error">{error}</div>}

          {step === 0 && (
            <section>
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Step 1</p>
                  <h2>データを読み込む</h2>
                </div>
              </div>
              <div className="card upload-card">
                <label className="drop-zone">
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={(event) => handleFile(event.target.files?.[0] ?? null)}
                    disabled={uploading}
                  />
                  <strong>{uploading ? "読み込み中..." : "CSVまたはExcelを選択"}</strong>
                  <span>ファイルはFastAPIへ送信され、メモリ上で解析されます。</span>
                </label>
              </div>
            </section>
          )}

          {step === 1 && dataset && (
            <section>
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Step 2</p>
                  <h2>目的変数と説明変数</h2>
                  <p>{dataset.name} · {dataset.profile.n_rows}行 × {dataset.profile.n_columns}列</p>
                </div>
                <button className="primary" disabled={!canMoveToSettings} onClick={() => setStep(2)}>
                  探索設定へ
                </button>
              </div>

              <div className="grid two-columns">
                <div className="card">
                  <h3>目的変数</h3>
                  <label className="field">
                    <span>数値列</span>
                    <select value={targetColumn} onChange={(event) => changeTarget(event.target.value)}>
                      <option value="">選択してください</option>
                      {targetCandidates.map((column) => (
                        <option key={column.name} value={column.name}>{column.name}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="card">
                  <h3>説明変数</h3>
                  <div className="checkbox-grid">
                    {selectableColumns.filter((column) => column.name !== targetColumn).map((column) => (
                      <label key={column.name} className="check-card">
                        <input
                          type="checkbox"
                          checked={featureColumns.includes(column.name)}
                          onChange={() => toggleFeature(column.name)}
                        />
                        <span>
                          <strong>{column.name}</strong>
                          <small>{column.kind} · 欠損 {Math.round(column.missing_rate * 1000) / 10}%</small>
                        </span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>

              <div className="card">
                <h3>データプレビュー</h3>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>{columns.map((column) => <th key={column.name}>{column.name}</th>)}</tr>
                    </thead>
                    <tbody>
                      {dataset.preview.slice(0, 20).map((row, rowIndex) => (
                        <tr key={rowIndex}>
                          {columns.map((column) => <td key={column.name}>{String(row[column.name] ?? "")}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          )}

          {step === 2 && dataset && (
            <section>
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Step 3</p>
                  <h2>モデルと探索空間</h2>
                </div>
                <button className="primary" onClick={execute} disabled={running || !canMoveToSettings}>
                  {running ? "学習・探索中..." : "候補を生成"}
                </button>
              </div>

              <div className="grid three-columns">
                <div className="card">
                  <h3>目的</h3>
                  <label className="field"><span>方向</span>
                    <select value={direction} onChange={(event) => setDirection(event.target.value as "maximize" | "minimize")}>
                      <option value="maximize">最大化</option>
                      <option value="minimize">最小化</option>
                    </select>
                  </label>
                </div>
                <div className="card">
                  <h3>モデル</h3>
                  <label className="field"><span>model_type</span>
                    <select value={modelType} onChange={(event) => setModelType(event.target.value)}>
                      <option value="base">Base GP</option>
                      <option value="saas">MAP-SAAS</option>
                      <option value="deepkernel">Deep Kernel GP</option>
                    </select>
                  </label>
                  <label className="field"><span>fit maxiter</span>
                    <input type="number" min={1} value={fitMaxiter} onChange={(event) => setFitMaxiter(Number(event.target.value))} />
                  </label>
                </div>
                <div className="card">
                  <h3>獲得関数</h3>
                  <label className="field"><span>acquisition</span>
                    <select value={acquisition} onChange={(event) => setAcquisition(event.target.value)}>
                      <option value="EI">EI</option>
                      <option value="NEI">NEI</option>
                      <option value="UCB">UCB</option>
                    </select>
                  </label>
                  {acquisition === "UCB" && (
                    <label className="field"><span>beta</span>
                      <input type="number" step="0.1" value={beta} onChange={(event) => setBeta(Number(event.target.value))} />
                    </label>
                  )}
                </div>
              </div>

              <div className="card">
                <div className="card-heading">
                  <div><h3>候補生成</h3><p>q個の候補を逐次最適化します。</p></div>
                </div>
                <div className="form-row">
                  <label className="field"><span>q</span><input type="number" min={1} value={q} onChange={(event) => setQ(Number(event.target.value))} /></label>
                  <label className="field"><span>num_restarts</span><input type="number" min={1} value={numRestarts} onChange={(event) => setNumRestarts(Number(event.target.value))} /></label>
                  <label className="field"><span>raw_samples</span><input type="number" min={1} value={rawSamples} onChange={(event) => setRawSamples(Number(event.target.value))} /></label>
                </div>
              </div>

              <div className="card">
                <h3>探索変数</h3>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>変数</th><th>型</th><th>下限</th><th>上限</th><th>刻み</th><th>固定</th><th>固定値</th></tr></thead>
                    <tbody>
                      {selectedVariables.map((variable) => (
                        <tr key={variable.name}>
                          <td><strong>{variable.name}</strong></td>
                          <td>{variable.type}</td>
                          <td>{variable.type === "numeric" ? <input type="number" value={variable.lower ?? ""} onChange={(event) => patchVariable(variable.name, { lower: numberOrUndefined(event.target.value) })} /> : "-"}</td>
                          <td>{variable.type === "numeric" ? <input type="number" value={variable.upper ?? ""} onChange={(event) => patchVariable(variable.name, { upper: numberOrUndefined(event.target.value) })} /> : "-"}</td>
                          <td>{variable.type === "numeric" ? <input type="number" min={0} value={variable.step ?? ""} placeholder="任意" onChange={(event) => patchVariable(variable.name, { step: numberOrUndefined(event.target.value) })} /> : "-"}</td>
                          <td><input type="checkbox" checked={variable.fixed} onChange={(event) => patchVariable(variable.name, { fixed: event.target.checked })} /></td>
                          <td>
                            {variable.fixed ? (
                              variable.type === "categorical" ? (
                                <select value={String(variable.fixed_value ?? variable.categories?.[0] ?? "")} onChange={(event) => patchVariable(variable.name, { fixed_value: event.target.value })}>
                                  {(variable.categories ?? []).map((value) => <option key={value} value={value}>{value}</option>)}
                                </select>
                              ) : (
                                <input type="number" value={variable.fixed_value ?? ""} onChange={(event) => patchVariable(variable.name, { fixed_value: numberOrUndefined(event.target.value) })} />
                              )
                            ) : "-"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          )}

          {step === 3 && result && (
            <section>
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Step 4</p>
                  <h2>候補生成結果</h2>
                  <p>{result.model_type} · 学習 {result.n_train}件 · best observed {formatNumber(result.best_observed)}</p>
                </div>
                <button className="secondary" onClick={() => setStep(2)}>設定を変更</button>
              </div>

              <div className="metric-grid">
                <div className="metric"><span>目的変数</span><strong>{result.target_column}</strong></div>
                <div className="metric"><span>方向</span><strong>{result.direction === "maximize" ? "最大化" : "最小化"}</strong></div>
                <div className="metric"><span>説明変数</span><strong>{result.n_features}</strong></div>
                <div className="metric"><span>候補数</span><strong>{result.candidates.length}</strong></div>
              </div>

              <ResultVisualizations
                visualizations={result.visualizations ?? []}
                warnings={result.visualization_warnings ?? []}
              />

              <div className="card">
                <h3>推奨候補</h3>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>順位</th>
                        {result.feature_columns.map((column) => <th key={column}>{column}</th>)}
                        <th>予測平均</th>
                        <th>予測標準偏差</th>
                        <th>獲得値</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.candidates.map((candidate) => (
                        <tr key={candidate.rank}>
                          <td><span className="rank">{candidate.rank}</span></td>
                          {result.feature_columns.map((column) => <td key={column}>{typeof candidate.values[column] === "number" ? formatNumber(candidate.values[column] as number) : String(candidate.values[column])}</td>)}
                          <td>{formatNumber(candidate.predicted_target_mean)}</td>
                          <td>{formatNumber(candidate.predicted_target_std)}</td>
                          <td>{formatNumber(candidate.acq_value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
