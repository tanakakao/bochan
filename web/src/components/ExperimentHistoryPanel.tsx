import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Data } from "plotly.js";
import { useWorkbench } from "../context/WorkbenchContext";
import {
  fetchExperimentHistory,
  type ExperimentCycle,
  type ExperimentHistoryResponse
} from "../experimentHistory";
import {
  defaultExperimentProjectFilename,
  downloadExperimentProject,
  normalizeExperimentProjectFilename
} from "../experimentProject";
import { RESULT_PLOT_CONFIG } from "../plotConfig";
import { themedPlotLayout } from "../plotLayout";
import "../experiment-history-cycle-plots.css";

interface ExperimentHistoryPanelProps {
  datasetId: string;
  refreshKey?: number;
}

interface ParetoPoint {
  x: number;
  y: number;
  cycle: ExperimentCycle;
  row: Record<string, unknown>;
  rowIndex: number;
}

const CYCLE_SYMBOLS = [
  "circle",
  "square",
  "diamond",
  "cross",
  "x",
  "triangle-up",
  "triangle-down",
  "star",
  "hexagon",
  "pentagon"
];

function formatNumber(value: unknown): string {
  const converted = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(converted)) return value === null || value === undefined ? "—" : String(value);
  return Math.abs(converted) >= 1000 || (Math.abs(converted) > 0 && Math.abs(converted) < 0.001)
    ? converted.toExponential(4)
    : converted.toFixed(4).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("ja-JP");
}

function textSetting(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function recordSetting(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function cycleModelDetails(cycle: ExperimentCycle): Record<string, unknown> {
  return recordSetting(cycle.model.details);
}

function cycleModelName(cycle: ExperimentCycle): string {
  const details = cycleModelDetails(cycle);
  return String(
    cycle.model.type ??
    cycle.model.model_type ??
    details.requested_model_type ??
    details.internal_model_type ??
    "—"
  );
}

function cycleModelClass(cycle: ExperimentCycle): string {
  const details = cycleModelDetails(cycle);
  return String(cycle.model.model_class ?? details.model_class ?? "—");
}

function shortClassName(value: string): string {
  if (!value || value === "—") return value;
  return value.split(".").at(-1) ?? value;
}

function cycleAcquisitionName(cycle: ExperimentCycle): string {
  const details = cycleModelDetails(cycle);
  return String(
    cycle.acquisition.name ??
    cycle.acquisition.effective_acquisition ??
    details.effective_acquisition ??
    "—"
  );
}

function cycleSearchMethod(cycle: ExperimentCycle): string {
  const details = cycleModelDetails(cycle);
  return String(
    cycle.optimizer.search_method ??
    cycle.optimizer.requested_search_method ??
    cycle.optimizer.requested_optimizer ??
    details.requested_search_method ??
    cycle.optimizer.name ??
    "—"
  );
}

function cycleEffectiveOptimizer(cycle: ExperimentCycle): string {
  const details = cycleModelDetails(cycle);
  return String(
    cycle.optimizer.effective_optimizer ??
    details.effective_optimizer ??
    cycle.optimizer.backend ??
    "—"
  );
}

function cycleOptimizerBackend(cycle: ExperimentCycle): string {
  const details = cycleModelDetails(cycle);
  return String(details.optimizer_backend ?? cycle.optimizer.backend ?? "—");
}

function htmlText(value: unknown): string {
  return formatNumber(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function cyclePointHover(
  cycle: ExperimentCycle,
  row: Record<string, unknown>,
  rowIndex: number
): string {
  const values = [...cycle.feature_columns, ...cycle.target_columns]
    .map((column) => `${htmlText(column)}: ${htmlText(row[column])}`)
    .join("<br>");
  return [
    `Cycle ${cycle.cycle_number} · data ${rowIndex + 1}`,
    `モデル: ${htmlText(cycleModelName(cycle))}`,
    `探索手法: ${htmlText(cycleSearchMethod(cycle))}`,
    `実行Optimizer: ${htmlText(cycleEffectiveOptimizer(cycle))}`,
    `獲得関数: ${htmlText(cycleAcquisitionName(cycle))}`,
    values
  ].filter(Boolean).join("<br>");
}

function uniqueFeatureColumns(history: ExperimentHistoryResponse | null): string[] {
  if (!history) return [];
  const columns: string[] = [];
  for (const cycle of history.cycles) {
    for (const column of cycle.feature_columns) {
      if (!columns.includes(column)) columns.push(column);
    }
  }
  return columns;
}

function regressionObjectiveTargets(history: ExperimentHistoryResponse | null): string[] {
  if (!history) return [];
  const targets: string[] = [];
  for (const cycle of history.cycles) {
    for (const target of cycle.target_columns) {
      const setting = cycle.target_settings.find((value) => value.target === target);
      const taskType = setting?.task_type ?? cycle.target_summary[target]?.task_type;
      if (taskType !== "regression" || setting?.optimize === false || targets.includes(target)) continue;
      targets.push(target);
    }
  }
  return targets;
}

function targetDirection(history: ExperimentHistoryResponse | null, target: string): string {
  if (!history) return "maximize";
  for (const cycle of [...history.cycles].reverse()) {
    const setting = cycle.target_settings.find((value) => value.target === target);
    if (setting?.direction) return setting.direction;
    if (cycle.target_summary[target]?.direction) return cycle.target_summary[target].direction;
  }
  return "maximize";
}

function cycleScatterTraces(
  history: ExperimentHistoryResponse | null,
  xColumn: string,
  yColumn: string
): Data[] {
  if (!history || !xColumn || !yColumn || xColumn === yColumn) return [];
  return history.cycles.flatMap((cycle, cycleIndex) => {
    const rows = cycle.rows.filter((row) => (
      row[xColumn] !== null && row[xColumn] !== undefined &&
      row[yColumn] !== null && row[yColumn] !== undefined
    ));
    if (!rows.length) return [];
    return [{
      type: "scatter",
      mode: "markers",
      name: `Cycle ${cycle.cycle_number}`,
      x: rows.map((row) => row[xColumn]) as any,
      y: rows.map((row) => row[yColumn]) as any,
      text: rows.map((row, rowIndex) => cyclePointHover(cycle, row, rowIndex)),
      marker: {
        size: cycle.cycle_number === 0 ? 8 : 11,
        opacity: cycle.cycle_number === 0 ? 0.48 : 0.84,
        symbol: CYCLE_SYMBOLS[cycleIndex % CYCLE_SYMBOLS.length] as any,
        line: { width: cycle.cycle_number === 0 ? 1 : 0 }
      },
      hovertemplate: "%{text}<extra></extra>"
    } as Data];
  });
}

function numericValue(value: unknown): number | null {
  const converted = typeof value === "number" ? value : Number(value);
  return Number.isFinite(converted) ? converted : null;
}

function dominates(
  left: ParetoPoint,
  right: ParetoPoint,
  xDirection: string,
  yDirection: string
): boolean {
  const xBetterOrEqual = xDirection === "minimize" ? left.x <= right.x : left.x >= right.x;
  const yBetterOrEqual = yDirection === "minimize" ? left.y <= right.y : left.y >= right.y;
  const xStrict = xDirection === "minimize" ? left.x < right.x : left.x > right.x;
  const yStrict = yDirection === "minimize" ? left.y < right.y : left.y > right.y;
  return xBetterOrEqual && yBetterOrEqual && (xStrict || yStrict);
}

function paretoPoints(
  history: ExperimentHistoryResponse | null,
  xTarget: string,
  yTarget: string
): ParetoPoint[] {
  if (!history || !xTarget || !yTarget || xTarget === yTarget) return [];
  return history.cycles.flatMap((cycle) => cycle.rows.flatMap((row, rowIndex) => {
    const x = numericValue(row[xTarget]);
    const y = numericValue(row[yTarget]);
    return x === null || y === null ? [] : [{ x, y, cycle, row, rowIndex }];
  }));
}

function cumulativeParetoFront(
  points: ParetoPoint[],
  xDirection: string,
  yDirection: string
): ParetoPoint[] {
  return points
    .filter((point, index) => !points.some((candidate, candidateIndex) => (
      index !== candidateIndex && dominates(candidate, point, xDirection, yDirection)
    )))
    .sort((left, right) => left.x - right.x);
}

function cycleParetoTraces(
  history: ExperimentHistoryResponse | null,
  xTarget: string,
  yTarget: string
): Data[] {
  const points = paretoPoints(history, xTarget, yTarget);
  if (!history || !points.length) return [];
  const traces: Data[] = history.cycles.flatMap((cycle, cycleIndex) => {
    const values = points.filter((point) => point.cycle.cycle_id === cycle.cycle_id);
    if (!values.length) return [];
    return [{
      type: "scatter",
      mode: "markers",
      name: `Cycle ${cycle.cycle_number}`,
      x: values.map((point) => point.x),
      y: values.map((point) => point.y),
      text: values.map((point) => cyclePointHover(cycle, point.row, point.rowIndex)),
      marker: {
        size: cycle.cycle_number === 0 ? 8 : 11,
        opacity: cycle.cycle_number === 0 ? 0.48 : 0.86,
        symbol: CYCLE_SYMBOLS[cycleIndex % CYCLE_SYMBOLS.length] as any,
        line: { width: cycle.cycle_number === 0 ? 1 : 0 }
      },
      hovertemplate: "%{text}<extra></extra>"
    } as Data];
  });
  const xDirection = targetDirection(history, xTarget);
  const yDirection = targetDirection(history, yTarget);
  const front = cumulativeParetoFront(points, xDirection, yDirection);
  if (front.length) {
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      name: "累積Pareto front",
      x: front.map((point) => point.x),
      y: front.map((point) => point.y),
      text: front.map((point) => cyclePointHover(point.cycle, point.row, point.rowIndex)),
      marker: { size: 10, symbol: "diamond-open" },
      line: { dash: "dash", width: 2 },
      hovertemplate: "%{text}<extra></extra>"
    } as Data);
  }
  return traces;
}

/** Displays experiment-cycle settings, appended rows, and cycle-aware progress figures. */
export default function ExperimentHistoryPanel({ datasetId, refreshKey = 0 }: ExperimentHistoryPanelProps) {
  const {
    theme,
    setError,
    dataset,
    result,
    featureColumns,
    targetColumn,
    targetColumns,
    selectedTargetSettings,
    targetDirections,
    direction,
    selectedVariables,
    normalize,
    inputPerturbation,
    nW,
    perturbationStd,
    projectionDimensions,
    modelType,
    acquisitionFamily,
    acquisition,
    beta,
    fitMaxiter,
    q,
    numRestarts,
    rawSamples
  } = useWorkbench();
  const suggestedProjectFilename = defaultExperimentProjectFilename(dataset?.name ?? "bochan_project");
  const [history, setHistory] = useState<ExperimentHistoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [includeLatestModel, setIncludeLatestModel] = useState(true);
  const [includePastModels, setIncludePastModels] = useState(false);
  const [projectFilename, setProjectFilename] = useState(suggestedProjectFilename);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [selectedFeatureX, setSelectedFeatureX] = useState("");
  const [selectedFeatureY, setSelectedFeatureY] = useState("");
  const [selectedParetoX, setSelectedParetoX] = useState("");
  const [selectedParetoY, setSelectedParetoY] = useState("");
  const [reloadVersion, setReloadVersion] = useState(0);

  useEffect(() => {
    setProjectFilename(suggestedProjectFilename);
  }, [suggestedProjectFilename]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchExperimentHistory(datasetId)
      .then((response) => {
        if (!active) return;
        setHistory(response);
        setSelectedTarget((current) => (
          response.targets.includes(current) ? current : response.targets[0] ?? ""
        ));
      })
      .catch((caught) => {
        if (active) setError(caught instanceof Error ? caught.message : String(caught));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [datasetId, refreshKey, reloadVersion, setError]);

  const historyFeatures = useMemo(() => uniqueFeatureColumns(history), [history]);
  const paretoTargets = useMemo(() => regressionObjectiveTargets(history), [history]);

  useEffect(() => {
    setSelectedFeatureX((current) => (
      historyFeatures.includes(current) ? current : historyFeatures[0] ?? ""
    ));
    setSelectedFeatureY((current) => (
      historyFeatures.includes(current) && current !== (historyFeatures[0] ?? "")
        ? current
        : historyFeatures[1] ?? historyFeatures[0] ?? ""
    ));
  }, [historyFeatures]);

  useEffect(() => {
    setSelectedParetoX((current) => (
      paretoTargets.includes(current) ? current : paretoTargets[0] ?? ""
    ));
    setSelectedParetoY((current) => (
      paretoTargets.includes(current) && current !== (paretoTargets[0] ?? "")
        ? current
        : paretoTargets[1] ?? paretoTargets[0] ?? ""
    ));
  }, [paretoTargets]);

  const selectedVisualization = useMemo(
    () => history?.visualizations.find((visualization) => visualization.target === selectedTarget) ?? null,
    [history, selectedTarget]
  );
  const featureScatterData = useMemo(
    () => cycleScatterTraces(history, selectedFeatureX, selectedFeatureY),
    [history, selectedFeatureX, selectedFeatureY]
  );
  const paretoData = useMemo(
    () => cycleParetoTraces(history, selectedParetoX, selectedParetoY),
    [history, selectedParetoX, selectedParetoY]
  );
  const latest = history?.cycles.at(-1);
  const canExport = Boolean(dataset && result && dataset.dataset_id === datasetId);

  async function exportProject() {
    if (!dataset || !result) return;
    const filename = normalizeExperimentProjectFilename(projectFilename, suggestedProjectFilename);
    setProjectFilename(filename);
    try {
      setExporting(true);
      setError(null);
      await downloadExperimentProject({
        datasetId: dataset.dataset_id,
        featureColumns,
        targetColumn,
        targetColumns,
        targetSettings: selectedTargetSettings,
        targetDirections,
        direction,
        modelType,
        projectionDimensions,
        fitMaxiter,
        normalize,
        inputPerturbation,
        nW,
        perturbationStd,
        acquisitionFamily,
        acquisition,
        beta,
        q,
        numRestarts,
        rawSamples,
        searchSpace: selectedVariables
      }, result, dataset.name, {
        includeLatestModel,
        includePastModels,
        filename
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setExporting(false);
    }
  }

  return (
    <article className="panel experiment-history-panel">
      <div className="panel-title history-panel-title">
        <div>
          <span className="panel-kicker">EXPERIMENT HISTORY</span>
          <h3>実験サイクル履歴</h3>
          <p>サイクルごとの追加データ、モデル・探索設定、条件空間と目的空間の推移を確認します。</p>
        </div>
        <div className="history-panel-actions">
          <div className="history-export-options">
            <label className="history-export-filename">
              保存名
              <input
                type="text"
                value={projectFilename}
                onChange={(event) => setProjectFilename(event.target.value)}
                onBlur={() => setProjectFilename(
                  normalizeExperimentProjectFilename(projectFilename, suggestedProjectFilename)
                )}
                disabled={exporting}
                aria-label="プロジェクト保存名"
              />
            </label>
            <label>
              <input
                type="checkbox"
                checked={includeLatestModel}
                onChange={(event) => setIncludeLatestModel(event.target.checked)}
                disabled={exporting}
              />
              最新モデルを含める
            </label>
            <label>
              <input
                type="checkbox"
                checked={includePastModels}
                onChange={(event) => setIncludePastModels(event.target.checked)}
                disabled={exporting}
              />
              過去サイクルのモデルも含める（標準OFF）
            </label>
          </div>
          <div className="history-action-buttons">
            <button
              onClick={() => void exportProject()}
              disabled={!canExport || exporting}
              title="データ、履歴、探索設定と選択した学習済みモデルを指定名のZIPへ保存します。"
            >
              {exporting ? "保存中" : "履歴込みプロジェクトを保存"}
            </button>
            <button
              className="secondary"
              onClick={() => setReloadVersion((current) => current + 1)}
              disabled={loading}
            >
              {loading ? "読込中" : `更新 · ${history?.count ?? 0} cycles`}
            </button>
          </div>
        </div>
      </div>

      <div className="alert warning">
        通常保存ではデータ・履歴・設定と最新の利用可能なモデルをまとめます。過去サイクルのモデルは標準では保存しません。モデルを含むZIPは信頼できる環境でのみ読み込んでください。
      </div>

      {loading && !history && <div className="empty-state">履歴を読み込んでいます。</div>}

      {!loading && history?.count === 0 && (
        <div className="empty-state">まだ完了した実験サイクルはありません。</div>
      )}

      {history && history.count > 0 && (
        <>
          <div className="history-overview-grid">
            <div><span>サイクル数</span><strong>{history.count}</strong></div>
            <div><span>最新追加件数</span><strong>{latest?.appended_rows ?? "—"}</strong></div>
            <div><span>最新モデル</span><strong>{latest ? cycleModelName(latest) : "—"}</strong></div>
            <div><span>最新探索手法</span><strong>{latest ? cycleSearchMethod(latest) : "—"}</strong></div>
            <div><span>最新獲得関数</span><strong>{latest ? cycleAcquisitionName(latest) : "—"}</strong></div>
          </div>

          <section className="history-plot-section">
            <div className="history-section-heading">
              <div>
                <h4>目的変数の推移</h4>
                <p>各データの実測値と、サイクル内ベスト・平均・累積ベストを比較します。</p>
              </div>
              <label>
                目的変数
                <select value={selectedTarget} onChange={(event) => setSelectedTarget(event.target.value)}>
                  {history.targets.map((target) => <option key={target} value={target}>{target}</option>)}
                </select>
              </label>
            </div>
            {selectedVisualization ? (
              <div className="history-plot-card">
                <Plot
                  data={selectedVisualization.figure.data as Data[]}
                  layout={themedPlotLayout(selectedVisualization.figure.layout, theme)}
                  config={RESULT_PLOT_CONFIG}
                  useResizeHandler
                  style={{ width: "100%", height: "100%" }}
                />
              </div>
            ) : (
              <div className="empty-state">選択した目的変数にはグラフ化できる実測値がありません。</div>
            )}
          </section>

          <section className="history-plot-section history-secondary-plot-section">
            <div className="history-section-heading">
              <div>
                <h4>説明変数の探索推移</h4>
                <p>サイクルごとに色とマーカーを変え、探索した条件領域の移動を表示します。</p>
              </div>
              <div className="history-axis-controls">
                <label>
                  X軸
                  <select value={selectedFeatureX} onChange={(event) => setSelectedFeatureX(event.target.value)}>
                    {historyFeatures.map((feature) => <option key={feature} value={feature}>{feature}</option>)}
                  </select>
                </label>
                <label>
                  Y軸
                  <select value={selectedFeatureY} onChange={(event) => setSelectedFeatureY(event.target.value)}>
                    {historyFeatures.map((feature) => <option key={feature} value={feature}>{feature}</option>)}
                  </select>
                </label>
              </div>
            </div>
            {featureScatterData.length ? (
              <div className="history-plot-card">
                <Plot
                  data={featureScatterData}
                  layout={themedPlotLayout({
                    title: `${selectedFeatureX} × ${selectedFeatureY}: サイクル別探索条件`,
                    autosize: true,
                    margin: { l: 64, r: 30, t: 70, b: 58 },
                    legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "left", x: 0 },
                    xaxis: { title: { text: selectedFeatureX } },
                    yaxis: { title: { text: selectedFeatureY } }
                  }, theme)}
                  config={RESULT_PLOT_CONFIG}
                  useResizeHandler
                  style={{ width: "100%", height: "100%" }}
                />
              </div>
            ) : (
              <div className="empty-state">異なる2つの説明変数を選択してください。</div>
            )}
          </section>

          {paretoTargets.length >= 2 && (
            <section className="history-plot-section history-secondary-plot-section">
              <div className="history-section-heading">
                <div>
                  <h4>多目的パレート推移</h4>
                  <p>各実験点をサイクル別に表示し、全履歴に対する累積Pareto frontを重ねます。</p>
                </div>
                <div className="history-axis-controls">
                  <label>
                    X目的
                    <select value={selectedParetoX} onChange={(event) => setSelectedParetoX(event.target.value)}>
                      {paretoTargets.map((target) => <option key={target} value={target}>{target}</option>)}
                    </select>
                  </label>
                  <label>
                    Y目的
                    <select value={selectedParetoY} onChange={(event) => setSelectedParetoY(event.target.value)}>
                      {paretoTargets.map((target) => <option key={target} value={target}>{target}</option>)}
                    </select>
                  </label>
                </div>
              </div>
              {paretoData.length ? (
                <div className="history-plot-card">
                  <Plot
                    data={paretoData}
                    layout={themedPlotLayout({
                      title: `${selectedParetoX} × ${selectedParetoY}: サイクル別Pareto履歴`,
                      autosize: true,
                      margin: { l: 64, r: 30, t: 70, b: 58 },
                      legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "left", x: 0 },
                      xaxis: { title: { text: selectedParetoX } },
                      yaxis: { title: { text: selectedParetoY } }
                    }, theme)}
                    config={RESULT_PLOT_CONFIG}
                    useResizeHandler
                    style={{ width: "100%", height: "100%" }}
                  />
                </div>
              ) : (
                <div className="empty-state">異なる2つの回帰目的を選択してください。</div>
              )}
            </section>
          )}

          <section className="history-cycle-section">
            <div className="history-section-heading">
              <div>
                <h4>サイクル別データと探索設定</h4>
                <p>モデル種別、探索手法、実行optimizer、追加条件・結果を確認できます。</p>
              </div>
            </div>

            <div className="table-wrap history-summary-wrap">
              <table className="history-summary-table">
                <thead>
                  <tr>
                    <th>Cycle</th>
                    <th>日時</th>
                    <th>追加</th>
                    <th>モデル</th>
                    <th>探索手法</th>
                    <th>獲得関数</th>
                    <th>目的変数</th>
                  </tr>
                </thead>
                <tbody>
                  {[...history.cycles].reverse().map((cycle) => (
                    <tr key={cycle.cycle_id}>
                      <td><span className="rank">{cycle.cycle_number}</span></td>
                      <td>{formatDate(cycle.created_at)}</td>
                      <td>{cycle.appended_rows} rows<br /><small>{cycle.append_mode === "initial" ? "初期データ" : cycle.append_mode === "manual" ? "直接入力" : "ファイル"}</small></td>
                      <td>{cycleModelName(cycle)}<br /><small>{shortClassName(cycleModelClass(cycle))} · 学習 {textSetting(cycle.model.n_train)}件</small></td>
                      <td>{cycleSearchMethod(cycle)}<br /><small>{cycleEffectiveOptimizer(cycle)}</small></td>
                      <td>{cycleAcquisitionName(cycle)}<br /><small>q={textSetting(cycle.optimizer.q)}</small></td>
                      <td>
                        {cycle.target_columns.map((target) => (
                          <div key={target} className="history-target-summary">
                            <strong>{target}</strong>
                            <span>best {formatNumber(cycle.target_summary[target]?.best)}</span>
                            <span>mean {formatNumber(cycle.target_summary[target]?.mean)}</span>
                          </div>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="history-cycle-details">
              {[...history.cycles].reverse().map((cycle) => (
                <details key={cycle.cycle_id}>
                  <summary>
                    <span>Cycle {cycle.cycle_number}</span>
                    <strong>{cycle.appended_rows}件 · {cycleModelName(cycle)} · {cycleSearchMethod(cycle)} · {cycleAcquisitionName(cycle)}</strong>
                    <small>{formatDate(cycle.created_at)}</small>
                  </summary>
                  <div className="history-detail-body">
                    <div className="history-setting-grid">
                      <div><span>Dataset</span><strong>{cycle.dataset_name}</strong></div>
                      <div><span>Rows</span><strong>{cycle.n_rows_before} → {cycle.n_rows_after}</strong></div>
                      <div><span>Model type</span><strong>{cycleModelName(cycle)}</strong><small>{textSetting(cycle.model.details)}</small></div>
                      <div><span>Model class</span><strong>{cycleModelClass(cycle)}</strong></div>
                      <div><span>Acquisition</span><strong>{cycleAcquisitionName(cycle)}</strong><small>{textSetting(cycle.acquisition)}</small></div>
                      <div><span>Search method</span><strong>{cycleSearchMethod(cycle)}</strong><small>normal / torch / GA / SA / PSO / CMA-ES / TS / NSGA-II</small></div>
                      <div><span>Effective optimizer</span><strong>{cycleEffectiveOptimizer(cycle)}</strong><small>{textSetting(cycle.optimizer)}</small></div>
                      <div><span>Optimizer backend</span><strong>{cycleOptimizerBackend(cycle)}</strong></div>
                      <div><span>Candidate settings</span><strong>q={textSetting(cycle.optimizer.q)}</strong><small>restarts={textSetting(cycle.optimizer.num_restarts)} · raw={textSetting(cycle.optimizer.raw_samples)}</small></div>
                      <div><span>Source run</span><strong>{cycle.source_run_id ?? "—"}</strong></div>
                    </div>
                    <div className="table-wrap history-data-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>#</th>
                            {[...cycle.feature_columns, ...cycle.target_columns].map((column) => (
                              <th key={column}>{column}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {cycle.rows.map((row, index) => (
                            <tr key={`${cycle.cycle_id}-${index}`}>
                              <td>{index + 1}</td>
                              {[...cycle.feature_columns, ...cycle.target_columns].map((column) => (
                                <td key={column}>{formatNumber(row[column])}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </details>
              ))}
            </div>
          </section>
        </>
      )}
    </article>
  );
}
