import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Data } from "plotly.js";
import { useWorkbench } from "../context/WorkbenchContext";
import {
  fetchExperimentHistory,
  type ExperimentCycle,
  type ExperimentHistoryResponse
} from "../experimentHistory";
import { RESULT_PLOT_CONFIG } from "../plotConfig";
import { themedPlotLayout } from "../plotLayout";

interface ExperimentHistoryPanelProps {
  datasetId: string;
  refreshKey?: number;
}

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

function cycleModelName(cycle: ExperimentCycle): string {
  return String(cycle.model.type ?? cycle.model.model_type ?? "—");
}

function cycleAcquisitionName(cycle: ExperimentCycle): string {
  return String(cycle.acquisition.name ?? cycle.acquisition.effective_acquisition ?? "—");
}

/** Displays experiment-cycle settings, appended rows, and objective progress figures. */
export default function ExperimentHistoryPanel({ datasetId, refreshKey = 0 }: ExperimentHistoryPanelProps) {
  const { theme, setError } = useWorkbench();
  const [history, setHistory] = useState<ExperimentHistoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [reloadVersion, setReloadVersion] = useState(0);

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

  const selectedVisualization = useMemo(
    () => history?.visualizations.find((visualization) => visualization.target === selectedTarget) ?? null,
    [history, selectedTarget]
  );
  const latest = history?.cycles.at(-1);

  return (
    <article className="panel experiment-history-panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">EXPERIMENT HISTORY</span>
          <h3>実験サイクル履歴</h3>
          <p>各サイクルの追加データ、モデル・獲得関数設定、目的変数の推移を確認します。</p>
        </div>
        <button
          className="secondary"
          onClick={() => setReloadVersion((current) => current + 1)}
          disabled={loading}
        >
          {loading ? "読込中" : `更新 · ${history?.count ?? 0} cycles`}
        </button>
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
            <div><span>最新獲得関数</span><strong>{latest ? cycleAcquisitionName(latest) : "—"}</strong></div>
          </div>

          <section className="history-plot-section">
            <div className="history-section-heading">
              <div>
                <h4>目的変数の推移</h4>
                <p>サイクル内ベスト・平均・初期データを含む累積ベストを比較します。</p>
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

          <section className="history-cycle-section">
            <div className="history-section-heading">
              <div>
                <h4>サイクル別データと探索設定</h4>
                <p>行を開くと、実際に追加した条件・結果と詳細設定を確認できます。</p>
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
                    <th>獲得関数</th>
                    <th>目的変数</th>
                  </tr>
                </thead>
                <tbody>
                  {[...history.cycles].reverse().map((cycle) => (
                    <tr key={cycle.cycle_id}>
                      <td><span className="rank">{cycle.cycle_number}</span></td>
                      <td>{formatDate(cycle.created_at)}</td>
                      <td>{cycle.appended_rows} rows<br /><small>{cycle.append_mode === "manual" ? "直接入力" : "ファイル"}</small></td>
                      <td>{cycleModelName(cycle)}<br /><small>学習 {textSetting(cycle.model.n_train)}件</small></td>
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
                    <strong>{cycle.appended_rows}件 · {cycleModelName(cycle)} · {cycleAcquisitionName(cycle)}</strong>
                    <small>{formatDate(cycle.created_at)}</small>
                  </summary>
                  <div className="history-detail-body">
                    <div className="history-setting-grid">
                      <div><span>Dataset</span><strong>{cycle.dataset_name}</strong></div>
                      <div><span>Rows</span><strong>{cycle.n_rows_before} → {cycle.n_rows_after}</strong></div>
                      <div><span>Model</span><strong>{cycleModelName(cycle)}</strong><small>{textSetting(cycle.model.details)}</small></div>
                      <div><span>Acquisition</span><strong>{cycleAcquisitionName(cycle)}</strong><small>{textSetting(cycle.acquisition)}</small></div>
                      <div><span>Optimizer</span><strong>{textSetting(cycle.optimizer.backend ?? cycle.optimizer.name)}</strong><small>{textSetting(cycle.optimizer)}</small></div>
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
