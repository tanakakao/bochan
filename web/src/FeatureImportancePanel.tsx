import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Data } from "plotly.js";
import { useWorkbench } from "./context/WorkbenchContext";
import { RESULT_PLOT_CONFIG } from "./plotConfig";
import { themedPlotLayout } from "./plotLayout";
import type { RegressionResult } from "./types";

/** Display server-computed permutation importance without recomputing it. */
export default function FeatureImportancePanel({ result }: { result: RegressionResult }) {
  const { theme } = useWorkbench();
  const summary = result.feature_importance_summary ?? [];
  const figures = result.feature_importance_visualizations ?? [];
  const warnings = [...new Set(result.feature_importance_warnings ?? [])];
  const outputs = [...new Set(summary.map((row) => row.output_name))];
  const kinds = [...new Set(summary.map((row) => row.importance_kind))];
  const [output, setOutput] = useState(outputs[0] ?? "");
  const [kind, setKind] = useState(kinds[0] ?? "predictive");
  const rows = useMemo(() => summary
    .filter((row) => row.output_name === output && row.importance_kind === kind)
    .sort((left, right) => (right.mean ?? -Infinity) - (left.mean ?? -Infinity)), [summary, output, kind]);
  const visibleFigures = figures.filter((figure) => figure.id.includes(output) && figure.id.includes(kind));
  const diagnostics = result.model_diagnostics ?? {};
  if (!summary.length && !figures.length && !warnings.length && !Object.keys(diagnostics).length) return null;

  return <section className="visualization-section feature-importance-panel">
    <div className="result-subheading"><div><span className="eyebrow">MODEL INSPECTION</span><h3>特徴量重要度</h3><p>Permutation importanceにより、各特徴量を崩したときの予測性能の悪化量を表示します。因果効果ではありません。</p></div></div>
    {result.feature_importance_source === "training" && <div className="alert warning">学習データ上で評価しているため、重要度が楽観的な可能性があります。</div>}
    {result.feature_importance_source === "cross_validation" && <p>各foldのValidationデータで計算した重要度を集約しています。エラーバーはfold間標準偏差です。</p>}
    {warnings.length > 0 && <div className="alert warning">{warnings.map((warning) => <div key={warning}>{warning}</div>)}</div>}
    <div className="model-settings-grid">
      {outputs.length > 1 && <label>出力<select value={output} onChange={(event) => setOutput(event.target.value)}>{outputs.map((name) => <option key={name}>{name}</option>)}</select></label>}
      {kinds.length > 1 && <label>重要度種別<select value={kind} onChange={(event) => setKind(event.target.value as "predictive" | "noise" | "classwise")}>{kinds.map((name) => <option key={name} value={name}>{name === "predictive" ? "予測重要度" : name === "noise" ? "ノイズ重要度" : "クラス別重要度"}</option>)}</select></label>}
    </div>
    <div className="visualization-grid">{visibleFigures.map((visualization) => <article className="panel visualization-card" key={visualization.id}><h3>{visualization.title}</h3><p>{visualization.description}</p><div className="plot-container" style={{ height: Math.min(900, Math.max(360, rows.length * 34 + 140)) }}><Plot data={visualization.figure.data as Data[]} layout={themedPlotLayout({ ...visualization.figure.layout, margin: { l: 150, r: 30, t: 70, b: 60 } }, theme)} config={RESULT_PLOT_CONFIG} useResizeHandler style={{ width: "100%", height: "100%" }} /></div></article>)}</div>
    {rows.length > 0 && <div className="table-wrap"><table><thead><tr><th>順位</th><th>特徴量</th><th>平均重要度</th><th>標準偏差</th><th>正規化重要度</th><th>役割</th><th>種類</th></tr></thead><tbody>{rows.map((row) => <tr key={`${row.method}-${row.feature}`}><td>{row.rank ?? "—"}</td><td>{row.feature}</td><td>{row.mean ?? "—"}</td><td>{result.feature_importance_source === "cross_validation" ? row.between_fold_std ?? "—" : row.std ?? "—"}</td><td>{row.normalized_mean ?? "—"}</td><td>{row.role ?? "—"}</td><td>{row.feature_type ?? "—"}</td></tr>)}</tbody></table></div>}
    {Object.keys(diagnostics).length > 0 && <details><summary>モデル固有診断</summary><p>ARDはカーネル内部の感度診断であり、Permutation importanceや因果効果ではありません。</p><pre>{JSON.stringify(diagnostics, null, 2)}</pre></details>}
  </section>;
}
