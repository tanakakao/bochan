import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import "./compositionFeatureImportanceTypes";
import { useWorkbench } from "./context/WorkbenchContext";
import { RESULT_PLOT_CONFIG } from "./plotConfig";
import { themedPlotLayout } from "./plotLayout";
import type { RegressionResult } from "./types";

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)
    ? value.toExponential(4)
    : value.toFixed(5).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

function plotHeight(count: number): number {
  return Math.min(760, Math.max(360, count * 38 + 150));
}

/** Display constrained element-wise perturbation importance separately from normal PI. */
export default function CompositionFeatureImportancePanel({
  result
}: {
  result: RegressionResult;
}) {
  const { theme } = useWorkbench();
  const payload = result.composition_feature_importance;
  const outputs = useMemo(
    () => [...new Set((payload?.elements ?? []).map((row) => row.output_name))],
    [payload]
  );
  const [output, setOutput] = useState(outputs[0] ?? "");

  useEffect(() => {
    if (!outputs.includes(output)) setOutput(outputs[0] ?? "");
  }, [output, outputs]);

  const rows = useMemo(
    () => (payload?.elements ?? [])
      .filter((row) => row.output_name === output)
      .sort((left, right) => (right.mean ?? -Infinity) - (left.mean ?? -Infinity)),
    [output, payload]
  );

  if (!payload || rows.length === 0) return null;

  const labels = rows.map((row) => row.label ?? `${row.feature} 比率`);
  const data: Data[] = [{
    type: "bar",
    orientation: "h",
    x: rows.map((row) => row.mean ?? 0),
    y: labels,
    error_x: {
      type: "data",
      array: rows.map((row) => row.std ?? 0),
      visible: true
    },
    customdata: rows.map((row) => [
      row.normalized_mean,
      row.metric_name,
      row.baseline_metric,
      row.n_repeats
    ]),
    hovertemplate: [
      "element=%{y}",
      "importance=%{x:.6g}",
      "composition-normalized=%{customdata[0]:.6g}",
      "metric=%{customdata[1]}",
      "baseline=%{customdata[2]}",
      "repeats=%{customdata[3]}",
      "<extra></extra>"
    ].join("<br>")
  }];
  const layout: Partial<Layout> = {
    title: { text: `${output}: 組成内の元素別影響度` },
    height: plotHeight(rows.length),
    margin: { l: 170, r: 30, t: 70, b: 60 },
    xaxis: {
      title: { text: "評価指標の悪化量" },
      zeroline: false
    },
    yaxis: {
      title: { text: "元素比率" },
      autorange: "reversed"
    },
    shapes: [{
      type: "line",
      xref: "x",
      yref: "paper",
      x0: 0,
      x1: 0,
      y0: 0,
      y1: 1,
      line: { width: 1, color: "gray" }
    }]
  };

  return (
    <section className="visualization-section feature-importance-panel">
      <div className="result-subheading">
        <div>
          <span className="eyebrow">COMPOSITION INSPECTION</span>
          <h3>組成内の元素別影響度</h3>
          <p>
            各元素比率を入れ替え、{payload.mode_label}ながら合計1と組成制約を維持したときの
            予測性能低下を表示します。元素単独の因果効果ではありません。
          </p>
        </div>
      </div>

      <div className="model-settings-grid">
        {outputs.length > 1 && (
          <label>
            出力
            <select value={output} onChange={(event) => setOutput(event.target.value)}>
              {outputs.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
        )}
        <label>
          組成変化方式
          <select value={payload.mode} disabled>
            <option value={payload.mode}>{payload.mode_label}</option>
          </select>
        </label>
      </div>

      {payload.requested_source === "cross_validation" && (
        <div className="alert warning">
          通常のPermutation Importanceはcross-validation集約ですが、組成内の元素別影響度は
          制約付き摂動が必要なため、最終モデルの学習データ上で評価しています。
        </div>
      )}
      {(payload.warnings ?? []).map((warning) => (
        <div className="alert warning" key={warning}>{warning}</div>
      ))}

      <article className="panel visualization-card">
        <h3>{output}: 元素比率の影響度</h3>
        <p>
          棒は評価指標の悪化量、エラーバーは{payload.n_repeats}回の摂動間標準偏差です。
          組成内正規化値は、正の元素別重要度の合計を1として計算します。
        </p>
        <div className="plot-container" style={{ height: plotHeight(rows.length) }}>
          <Plot
            data={data}
            layout={themedPlotLayout(layout as Record<string, unknown>, theme)}
            config={RESULT_PLOT_CONFIG}
            useResizeHandler
            style={{ width: "100%", height: "100%" }}
          />
        </div>
      </article>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>順位</th>
              <th>元素</th>
              <th>平均影響度</th>
              <th>標準偏差</th>
              <th>組成内正規化</th>
              <th>評価指標</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={`${row.output_name}-${row.feature}`}>
                <td>{index + 1}</td>
                <td>{row.feature}</td>
                <td>{formatNumber(row.mean)}</td>
                <td>{formatNumber(row.std)}</td>
                <td>{formatNumber(row.normalized_mean)}</td>
                <td>{row.metric_name ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
