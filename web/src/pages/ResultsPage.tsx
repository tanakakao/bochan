import { useMemo, useState } from "react";
import { EmptyState, MetricCard, SectionHeader } from "../components/Common";
import ResultVisualizations from "../ResultVisualizations";
import { useWorkbench } from "../context/WorkbenchContext";

/**
 * Formats numeric table values for compact display.
 *
 * Args:
 *   value: Numeric value from the API.
 *
 * Returns:
 *   A human-readable number or an em dash for missing values.
 */
function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)
    ? value.toExponential(4)
    : value.toFixed(4).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

/** Renders candidate results and selectable-axis visualizations. */
export default function ResultsPage() {
  const { result, setStep } = useWorkbench();
  const [xAxis, setXAxis] = useState("rank");
  const [yAxis, setYAxis] = useState("predicted_target_mean");


  const axisOptions = useMemo(() => [
    { value: "rank", label: "順位" },
    ...(result?.feature_columns ?? []).map((column) => ({ value: `value:${column}`, label: column })),
    { value: "predicted_target_mean", label: "予測平均" },
    { value: "predicted_target_std", label: "予測標準偏差" },
    { value: "acq_value", label: "獲得値" }
  ], [result?.feature_columns]);

  const axisVisualization = useMemo(() => {
    const readValue = (candidate: NonNullable<typeof result>["candidates"][number], key: string) => {
      if (key === "rank") return candidate.rank;
      if (key === "predicted_target_mean") return candidate.predicted_target_mean;
      if (key === "predicted_target_std") return candidate.predicted_target_std;
      if (key === "acq_value") return candidate.acq_value;
      if (key.startsWith("value:")) return candidate.values[key.slice(6)];
      return null;
    };
    const xLabel = axisOptions.find((option) => option.value === xAxis)?.label ?? xAxis;
    const yLabel = axisOptions.find((option) => option.value === yAxis)?.label ?? yAxis;
    return {
      id: "selected_axes",
      title: "選択軸グラフ",
      description: "候補テーブルから縦軸・横軸を選んで表示します。",
      figure: {
        data: [{
          type: "scatter",
          mode: "markers+text",
          x: (result?.candidates ?? []).map((candidate) => readValue(candidate, xAxis)),
          y: (result?.candidates ?? []).map((candidate) => readValue(candidate, yAxis)),
          text: (result?.candidates ?? []).map((candidate) => `#${candidate.rank}`),
          textposition: "top center"
        }],
        layout: { xaxis: { title: xLabel }, yaxis: { title: yLabel }, margin: { t: 24, r: 24, b: 56, l: 64 } }
      }
    };
  }, [axisOptions, result?.candidates, xAxis, yAxis]);


  if (!result) {
    return (
      <>
        <SectionHeader
          step="4 · RESULTS"
          title="候補と予測結果を確認する"
          text="Optimizeページで候補を生成してください。"
        />
        <EmptyState>候補生成結果がありません。</EmptyState>
      </>
    );
  }

  return (
    <>
      <SectionHeader
        step="4 · RESULTS"
        title="候補と予測結果を確認する"
        text={`${result.model_type} · 学習 ${result.n_train}件 · best observed ${formatNumber(result.best_observed)}`}
        action={
          <>
            <button className="secondary" onClick={() => setStep("optimize")}>設定を変更</button>
            <button onClick={() => setStep("logs")}>実行ログ</button>
          </>
        }
      />

      <div className="cards metric-grid">
        <MetricCard icon="◎" label="Target" value={result.target_column} detail="目的変数" />
        <MetricCard icon="↗" label="Direction" value={result.direction === "maximize" ? "最大化" : "最小化"} />
        <MetricCard icon="◇" label="Features" value={result.n_features} detail={result.feature_columns.join(", ")} />
        <MetricCard icon="▧" label="Candidates" value={result.candidates.length} detail="提案数" tone="success" />
      </div>

      <article className="panel compact-panel">
        <div className="panel-title"><div><span className="panel-kicker">GRAPH AXES</span><h3>グラフ軸の選択</h3><p>候補グラフの横軸・縦軸に使う項目を選択します。</p></div></div>
        <div className="form-grid candidate-settings">
          <label>横軸<select value={xAxis} onChange={(event) => setXAxis(event.target.value)}>{axisOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
          <label>縦軸<select value={yAxis} onChange={(event) => setYAxis(event.target.value)}>{axisOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
        </div>
      </article>

      <ResultVisualizations
        visualizations={[axisVisualization, ...(result.visualizations ?? [])]}
        warnings={result.visualization_warnings ?? []}
      />

      <article className="panel best-model-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">RECOMMENDED CANDIDATES</span>
            <h3>推奨候補</h3>
            <p>予測平均、予測標準偏差、獲得関数値をまとめて確認します。</p>
          </div>
          <span className="status-chip success">Ready</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>順位</th>
                {result.feature_columns.map((column) => <th key={column}>{column}</th>)}
                <th>予測平均</th>
                <th>予測標準偏差</th>
                <th>獲得値</th>
                <th>制約</th>
              </tr>
            </thead>
            <tbody>
              {result.candidates.map((candidate) => (
                <tr key={candidate.rank} className={candidate.rank === 1 ? "candidate-best" : ""}>
                  <td><span className="rank">{candidate.rank}</span></td>
                  {result.feature_columns.map((column) => (
                    <td key={column}>
                      {typeof candidate.values[column] === "number"
                        ? formatNumber(candidate.values[column] as number)
                        : String(candidate.values[column])}
                    </td>
                  ))}
                  <td>{formatNumber(candidate.predicted_target_mean)}</td>
                  <td>{formatNumber(candidate.predicted_target_std)}</td>
                  <td>{formatNumber(candidate.acq_value)}</td>
                  <td><span className={`status-chip ${candidate.constraints_ok ? "success" : "warning"}`}>{candidate.constraints_ok ? "OK" : "NG"}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </>
  );
}
