import { useMemo, useState } from "react";
import { EmptyState, MetricCard, SectionHeader } from "../components/Common";
import ResultVisualizations from "../ResultVisualizations";
import { useWorkbench } from "../context/WorkbenchContext";

/** Formats numeric table values for compact display. */
function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)
    ? value.toExponential(4)
    : value.toFixed(4).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

function csvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** Renders candidate results for one or more regression targets. */
export default function ResultsPage() {
  const { result, setStep } = useWorkbench();
  const targetColumns = result?.target_columns?.length
    ? result.target_columns
    : result?.target_column
      ? [result.target_column]
      : [];
  const [xAxis, setXAxis] = useState("rank");
  const [yAxis, setYAxis] = useState(() => targetColumns[0] ? `prediction:${targetColumns[0]}:mean` : "predicted_target_mean");

  const axisOptions = useMemo(() => [
    { value: "rank", label: "順位" },
    ...(result?.feature_columns ?? []).map((column) => ({ value: `value:${column}`, label: column })),
    ...targetColumns.flatMap((target) => [
      { value: `prediction:${target}:mean`, label: `${target} 予測平均` },
      { value: `prediction:${target}:std`, label: `${target} 予測標準偏差` }
    ]),
    { value: "acq_value", label: "獲得値" }
  ], [result?.feature_columns, targetColumns.join("\u0000")]);

  const axisVisualization = useMemo(() => {
    const readValue = (candidate: NonNullable<typeof result>["candidates"][number], key: string) => {
      if (key === "rank") return candidate.rank;
      if (key === "predicted_target_mean") return candidate.predicted_target_mean;
      if (key === "predicted_target_std") return candidate.predicted_target_std;
      if (key === "acq_value") return candidate.acq_value;
      if (key.startsWith("value:")) return candidate.values[key.slice(6)];
      if (key.startsWith("prediction:")) {
        const [, target, statistic] = key.split(":");
        const prediction = candidate.predictions?.[target];
        return statistic === "std" ? prediction?.std : prediction?.mean;
      }
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

  function downloadCandidates() {
    if (!result) return;
    const header = [
      "rank",
      ...result.feature_columns,
      ...targetColumns.flatMap((target) => [`${target}_mean`, `${target}_std`]),
      "acq_value",
      "constraints_ok"
    ];
    const rows = result.candidates.map((candidate) => [
      candidate.rank,
      ...result.feature_columns.map((column) => candidate.values[column]),
      ...targetColumns.flatMap((target) => [
        candidate.predictions?.[target]?.mean,
        candidate.predictions?.[target]?.std
      ]),
      candidate.acq_value,
      candidate.constraints_ok
    ]);
    const csv = `\uFEFF${[header, ...rows].map((row) => row.map(csvCell).join(",")).join("\r\n")}`;
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${result.dataset_name.replace(/\.[^.]+$/, "")}_bo_candidates.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const droppedRows = Number(result.metadata?.dropped_rows ?? 0);
  const bestObservedMap = typeof result.best_observed === "number" ? null : result.best_observed;
  const bestObservedText = typeof result.best_observed === "number"
    ? formatNumber(result.best_observed)
    : targetColumns.map((target) => `${target}: ${formatNumber(bestObservedMap?.[target])}`).join(" / ");
  const directionText = targetColumns
    .map((target) => `${target}: ${(result.directions?.[target] ?? result.direction) === "minimize" ? "最小化" : "最大化"}`)
    .join(" / ");

  return (
    <>
      <SectionHeader
        step="4 · RESULTS"
        title="候補と予測結果を確認する"
        text={`${result.model_type} · 学習 ${result.n_train}件 · best observed ${bestObservedText}`}
        action={
          <>
            <button className="secondary" onClick={() => setStep("optimize")}>設定を変更</button>
            <button className="secondary" onClick={downloadCandidates}>候補CSVを保存</button>
            <button onClick={() => setStep("logs")}>実行ログ</button>
          </>
        }
      />

      <div className="cards metric-grid">
        <MetricCard icon="◎" label="Targets" value={targetColumns.length} detail={targetColumns.join(", ")} />
        <MetricCard icon="↗" label="Directions" value={directionText} />
        <MetricCard icon="◇" label="Features" value={result.n_features} detail={result.feature_columns.join(", ")} />
        <MetricCard icon="▧" label="Candidates" value={result.candidates.length} detail="提案数" tone="success" />
      </div>

      {droppedRows > 0 && (
        <article className="panel compact-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">DATA CLEANING</span>
              <h3>欠損行を除外して学習しました</h3>
              <p>選択した目的変数または説明変数に欠損がある {droppedRows} 行を除外しています。</p>
            </div>
            <span className="status-chip warning">{droppedRows} rows</span>
          </div>
        </article>
      )}

      <article className="panel compact-panel">
        <div className="panel-title"><div><span className="panel-kicker">GRAPH AXES</span><h3>グラフ軸の選択</h3><p>説明変数、各目的の予測平均・標準偏差、獲得値から選択します。</p></div></div>
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
            <p>各目的の予測平均・標準偏差、獲得関数値、目的制約の判定を確認します。</p>
          </div>
          <span className="status-chip success">Ready</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>順位</th>
                {result.feature_columns.map((column) => <th key={column}>{column}</th>)}
                {targetColumns.flatMap((target) => [
                  <th key={`${target}-mean`}>{target}<br />予測平均</th>,
                  <th key={`${target}-std`}>{target}<br />予測標準偏差</th>
                ])}
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
                  {targetColumns.flatMap((target) => [
                    <td key={`${target}-mean`}>{formatNumber(candidate.predictions?.[target]?.mean)}</td>,
                    <td key={`${target}-std`}>{formatNumber(candidate.predictions?.[target]?.std)}</td>
                  ])}
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
