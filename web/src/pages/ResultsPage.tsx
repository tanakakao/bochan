import { EmptyState, MetricCard, SectionHeader } from "../components/Common";
import ResultVisualizations from "../ResultVisualizations";
import { useWorkbench } from "../context/WorkbenchContext";

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)
    ? value.toExponential(4)
    : value.toFixed(4).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

export default function ResultsPage() {
  const { result, setStep } = useWorkbench();

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
          <div className="section-actions">
            <button className="secondary" onClick={() => setStep("optimize")}>設定を変更</button>
            <button onClick={() => setStep("logs")}>実行ログ</button>
          </div>
        }
      />

      <div className="cards metric-grid">
        <MetricCard icon="◎" label="Target" value={result.target_column} detail="目的変数" />
        <MetricCard icon="↗" label="Direction" value={result.direction === "maximize" ? "最大化" : "最小化"} />
        <MetricCard icon="◇" label="Features" value={result.n_features} detail={result.feature_columns.join(", ")} />
        <MetricCard icon="▧" label="Candidates" value={result.candidates.length} detail="提案数" tone="success" />
      </div>

      <ResultVisualizations
        visualizations={result.visualizations ?? []}
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
