import { useEffect, useState } from "react";
import { EmptyState, SectionHeader } from "../components/Common";
import InteractiveResultPlots from "../InteractiveResultPlots";
import { withImportanceFeatureLabels } from "../importanceFeatureLabels";
import { downloadNamedModelArtifact } from "../modelArtifactDownload";
import { useWorkbench } from "../context/WorkbenchContext";
import FeatureImportancePanel from "../FeatureImportancePanel";

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

function defaultModelFilename(datasetName: string | undefined): string {
  const stem = String(datasetName ?? "bochan_model").replace(/\.[^.]+$/, "").trim();
  return `${stem || "bochan_model"}.bochan.pt`;
}

function normalizedModelFilename(value: string, fallback: string): string {
  let filename = value.trim() || fallback;
  filename = filename.replace(/[\\/:*?"<>|\u0000-\u001f]+/g, "_").replace(/[. ]+$/g, "");
  if (filename.toLowerCase().endsWith(".bochan.pt")) return filename;
  if (filename.toLowerCase().endsWith(".pt")) filename = filename.slice(0, -3);
  return `${filename || "bochan_model"}.bochan.pt`;
}

/** Renders recommended candidates first, followed by selectable existing Plotly figures. */
export default function ResultsPage() {
  const { result, setError, setStep } = useWorkbench();
  const suggestedFilename = defaultModelFilename(result?.dataset_name);
  const [modelDownloading, setModelDownloading] = useState(false);
  const [modelFilename, setModelFilename] = useState(suggestedFilename);

  useEffect(() => {
    setModelFilename(suggestedFilename);
  }, [suggestedFilename]);

  if (!result) {
    return (
      <>
        <SectionHeader
          step="5 · RESULTS"
          title="候補と予測結果を確認する"
          text="Suggestページで候補を生成してください。"
        />
        <EmptyState>候補生成結果がありません。</EmptyState>
      </>
    );
  }

  const completedResult = result;
  const importanceResult = withImportanceFeatureLabels(completedResult);
  const staleAfterAppend = Boolean(completedResult.metadata?.stale_after_data_append);
  const targetColumns = completedResult.target_columns?.length
    ? completedResult.target_columns
    : completedResult.target_column
      ? [completedResult.target_column]
      : [];
  const candidates = [...completedResult.candidates].sort((left, right) => left.rank - right.rank);

  function downloadCandidates() {
    const header = [
      "rank",
      ...completedResult.feature_columns,
      ...targetColumns.flatMap((target) => [`${target}_mean`, `${target}_std`]),
      "acq_value",
      "constraints_ok"
    ];
    const rows = candidates.map((candidate) => [
      candidate.rank,
      ...completedResult.feature_columns.map((column) => candidate.values[column]),
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
    anchor.download = `${completedResult.dataset_name.replace(/\.[^.]+$/, "")}_bo_candidates.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function downloadModel() {
    if (staleAfterAppend) {
      setError("実験データ追加前のモデルです。更新データで再学習してから保存してください。");
      return;
    }
    const runId = completedResult.visualization_run_id;
    if (!runId) {
      setError("保存対象の学習済みモデルがありません。候補を再生成してください。");
      return;
    }
    const filename = normalizedModelFilename(modelFilename, suggestedFilename);
    setModelFilename(filename);
    setModelDownloading(true);
    setError(null);
    try {
      await downloadNamedModelArtifact(runId, filename);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setModelDownloading(false);
    }
  }

  const droppedRows = Number(completedResult.metadata?.dropped_rows ?? 0);
  const modelDetails = completedResult.metadata?.model_details as Record<string, unknown> | undefined;
  const effectiveAcquisition = String(
    modelDetails?.effective_acquisition ?? completedResult.metadata?.acquisition ?? "—"
  );
  const backend = String(
    modelDetails?.optimizer_backend ?? completedResult.metadata?.optimizer_backend ?? "TabularBayesianOptimizer"
  );
  const cv = completedResult.metadata?.cross_validation as Record<string, any> | undefined;

  return (
    <>
      <SectionHeader
        step="5 · RESULTS"
        title="推奨候補とモデル結果"
        text={`${completedResult.model_type} · ${effectiveAcquisition} · ${backend} · 学習 ${completedResult.n_train}件`}
        action={
          <>
            <button className="secondary" onClick={() => setStep("settings")}>モデル設定</button>
            <button className="secondary" onClick={() => setStep("optimize")}>候補提案設定</button>
            <button onClick={() => { window.location.hash = "experiment"; }}>実験結果を追加</button>
            <button className="secondary" onClick={downloadCandidates}>候補CSVを保存</button>
            <div className="model-save-control">
              <label>
                保存名
                <input
                  type="text"
                  value={modelFilename}
                  onChange={(event) => setModelFilename(event.target.value)}
                  onBlur={() => setModelFilename(normalizedModelFilename(modelFilename, suggestedFilename))}
                  aria-label="モデル保存名"
                />
              </label>
              <button
                className="secondary"
                disabled={!completedResult.visualization_run_id || modelDownloading || staleAfterAppend}
                onClick={() => void downloadModel()}
                title="モデル、学習データ、設定、候補結果を指定した名前で保存します。"
              >
                {modelDownloading ? "モデル保存中" : "モデルを保存"}
              </button>
            </div>
          </>
        }
      />

      {staleAfterAppend && (
        <div className="alert warning stale-result-note">
          実験データを追加したため、この候補と予測は追加前の最新モデル結果です。グラフは引き続き確認できますが、候補更新とモデル再利用には更新データでの再学習が必要です。
        </div>
      )}

      {Boolean(completedResult.metadata?.model_artifact_loaded) && (
        <div className="alert success artifact-loaded-note">
          保存モデルから復元した結果です。学習済みモデルと設定を保持しているため、可視化の再生成や候補提案設定の変更ができます。
        </div>
      )}

      {cv?.outputs && (
        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">CROSS VALIDATION</span><h3>交差検証による精度評価</h3><p>{String(cv.splitter_name)} · {Number(cv.n_splits)} folds</p></div></div>
          {Object.entries(cv.outputs as Record<string, any>).map(([outputName, output], outputIndex) => (
            <details key={outputName} open={outputIndex === 0}>
              <summary>{outputName}</summary>
              <div className="table-wrap"><table><thead><tr><th>指標</th><th>Train</th><th>Validation</th><th>OOF</th></tr></thead><tbody>
                {Object.keys(output.test_metric_summary ?? {}).map((metric) => <tr key={metric}><td>{metric.toUpperCase()}</td><td>{formatNumber(output.train_metric_summary?.[metric]?.mean)} ± {formatNumber(output.train_metric_summary?.[metric]?.std)}</td><td>{formatNumber(output.test_metric_summary?.[metric]?.mean)} ± {formatNumber(output.test_metric_summary?.[metric]?.std)}</td><td>{formatNumber(output.oof_metrics?.[metric])}</td></tr>)}
              </tbody></table></div>
            </details>
          ))}
          {(cv.warnings ?? []).length > 0 && <div className="alert warning">交差検証の一部指標を計算できませんでした。詳細: {(cv.warnings as string[]).join(" / ")}</div>}
        </article>
      )}

      <article className="panel best-model-panel recommended-first">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">RECOMMENDED CANDIDATES</span>
            <h3>推奨候補</h3>
            <p>順位1を先頭に、各目的の予測値・標準偏差、獲得関数値、制約判定を表示します。</p>
          </div>
          <span className={`status-chip ${staleAfterAppend ? "warning" : "success"}`}>
            {candidates.length} candidates{staleAfterAppend ? " · stale" : ""}
          </span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>順位</th>
                {completedResult.feature_columns.map((column) => <th key={column}>{column}</th>)}
                {targetColumns.flatMap((target) => [
                  <th key={`${target}-mean`}>{target}<br />予測値</th>,
                  <th key={`${target}-std`}>{target}<br />予測標準偏差</th>
                ])}
                <th>獲得値</th>
                <th>条件</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((candidate) => (
                <tr key={candidate.rank} className={candidate.rank === 1 ? "candidate-best" : ""}>
                  <td><span className="rank">{candidate.rank}</span></td>
                  {completedResult.feature_columns.map((column) => (
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
                  <td>
                    <span className={`status-chip ${candidate.constraints_ok ? "success" : "warning"}`}>
                      {candidate.constraints_ok ? "OK" : "NG"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>

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

      <InteractiveResultPlots result={completedResult} />

      <FeatureImportancePanel result={importanceResult} />

      {(completedResult.visualization_warnings ?? []).length > 0 && (
        <div className="alert warning">
          {completedResult.visualization_warnings.map((warning) => <div key={warning}>{warning}</div>)}
        </div>
      )}
    </>
  );
}
