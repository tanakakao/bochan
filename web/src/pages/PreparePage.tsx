import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

/** Selects target and feature columns without exposing optimization settings. */
export default function PreparePage() {
  const {
    dataset,
    targetCandidates,
    selectableColumns,
    targetColumns,
    featureColumns,
    toggleTarget,
    toggleFeature,
    canConfigure,
    setStep
  } = useWorkbench();

  if (!dataset) {
    return (
      <>
        <SectionHeader
          step="2 · SELECT"
          title="目的変数と説明変数を選択する"
          text="先にDataページでデータを読み込んでください。"
        />
        <EmptyState>データがありません。</EmptyState>
      </>
    );
  }

  const targetSet = new Set(targetColumns);
  const featureCandidates = selectableColumns.filter((column) => !targetSet.has(column.name));

  function replaceFeatureSelection(names: string[]) {
    const desired = new Set(names);
    featureCandidates.forEach((column) => {
      const selected = featureColumns.includes(column.name);
      if (selected !== desired.has(column.name)) toggleFeature(column.name);
    });
  }

  return (
    <>
      <SectionHeader
        step="2 · SELECT"
        title="目的変数と説明変数を選択する"
        text="この画面では使用する列だけを選択します。タスク種別、目標、探索範囲は次のSettingsページで設定します。"
        action={
          <button disabled={!canConfigure} onClick={() => setStep("settings")}>
            設定ページへ
          </button>
        }
      />

      <div className="workspace-two">
        <article className="panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">TARGET COLUMNS</span>
              <h3>目的変数</h3>
              <p>最適化・判定に使用する出力列を1列以上選択します。</p>
            </div>
            <span className={`status-chip ${targetColumns.length ? "success" : "warning"}`}>
              {targetColumns.length ? `${targetColumns.length} selected` : "Required"}
            </span>
          </div>
          <div className="checklist feature-checklist">
            {targetCandidates.map((column) => (
              <label key={column.name}>
                <input
                  type="checkbox"
                  checked={targetSet.has(column.name)}
                  onChange={() => toggleTarget(column.name)}
                />
                <span className="feature-name">{column.name}</span>
                <span className="feature-meta">
                  {column.kind} · 欠損 {Math.round(column.missing_rate * 1000) / 10}% · ユニーク {column.unique_count}
                </span>
              </label>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">FEATURE COLUMNS</span>
              <h3>説明変数</h3>
              <p>モデル入力と候補探索に使用する列を選択します。</p>
            </div>
            <span className={`status-chip ${featureColumns.length ? "success" : "warning"}`}>
              {featureColumns.length ? `${featureColumns.length} selected` : "Required"}
            </span>
          </div>

          <div className="button-row">
            <button
              className="secondary"
              onClick={() => replaceFeatureSelection(featureCandidates.map((column) => column.name))}
            >
              全選択
            </button>
            <button
              className="secondary"
              onClick={() => replaceFeatureSelection(
                featureCandidates.filter((column) => column.kind === "numeric").map((column) => column.name)
              )}
            >
              数値列のみ
            </button>
            <button className="secondary" onClick={() => replaceFeatureSelection([])}>
              解除
            </button>
          </div>

          <div className="checklist feature-checklist">
            {featureCandidates.map((column) => (
              <label key={column.name}>
                <input
                  type="checkbox"
                  checked={featureColumns.includes(column.name)}
                  onChange={() => toggleFeature(column.name)}
                />
                <span className="feature-name">{column.name}</span>
                <span className="feature-meta">
                  {column.kind} · 欠損 {Math.round(column.missing_rate * 1000) / 10}% · ユニーク {column.unique_count}
                </span>
              </label>
            ))}
          </div>
        </article>
      </div>
    </>
  );
}
