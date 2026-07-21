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

  function clearTargets() {
    [...targetColumns].forEach(toggleTarget);
  }

  return (
    <>
      <SectionHeader
        step="2 · SELECT"
        title="目的変数と説明変数を選択する"
        text="列名をクリックして選択します。選択済みの列はアクセントカラーで表示されます。"
        action={
          <button disabled={!canConfigure} onClick={() => setStep("settings")}>
            設定ページへ
          </button>
        }
      />

      <div className="selection-grid">
        <article className="panel selection-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">TARGET COLUMNS</span>
              <h3>目的変数</h3>
              <p>モデル化、最適化、制約判定に使用する出力列を選択します。</p>
            </div>
            <span className={`status-chip ${targetColumns.length ? "success" : "warning"}`}>
              {targetColumns.length ? `${targetColumns.length} selected` : "Required"}
            </span>
          </div>
          <div className="button-row selection-actions">
            <button className="secondary" disabled={targetColumns.length === 0} onClick={clearTargets}>
              解除
            </button>
          </div>
          <div className="variable-selection-list" role="group" aria-label="目的変数">
            {targetCandidates.map((column) => {
              const selected = targetSet.has(column.name);
              return (
                <button
                  type="button"
                  key={column.name}
                  className={`variable-choice ${selected ? "selected" : ""}`}
                  aria-pressed={selected}
                  onClick={() => toggleTarget(column.name)}
                >
                  {column.name}
                </button>
              );
            })}
          </div>
        </article>

        <article className="panel selection-panel">
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

          <div className="button-row selection-actions">
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

          <div className="variable-selection-list" role="group" aria-label="説明変数">
            {featureCandidates.map((column) => {
              const selected = featureColumns.includes(column.name);
              return (
                <button
                  type="button"
                  key={column.name}
                  className={`variable-choice ${selected ? "selected" : ""}`}
                  aria-pressed={selected}
                  onClick={() => toggleFeature(column.name)}
                >
                  {column.name}
                </button>
              );
            })}
          </div>
        </article>
      </div>
    </>
  );
}
