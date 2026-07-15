import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

/** Renders target and feature selection for the workbench. */
export default function PreparePage() {
  const {
    dataset,
    targetCandidates,
    selectableColumns,
    targetColumn,
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
          step="2 · PREPARE"
          title="目的変数と説明変数を設定する"
          text="先にDataページでデータを読み込んでください。"
        />
        <EmptyState>データがありません。</EmptyState>
      </>
    );
  }

  return (
    <>
      <SectionHeader
        step="2 · PREPARE"
        title="目的変数と説明変数を設定する"
        text="目的変数は複数選択できます。説明変数にはモデルへ入力する数値・カテゴリ特徴量を選択します。"
        action={
          <button disabled={!canConfigure} onClick={() => setStep("optimize")}>
            探索設定へ
          </button>
        }
      />

      <div className="workspace-two">
        <aside className="settings-card">
          <div className="settings-title">
            <span>TARGETS</span>
            <h3>目的変数</h3>
          </div>
          <div className="checklist compact-checklist">
            {targetCandidates.map((column) => (
              <label key={column.name}>
                <input
                  type="checkbox"
                  checked={targetColumns.includes(column.name)}
                  onChange={() => toggleTarget(column.name)}
                />
                <span className="feature-name">{column.name}</span>
                <span className="feature-meta">{column.kind} · ユニーク {column.unique_count}</span>
              </label>
            ))}
          </div>
          <div className="settings-note">
            選択中: {targetColumns.length ? targetColumns.join(", ") : "未選択"}
          </div>
        </aside>

        <section className="panel canvas-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">FEATURES</span>
              <h3>説明変数</h3>
              <p>選択済み {featureColumns.length} / {selectableColumns.filter((column) => !targetColumns.includes(column.name)).length}</p>
            </div>
            <span className={`status-chip ${featureColumns.length ? "success" : ""}`}>
              {featureColumns.length ? "Configured" : "Required"}
            </span>
          </div>

          <div className="checklist feature-checklist">
            {selectableColumns
              .filter((column) => !targetColumns.includes(column.name))
              .map((column) => (
                <label key={column.name}>
                  <input
                    type="checkbox"
                    checked={featureColumns.includes(column.name)}
                    onChange={() => toggleFeature(column.name)}
                  />
                  <span className="feature-name">{column.name}</span>
                  <span className="feature-meta">
                    {column.kind} · 欠損 {Math.round(column.missing_rate * 1000) / 10}%
                  </span>
                </label>
              ))}
          </div>
        </section>
      </div>

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">PIPELINE</span>
            <h3>現在の学習フロー</h3>
            <p>データ読込から候補生成までの処理対象を確認します。</p>
          </div>
        </div>
        <div className="pipeline-flow">
          <div className="pipeline-node enabled"><span>1</span><strong>Dataset</strong><small>{dataset.name}</small></div>
          <i>›</i>
          <div className={`pipeline-node ${targetColumn ? "enabled" : ""}`}><span>2</span><strong>Targets</strong><small>{targetColumns.join(", ") || "未選択"}</small></div>
          <i>›</i>
          <div className={`pipeline-node ${featureColumns.length ? "enabled" : ""}`}><span>3</span><strong>Features</strong><small>{featureColumns.length} columns</small></div>
          <i>›</i>
          <div className="pipeline-node"><span>4</span><strong>Model</strong><small>次ページで設定</small></div>
          <i>›</i>
          <div className="pipeline-node accent"><span>5</span><strong>Candidates</strong><small>BO / AL / LSE</small></div>
        </div>
      </article>
    </>
  );
}
