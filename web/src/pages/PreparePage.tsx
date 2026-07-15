import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

/** Renders single-target regression and feature selection for the workbench. */
export default function PreparePage() {
  const {
    dataset,
    targetCandidates,
    selectableColumns,
    targetColumn,
    featureColumns,
    changeTarget,
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

  const featureCandidates = selectableColumns.filter((column) => column.name !== targetColumn);
  const selectedProfiles = dataset.profile.columns.filter(
    (column) => column.name === targetColumn || featureColumns.includes(column.name)
  );
  const missingCount = selectedProfiles.reduce((total, column) => total + column.missing_count, 0);
  const numericFeatureNames = featureCandidates
    .filter((column) => column.kind === "numeric")
    .map((column) => column.name);

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
        step="2 · PREPARE"
        title="目的変数と説明変数を設定する"
        text="現在のWeb APIは単目的回帰に対応しています。数値目的変数を1列、説明変数を1列以上選択してください。"
        action={
          <button disabled={!canConfigure} onClick={() => setStep("optimize")}>
            探索設定へ
          </button>
        }
      />

      <div className="workspace-two">
        <aside className="settings-card">
          <div className="settings-title">
            <span>TARGET</span>
            <h3>目的変数</h3>
          </div>
          <label>
            数値目的変数
            <select value={targetColumn} onChange={(event) => changeTarget(event.target.value)}>
              <option value="">選択してください</option>
              {targetCandidates.map((column) => (
                <option key={column.name} value={column.name}>
                  {column.name}（欠損 {Math.round(column.missing_rate * 1000) / 10}%）
                </option>
              ))}
            </select>
          </label>
          <div className="settings-note">
            選択中: {targetColumn || "未選択"}
          </div>
        </aside>

        <section className="panel canvas-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">FEATURES</span>
              <h3>説明変数</h3>
              <p>選択済み {featureColumns.length} / {featureCandidates.length}</p>
            </div>
            <span className={`status-chip ${featureColumns.length ? "success" : ""}`}>
              {featureColumns.length ? "Configured" : "Required"}
            </span>
          </div>

          <div className="button-row">
            <button className="secondary" onClick={() => replaceFeatureSelection(featureCandidates.map((column) => column.name)}>
              全選択
            </button>
            <button className="secondary" onClick={() => replaceFeatureSelection(numericFeatureNames)}>
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
        </section>
      </div>

      {missingCount > 0 && (
        <article className="panel compact-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">MISSING VALUES</span>
              <h3>選択列に欠損値があります</h3>
              <p>
                現在のWebワークフローでは、目的変数または説明変数に欠損がある行を学習前に除外します。
                結果画面の実行メタデータで除外行数を確認してください。
              </p>
            </div>
            <span className="status-chip warning">{missingCount} cells</span>
          </div>
        </article>
      )}

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
          <div className={`pipeline-node ${targetColumn ? "enabled" : ""}`}><span>2</span><strong>Target</strong><small>{targetColumn || "未選択"}</small></div>
          <i>›</i>
          <div className={`pipeline-node ${featureColumns.length ? "enabled" : ""}`}><span>3</span><strong>Features</strong><small>{featureColumns.length} columns</small></div>
          <i>›</i>
          <div className="pipeline-node"><span>4</span><strong>Model</strong><small>次ページで設定</small></div>
          <i>›</i>
          <div className="pipeline-node accent"><span>5</span><strong>Candidates</strong><small>single-objective BO</small></div>
        </div>
      </article>
    </>
  );
}
