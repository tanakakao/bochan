import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

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

  return (
    <>
      <SectionHeader
        step="2 · PREPARE"
        title="目的変数と説明変数を設定する"
        text="単一目的回帰の対象列と、モデルへ入力する数値・カテゴリ特徴量を選択します。"
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
            数値列
            <select value={targetColumn} onChange={(event) => changeTarget(event.target.value)}>
              <option value="">選択してください</option>
              {targetCandidates.map((column) => (
                <option key={column.name} value={column.name}>{column.name}</option>
              ))}
            </select>
          </label>
          <div className="settings-note">
            現在のWeb MVPは単一目的回帰です。目的変数には数値列を選択します。
          </div>
        </aside>

        <section className="panel canvas-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">FEATURES</span>
              <h3>説明変数</h3>
              <p>選択済み {featureColumns.length} / {selectableColumns.filter((column) => column.name !== targetColumn).length}</p>
            </div>
            <span className={`status-chip ${featureColumns.length ? "success" : ""}`}>
              {featureColumns.length ? "Configured" : "Required"}
            </span>
          </div>

          <div className="checklist feature-checklist">
            {selectableColumns
              .filter((column) => column.name !== targetColumn)
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
          <div className={`pipeline-node ${targetColumn ? "enabled" : ""}`}><span>2</span><strong>Target</strong><small>{targetColumn || "未選択"}</small></div>
          <i>›</i>
          <div className={`pipeline-node ${featureColumns.length ? "enabled" : ""}`}><span>3</span><strong>Features</strong><small>{featureColumns.length} columns</small></div>
          <i>›</i>
          <div className="pipeline-node"><span>4</span><strong>GP Model</strong><small>次ページで設定</small></div>
          <i>›</i>
          <div className="pipeline-node accent"><span>5</span><strong>Candidates</strong><small>Bayesian optimization</small></div>
        </div>
      </article>
    </>
  );
}
