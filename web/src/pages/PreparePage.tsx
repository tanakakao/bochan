import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import type { Direction } from "../types";

/** Renders multi-target regression, target constraints, and feature selection. */
export default function PreparePage() {
  const {
    dataset,
    targetCandidates,
    selectableColumns,
    targetColumns,
    targetDirections,
    featureColumns,
    toggleTarget,
    setTargetDirection,
    toggleFeature,
    outcomeConstraints,
    addOutcomeConstraint,
    patchOutcomeConstraint,
    removeOutcomeConstraint,
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

  const targetSet = new Set(targetColumns);
  const featureCandidates = selectableColumns.filter((column) => !targetSet.has(column.name));
  const selectedProfiles = dataset.profile.columns.filter(
    (column) => targetSet.has(column.name) || featureColumns.includes(column.name)
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
        text="数値目的変数を1列以上選択し、目的ごとの方向と必要な制約を設定してください。"
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
          <p className="settings-note">選択済み {targetColumns.length} / {targetCandidates.length}</p>
          <div className="checklist feature-checklist">
            {targetCandidates.map((column) => {
              const selected = targetSet.has(column.name);
              return (
                <label key={column.name}>
                  <input
                    type="checkbox"
                    checked={selected}
                    onChange={() => toggleTarget(column.name)}
                  />
                  <span className="feature-name">{column.name}</span>
                  <span className="feature-meta">
                    欠損 {Math.round(column.missing_rate * 1000) / 10}% · ユニーク {column.unique_count}
                  </span>
                </label>
              );
            })}
          </div>
        </aside>

        <section className="panel canvas-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">OBJECTIVES</span>
              <h3>目的ごとの設定</h3>
              <p>最適化方向と、候補が満たすべき予測値の制約を設定します。</p>
            </div>
            <span className={`status-chip ${targetColumns.length ? "success" : ""}`}>
              {targetColumns.length ? `${targetColumns.length} targets` : "Required"}
            </span>
          </div>

          {targetColumns.length === 0 ? (
            <EmptyState>左側から目的変数を選択してください。</EmptyState>
          ) : (
            <div className="cards">
              {targetColumns.map((target) => {
                const constraints = outcomeConstraints.filter((constraint) => constraint.target === target);
                return (
                  <article className="panel compact-panel" key={target}>
                    <div className="panel-title">
                      <div>
                        <span className="panel-kicker">TARGET</span>
                        <h3>{target}</h3>
                      </div>
                      <button className="secondary" onClick={() => addOutcomeConstraint(target)}>
                        制約を追加
                      </button>
                    </div>
                    <label>
                      最適化方向
                      <select
                        value={targetDirections[target] ?? "maximize"}
                        onChange={(event) => setTargetDirection(target, event.target.value as Direction)}
                      >
                        <option value="maximize">最大化</option>
                        <option value="minimize">最小化</option>
                      </select>
                    </label>

                    {constraints.length === 0 ? (
                      <p className="settings-note">制約なし。この目的はPareto目的としてのみ使用します。</p>
                    ) : (
                      <div className="form-grid candidate-settings">
                        {constraints.map((constraint) => (
                          <div key={constraint.id} className="settings-note">
                            <strong>{target}</strong>
                            <select
                              value={constraint.operator}
                              onChange={(event) => patchOutcomeConstraint(
                                constraint.id,
                                { operator: event.target.value as "<=" | ">=" }
                              )}
                            >
                              <option value=">=">以上</option>
                              <option value="<=">以下</option>
                            </select>
                            <input
                              type="number"
                              value={constraint.value}
                              onChange={(event) => patchOutcomeConstraint(
                                constraint.id,
                                { value: Number(event.target.value) }
                              )}
                            />
                            <button
                              className="secondary"
                              onClick={() => removeOutcomeConstraint(constraint.id)}
                            >
                              削除
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </div>

      <article className="panel">
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
          <button
            className="secondary"
            onClick={() => replaceFeatureSelection(featureCandidates.map((column) => column.name))}
          >
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
      </article>

      {missingCount > 0 && (
        <article className="panel compact-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">MISSING VALUES</span>
              <h3>選択列に欠損値があります</h3>
              <p>
                目的変数または説明変数に欠損がある行を学習前に除外します。
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
          <div className={`pipeline-node ${targetColumns.length ? "enabled" : ""}`}>
            <span>2</span><strong>Targets</strong><small>{targetColumns.length ? targetColumns.join(", ") : "未選択"}</small>
          </div>
          <i>›</i>
          <div className={`pipeline-node ${featureColumns.length ? "enabled" : ""}`}>
            <span>3</span><strong>Features</strong><small>{featureColumns.length} columns</small>
          </div>
          <i>›</i>
          <div className="pipeline-node"><span>4</span><strong>Model</strong><small>次ページで設定</small></div>
          <i>›</i>
          <div className="pipeline-node accent">
            <span>5</span><strong>Candidates</strong><small>{targetColumns.length > 1 ? "multi-objective BO" : "single-objective BO"}</small>
          </div>
        </div>
      </article>
    </>
  );
}
