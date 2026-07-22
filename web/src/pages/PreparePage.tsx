import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import { getColumnClassValues } from "../targetSettingUtils";

/** Selects targets/features and defines whether each selected feature is numeric or categorical. */
export default function PreparePage() {
  const {
    dataset,
    targetCandidates,
    selectableColumns,
    targetColumns,
    featureColumns,
    variables,
    patchVariable,
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

  const preview = dataset.preview;
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

  function setFeatureCategorical(name: string, categorical: boolean) {
    const column = selectableColumns.find((candidate) => candidate.name === name);
    const variable = variables[name];
    if (!column || !variable) return;
    if (!featureColumns.includes(name)) toggleFeature(name);
    const nextType = categorical || column.kind === "categorical" ? "categorical" : "numeric";
    patchVariable(name, {
      type: nextType,
      fixed: false,
      fixed_value: undefined,
      categories: nextType === "categorical"
        ? getColumnClassValues(column, preview)
        : undefined,
      lower: nextType === "numeric" ? variable.lower ?? column.min ?? undefined : undefined,
      upper: nextType === "numeric" ? variable.upper ?? column.max ?? undefined : undefined,
      step: nextType === "numeric" ? variable.step : undefined
    });
  }

  return (
    <>
      <SectionHeader
        step="2 · SELECT"
        title="変数と説明変数の型を設定する"
        text="列名をクリックして選択します。説明変数は同じ枠内で数値／カテゴリ扱いを設定できます。"
        action={
          <button disabled={!canConfigure} onClick={() => setStep("settings")}>
            モデル設定へ
          </button>
        }
      />

      <div className="selection-grid">
        <article className="panel selection-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">TARGET COLUMNS</span>
              <h3>目的変数</h3>
              <p>モデル化、候補提案、制約判定に使用する出力列を選択します。</p>
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
              <p>青は数値、紫はカテゴリ扱いです。カテゴリ設定を変更すると、その列も選択されます。</p>
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
              const variable = variables[column.name];
              const categorical = variable?.type === "categorical" || column.kind === "categorical";
              return (
                <div
                  key={column.name}
                  className={`variable-choice feature-variable-choice ${selected ? "selected" : ""} ${selected && categorical ? "selected-categorical" : ""}`}
                >
                  <button
                    type="button"
                    className="variable-choice-main"
                    aria-pressed={selected}
                    onClick={() => toggleFeature(column.name)}
                  >
                    <span>{column.name}</span>
                    <small>{categorical ? "categorical" : "numeric"}</small>
                  </button>
                  <label className="feature-type-toggle" title={column.kind === "categorical" ? "入力データ上カテゴリ列のため固定です。" : "カテゴリ変数として扱う"}>
                    <input
                      type="checkbox"
                      checked={categorical}
                      disabled={column.kind === "categorical"}
                      onChange={(event) => setFeatureCategorical(column.name, event.target.checked)}
                    />
                    <span>カテゴリ</span>
                  </label>
                </div>
              );
            })}
          </div>
        </article>
      </div>
    </>
  );
}
