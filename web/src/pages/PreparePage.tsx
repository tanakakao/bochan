import { useEffect, useRef, useState } from "react";
import {
  applyAnalysisConfig,
  applyStoredRunSettings,
  captureStoredRunSettings,
  createGuidedAnalysisConfig,
  createGuidedTargetPatch,
  createGuidedVariablePatch,
  restoreStoredRunSettings,
  type StoredRunSettingsSnapshot
} from "../analysisConfig";
import { EmptyState, SectionHeader } from "../components/Common";
import CompositionKindControl from "../components/CompositionKindControl";
import { useWorkbench } from "../context/WorkbenchContext";
import { getColumnClassValues } from "../targetSettingUtils";
import type { ColumnProfile, Direction } from "../types";
import { useWorkbenchMode } from "../workbenchMode";
import "./PreparePage.css";

function missingValueSummary(column: ColumnProfile): string {
  const percent = column.missing_rate * 100;
  const digits = percent > 0 && percent < 0.1 ? 2 : 1;
  return `欠損 ${column.missing_count}件 (${percent.toFixed(digits)}%)`;
}

function columnIsSelectable(column: ColumnProfile): boolean {
  return column.unique_count > 1;
}

function columnSelectionDisabledReason(column: ColumnProfile): string {
  return column.unique_count === 0 ? "有効な値がありません" : "値が1種類のみ";
}

/** Selects targets/features and defines whether each selected feature is numeric or categorical. */
export default function PreparePage() {
  const mode = useWorkbenchMode();
  const {
    dataset,
    targetCandidates,
    selectableColumns,
    targetColumns,
    targetSettings,
    featureColumns,
    variables,
    patchVariable,
    patchTargetSetting,
    toggleTarget,
    toggleFeature,
    canConfigure,
    candidateSettingsValid,
    setNormalize,
    setInputPerturbation,
    setNW,
    setPerturbationStd,
    setProjectionDimensions,
    setModelType,
    setAcquisitionFamily,
    setAcquisition,
    setBeta,
    setFitMaxiter,
    q,
    setQ,
    setNumRestarts,
    setRawSamples,
    execute,
    setError,
    setStep
  } = useWorkbench();
  const [simpleExecutionPending, setSimpleExecutionPending] = useState(false);
  const simpleExecutionStarted = useRef(false);
  const storedRunSettings = useRef<StoredRunSettingsSnapshot | null>(null);
  const normalizedDatasetId = useRef<string | null>(null);

  useEffect(() => {
    if (!simpleExecutionPending || simpleExecutionStarted.current) return;
    if (!candidateSettingsValid) {
      restoreStoredRunSettings(storedRunSettings.current);
      storedRunSettings.current = null;
      setSimpleExecutionPending(false);
      setError(
        "簡易モードの既定値では実行できません。値が1種類しかない列、探索範囲を作れない列、または分類クラス数を確認してください。"
      );
      return;
    }

    simpleExecutionStarted.current = true;
    setSimpleExecutionPending(false);
    void execute().finally(() => {
      restoreStoredRunSettings(storedRunSettings.current);
      storedRunSettings.current = null;
    });
  }, [candidateSettingsValid, execute, setError, simpleExecutionPending]);

  useEffect(() => {
    const datasetId = dataset?.dataset_id ?? null;
    if (!datasetId || normalizedDatasetId.current === datasetId) return;
    normalizedDatasetId.current = datasetId;
    if (dataset?.source_type === "model_artifact") return;

    targetColumns.forEach((name) => {
      const column = targetCandidates.find((candidate) => candidate.name === name);
      if (column && !columnIsSelectable(column)) toggleTarget(name);
    });
    featureColumns.forEach((name) => {
      const column = selectableColumns.find((candidate) => candidate.name === name);
      if (column && !columnIsSelectable(column)) toggleFeature(name);
    });
  }, [dataset, featureColumns, selectableColumns, targetCandidates, targetColumns, toggleFeature, toggleTarget]);

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
  const selectedTargetMissingCount = targetCandidates.filter(
    (column) => targetSet.has(column.name) && column.missing_count > 0
  ).length;
  const selectedFeatureMissingCount = featureCandidates.filter(
    (column) => featureColumns.includes(column.name) && column.missing_count > 0
  ).length;

  function replaceFeatureSelection(names: string[]) {
    const eligibleNames = new Set(
      names.filter((name) => {
        const column = featureCandidates.find((candidate) => candidate.name === name);
        return Boolean(column && columnIsSelectable(column));
      })
    );
    featureCandidates.forEach((column) => {
      const selected = featureColumns.includes(column.name);
      if (selected !== eligibleNames.has(column.name)) toggleFeature(column.name);
    });
  }

  function clearTargets() {
    [...targetColumns].forEach(toggleTarget);
  }

  function setFeatureCategorical(name: string, categorical: boolean) {
    const column = selectableColumns.find((candidate) => candidate.name === name);
    const variable = variables[name];
    if (!column || !variable || !columnIsSelectable(column)) return;
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

  function changeSimpleCandidateCount(value: string) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return;
    setQ(Math.min(20, Math.max(1, Math.trunc(parsed))));
  }

  function changeSimpleDirection(name: string, direction: Direction) {
    patchTargetSetting(name, { direction });
  }

  function executeSimpleMode() {
    if (!canConfigure || simpleExecutionPending) return;
    setError(null);
    simpleExecutionStarted.current = false;

    const config = createGuidedAnalysisConfig({
      featureCount: featureColumns.length,
      targetCount: targetColumns.length,
      q
    });
    storedRunSettings.current = captureStoredRunSettings();
    applyStoredRunSettings(config.persisted);

    targetColumns.forEach((name) => {
      const column = selectableColumns.find((candidate) => candidate.name === name);
      if (column) {
        patchTargetSetting(
          name,
          createGuidedTargetPatch(column, preview, targetSettings[name]?.direction ?? "maximize")
        );
      }
    });
    featureColumns.forEach((name) => {
      const column = selectableColumns.find((candidate) => candidate.name === name);
      if (column) patchVariable(name, createGuidedVariablePatch(column, preview));
    });

    applyAnalysisConfig(config, {
      setNormalize,
      setInputPerturbation,
      setNW,
      setPerturbationStd,
      setProjectionDimensions,
      setModelType,
      setAcquisitionFamily,
      setAcquisition,
      setBeta,
      setFitMaxiter,
      setQ,
      setNumRestarts,
      setRawSamples
    });
    setSimpleExecutionPending(true);
  }

  return (
    <>
      <SectionHeader
        step="2 · SELECT"
        title={mode === "simple" ? "目的と変更できる条件を選択する" : "変数と説明変数の型を設定する"}
        text={mode === "simple"
          ? "目的変数、最大化・最小化、変更できる条件、提案件数だけ指定します。モデルや探索方法はbochanが自動設定します。"
          : "列名をクリックして選択します。説明変数は同じ枠内で数値／カテゴリ扱いを設定できます。"}
        action={mode === "simple" ? (
          <button
            disabled={!canConfigure || simpleExecutionPending}
            onClick={executeSimpleMode}
          >
            {simpleExecutionPending ? "設定を準備中" : "候補を提案"}
          </button>
        ) : (
          <button disabled={!canConfigure} onClick={() => setStep("settings")}>
            モデル設定へ
          </button>
        )}
      />

      {mode === "simple" && (
        <article className="panel compact-panel simple-mode-summary">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">QUICK SETUP</span>
              <h3>必要な項目だけ設定</h3>
              <p>目的の方向と提案件数を決め、下で目的変数と変更できる条件を選択してください。</p>
            </div>
            <span className="status-chip success">自動設定</span>
          </div>

          <div className="simple-primary-controls">
            <section className="simple-direction-control">
              <span className="simple-control-label">目的の方向</span>
              {targetColumns.length === 0 ? (
                <p className="simple-control-empty">下の「目的変数」から対象を選択してください。</p>
              ) : (
                <div className="simple-direction-list">
                  {targetColumns.map((name) => {
                    const column = targetCandidates.find((candidate) => candidate.name === name);
                    if (column?.kind !== "numeric") {
                      return (
                        <div className="simple-direction-row" key={name}>
                          <strong>{name}</strong>
                          <span className="simple-auto-note">分類対象を自動設定</span>
                        </div>
                      );
                    }
                    const direction = targetSettings[name]?.direction ?? "maximize";
                    return (
                      <div className="simple-direction-row" key={name}>
                        <strong>{name}</strong>
                        <div className="simple-direction-toggle" role="group" aria-label={`${name}の最適化方向`}>
                          <button
                            type="button"
                            className={direction === "maximize" ? "active" : ""}
                            aria-pressed={direction === "maximize"}
                            onClick={() => changeSimpleDirection(name, "maximize")}
                          >
                            最大化
                          </button>
                          <button
                            type="button"
                            className={direction === "minimize" ? "active" : ""}
                            aria-pressed={direction === "minimize"}
                            onClick={() => changeSimpleDirection(name, "minimize")}
                          >
                            最小化
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>

            <label className="simple-candidate-count" data-tutorial="simple-candidate-count">
              <span className="simple-control-label">次に試す条件</span>
              <div className="simple-candidate-input-row">
                <input
                  type="number"
                  min={1}
                  max={20}
                  step={1}
                  value={q}
                  aria-label="簡易モードの提案点数"
                  onChange={(event) => changeSimpleCandidateCount(event.target.value)}
                />
                <span>件</span>
              </div>
              <small>1〜20件</small>
            </label>
          </div>

          <details className="simple-auto-settings">
            <summary>自動設定の内容を見る</summary>
            <p>Base GP、EI（多目的ではEHVI）、入力正規化、BoTorch探索を使用します。欠損行は既定設定で処理し、制約は追加しません。</p>
          </details>
        </article>
      )}

      <div className="selection-grid">
        <article className="panel selection-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">TARGET COLUMNS</span>
              <h3>目的変数</h3>
              <p>{mode === "simple"
                ? "良くしたい値を選択します。グレーの列は値が1種類以下のため選択できません。"
                : "モデル化、候補提案、制約判定に使用する出力列を選択します。グレーの列は値が1種類以下のため選択できません。"}</p>
            </div>
            <span className={`status-chip ${targetColumns.length && selectedTargetMissingCount === 0 ? "success" : "warning"}`}>
              {targetColumns.length === 0
                ? "Required"
                : selectedTargetMissingCount > 0
                  ? `${selectedTargetMissingCount}列に欠損`
                  : `${targetColumns.length} selected`}
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
              const unavailable = !columnIsSelectable(column);
              const selectedWithMissing = selected && column.missing_count > 0;
              const title = unavailable
                ? `${column.name}: ${columnSelectionDisabledReason(column)}`
                : selectedWithMissing
                  ? `${column.name}: ${missingValueSummary(column)}`
                  : undefined;
              return (
                <button
                  type="button"
                  key={column.name}
                  className={`variable-choice target-variable-choice ${selected ? "selected" : ""} ${selectedWithMissing ? "selected-with-missing" : ""} ${unavailable ? "constant-column" : ""}`}
                  aria-pressed={selected}
                  disabled={unavailable && !selected}
                  title={title}
                  onClick={() => toggleTarget(column.name)}
                >
                  <span className="variable-choice-copy">
                    <span>{column.name}</span>
                    {unavailable && (
                      <small className="column-selection-disabled">{columnSelectionDisabledReason(column)}</small>
                    )}
                    {selectedWithMissing && (
                      <small className="missing-value-warning">⚠ {missingValueSummary(column)}</small>
                    )}
                  </span>
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
              <p>{mode === "simple"
                ? "実験で変更できる条件を選択します。数値／カテゴリはデータから自動判定します。グレーの列は選択できません。"
                : "淡い赤は数値、オレンジはカテゴリ扱いです。グレーの列は値が1種類以下のため選択できません。"}</p>
            </div>
            <span className={`status-chip ${featureColumns.length && selectedFeatureMissingCount === 0 ? "success" : "warning"}`}>
              {featureColumns.length === 0
                ? "Required"
                : selectedFeatureMissingCount > 0
                  ? `${selectedFeatureMissingCount}列に欠損`
                  : `${featureColumns.length} selected`}
            </span>
          </div>

          <div className="button-row selection-actions">
            <button
              className="secondary"
              onClick={() => replaceFeatureSelection(
                featureCandidates.filter(columnIsSelectable).map((column) => column.name)
              )}
            >
              全選択
            </button>
            {mode !== "simple" && (
              <button
                className="secondary"
                onClick={() => replaceFeatureSelection(
                  featureCandidates
                    .filter((column) => column.kind === "numeric" && columnIsSelectable(column))
                    .map((column) => column.name)
                )}
              >
                数値列のみ
              </button>
            )}
            <button className="secondary" onClick={() => replaceFeatureSelection([])}>
              解除
            </button>
          </div>

          <div className="variable-selection-list" role="group" aria-label="説明変数">
            {featureCandidates.map((column) => {
              const selected = featureColumns.includes(column.name);
              const variable = variables[column.name];
              const categorical = variable?.type === "categorical" || column.kind === "categorical";
              const unavailable = !columnIsSelectable(column);
              const selectedWithMissing = selected && column.missing_count > 0;
              const title = unavailable
                ? `${column.name}: ${columnSelectionDisabledReason(column)}`
                : selectedWithMissing
                  ? `${column.name}: ${missingValueSummary(column)}`
                  : undefined;

              if (mode === "simple") {
                return (
                  <button
                    type="button"
                    key={column.name}
                    className={`variable-choice simple-feature-choice ${selected ? "selected" : ""} ${selectedWithMissing ? "selected-with-missing" : ""} ${unavailable ? "constant-column" : ""}`}
                    aria-pressed={selected}
                    disabled={unavailable && !selected}
                    title={title}
                    onClick={() => toggleFeature(column.name)}
                  >
                    <span className="variable-choice-copy">
                      <span>{column.name}</span>
                      <small>{categorical ? "カテゴリ" : "数値"}</small>
                      {unavailable && (
                        <small className="column-selection-disabled">{columnSelectionDisabledReason(column)}</small>
                      )}
                      {selectedWithMissing && (
                        <small className="missing-value-warning">⚠ {missingValueSummary(column)}</small>
                      )}
                    </span>
                  </button>
                );
              }

              return (
                <div
                  key={column.name}
                  className={`variable-choice feature-variable-choice ${selected ? "selected" : ""} ${selected && categorical ? "selected-categorical" : ""} ${selectedWithMissing ? "selected-with-missing" : ""} ${unavailable ? "constant-column" : ""}`}
                  aria-disabled={unavailable && !selected}
                  title={title}
                >
                  <button
                    type="button"
                    className="variable-choice-main"
                    aria-pressed={selected}
                    disabled={unavailable && !selected}
                    onClick={() => toggleFeature(column.name)}
                  >
                    <span>{column.name}</span>
                    <small>{categorical ? "categorical" : "numeric"}</small>
                    {unavailable && (
                      <span className="column-selection-disabled">{columnSelectionDisabledReason(column)}</span>
                    )}
                    {selectedWithMissing && (
                      <span className="missing-value-warning">⚠ {missingValueSummary(column)}</span>
                    )}
                  </button>
                  <label className="feature-type-toggle" title={unavailable ? columnSelectionDisabledReason(column) : column.kind === "categorical" ? "入力データ上カテゴリ列のため固定です。" : "カテゴリ変数として扱う"}>
                    <input
                      type="checkbox"
                      checked={categorical}
                      disabled={unavailable || column.kind === "categorical"}
                      onChange={(event) => setFeatureCategorical(column.name, event.target.checked)}
                    />
                    <span>カテゴリ</span>
                  </label>
                  <CompositionKindControl
                    column={column.name}
                    preview={preview}
                    categorical={categorical}
                  />
                </div>
              );
            })}
          </div>
        </article>
      </div>
    </>
  );
}
