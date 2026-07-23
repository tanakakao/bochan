import { useEffect, useMemo } from "react";
import { EmptyState, SectionHeader } from "../components/Common";
import FeatureMissingSettings from "../components/FeatureMissingSettings";
import TargetModelSettings from "../components/TargetModelSettings";
import { useWorkbench } from "../context/WorkbenchContext";
import {
  MODEL_DESCRIPTIONS,
  MODEL_FAMILY_OPTIONS,
  MODEL_OPTIONS,
  modelFamilyFor,
  type ModelFamily,
  type WebModelType
} from "../modelOptions";

/** Configures only settings that define the fitted surrogate model. */
export default function SettingsPage() {
  const {
    dataset,
    columns,
    targetColumns,
    targetSettings,
    patchTargetSetting,
    selectedVariables,
    normalize,
    setNormalize,
    inputPerturbation,
    setInputPerturbation,
    nW,
    setNW,
    perturbationStd,
    setPerturbationStd,
    projectionDimensions,
    setProjectionDimensions,
    modelType,
    setModelType,
    fitMaxiter,
    setFitMaxiter,
    settingsValid,
    setStep
  } = useWorkbench();

  if (!dataset || targetColumns.length === 0 || selectedVariables.length === 0) {
    return (
      <>
        <SectionHeader
          step="3 · MODEL"
          title="モデル作成条件を設定する"
          text="先にSelectページで目的変数と説明変数を選択してください。"
        />
        <EmptyState>モデル設定対象の変数が選択されていません。</EmptyState>
      </>
    );
  }

  const preview = dataset.preview;
  const taskTypes = targetColumns.map((target) => targetSettings[target]?.task_type).filter(Boolean);
  const allRegression = taskTypes.length > 0 && taskTypes.every((task) => task === "regression");
  const hasCategoricalFeatures = selectedVariables.some((variable) => variable.type === "categorical");
  const canUseMultitask = targetColumns.length > 1 && allRegression && !hasCategoricalFeatures;
  const projectedModel = modelType === "pca" || modelType === "rembo";
  const maxProjectionDimensions = Math.max(selectedVariables.length, 1);

  const availableModels = useMemo(
    () => MODEL_OPTIONS.filter((option) => option.value !== "multitask" || canUseMultitask),
    [canUseMultitask]
  );
  const modelFamily = modelFamilyFor(modelType);
  const availableModelFamilies = useMemo(
    () => MODEL_FAMILY_OPTIONS.filter((family) => (
      availableModels.some((model) => model.family === family.value)
    )),
    [availableModels]
  );
  const modelOptions = useMemo(
    () => availableModels.filter((option) => option.family === modelFamily),
    [availableModels, modelFamily]
  );

  useEffect(() => {
    if (!availableModels.some((option) => option.value === modelType)) setModelType("base");
  }, [availableModels, modelType, setModelType]);

  useEffect(() => {
    if (projectionDimensions > maxProjectionDimensions) {
      setProjectionDimensions(Math.min(2, maxProjectionDimensions));
    }
  }, [maxProjectionDimensions, projectionDimensions, setProjectionDimensions]);

  function changeModelFamily(nextFamily: ModelFamily) {
    const firstModel = availableModels.find((option) => option.family === nextFamily);
    if (firstModel) setModelType(firstModel.value);
  }

  const selectedModelDescription = MODEL_DESCRIPTIONS[modelType as WebModelType] ?? "";

  return (
    <>
      <SectionHeader
        step="3 · MODEL"
        title="モデル作成条件を設定する"
        text="目的変数のタスク、入力前処理、欠損値処理、学習モデルを設定します。"
        action={
          <button disabled={!settingsValid} onClick={() => setStep("optimize")}>
            候補提案へ
          </button>
        }
      />

      <div className="model-primary-grid">
        <TargetModelSettings
          columns={columns}
          preview={preview}
          targetColumns={targetColumns}
          targetSettings={targetSettings}
          patchTargetSetting={patchTargetSetting}
        />

        <article className="panel model-selection-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">2 · SURROGATE MODEL</span>
              <h3>学習モデル</h3>
              <p>モデルの大分類、種類、学習反復数を設定します。</p>
            </div>
            <span className="status-chip success">{modelType}</span>
          </div>
          <div className="model-settings-grid">
            <label>
              大分類
              <select value={modelFamily} onChange={(event) => changeModelFamily(event.target.value as ModelFamily)}>
                {availableModelFamilies.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            <label>
              モデル種類
              <select value={modelType} onChange={(event) => setModelType(event.target.value)}>
                {modelOptions.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
            {projectedModel && (
              <label>
                射影・潜在次元数
                <input
                  type="number"
                  min={1}
                  max={maxProjectionDimensions}
                  step={1}
                  value={projectionDimensions}
                  onChange={(event) => setProjectionDimensions(Number(event.target.value))}
                />
              </label>
            )}
            <label>
              Fit maxiter
              <input type="number" min={1} step={1} value={fitMaxiter} onChange={(event) => setFitMaxiter(Number(event.target.value))} />
            </label>
          </div>
          <p className="settings-note">
            {selectedModelDescription}
            {modelType === "multitask" ? " 欠損目的値があればWideMultiTask、なければKroneckerを使用します。" : null}
          </p>
        </article>
      </div>

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">3 · INPUT TRANSFORM</span>
            <h3>説明変数の前処理</h3>
            <p>学習モデルへ入力する前の正規化と入力摂動を設定します。</p>
          </div>
        </div>
        <div className="search-transform-grid">
          <section className="transform-card">
            <div className="transform-card-heading">
              <div><span className="panel-kicker">NORMALIZATION</span><h4>正規化</h4></div>
              <label className="switch-field">
                <input type="checkbox" checked={normalize} onChange={(event) => setNormalize(event.target.checked)} />
                <span>使用する</span>
              </label>
            </div>
            <p>候補提案画面で設定する探索boundsを使って入力を正規化します。デフォルトは有効です。</p>
          </section>

          <section className="transform-card">
            <div className="transform-card-heading">
              <div><span className="panel-kicker">INPUT PERTURBATION</span><h4>入力摂動</h4></div>
              <label className="switch-field">
                <input type="checkbox" checked={inputPerturbation} onChange={(event) => setInputPerturbation(event.target.checked)} />
                <span>使用する</span>
              </label>
            </div>
            <p>候補入力のばらつきをサンプリングし、頑健な候補評価へ反映します。デフォルトは無効です。</p>
            {inputPerturbation && (
              <div className="transform-fields">
                <label>
                  摂動サンプル数 n
                  <input type="number" min={1} step={1} value={nW} onChange={(event) => setNW(Number(event.target.value))} />
                </label>
                <label>
                  ばらつき（標準偏差）
                  <input type="number" min={0.000001} step="any" value={perturbationStd} onChange={(event) => setPerturbationStd(Number(event.target.value))} />
                </label>
              </div>
            )}
          </section>
        </div>
      </article>

      <FeatureMissingSettings />

      {!settingsValid && (
        <article className="panel compact-panel validation-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">VALIDATION</span>
              <h3>モデル設定を確認してください</h3>
              <p>タスク、Binaryクラス、Ordinal順序、カテゴリ候補、入力摂動、モデル条件を確認してください。</p>
            </div>
            <span className="status-chip warning">Not ready</span>
          </div>
        </article>
      )}
    </>
  );
}
