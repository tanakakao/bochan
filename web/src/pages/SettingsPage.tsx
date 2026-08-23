import { useEffect, useMemo, useState } from "react";
import { EmptyState, SectionHeader } from "../components/Common";
import CompositionModelSettings from "../components/CompositionModelSettings";
import CrabNetModelSettings from "../components/CrabNetModelSettings";
import {
  FeatureMissingImputationSettings,
  FeatureMissingStrategySettings
} from "../components/FeatureMissingSettings";
import NoiseAlphaSettings from "../components/NoiseAlphaSettings";
import TargetModelSettings from "../components/TargetModelSettings";
import { useWorkbench } from "../context/WorkbenchContext";
import {
  MODEL_DESCRIPTIONS,
  MODEL_FAMILY_OPTIONS,
  MODEL_OPTIONS,
  isMultitaskModelType,
  isCrabNetModelType,
  isNonGaussianModelType,
  isProjectedModelType,
  modelFamilyFor,
  modelSupportsTaskType,
  type ModelFamily,
  type WebModelType
} from "../modelOptions";
import {
  regressionLikelihoodFor,
  regressionModelVariantFor,
  regressionModelVariantLabel,
  selectRegressionModelType
} from "../regressionLikelihoodOptions";
import {
  loadFeatureMissingSettings,
  saveFeatureMissingSettings,
  type FeatureMissingSettings as FeatureMissingSettingsValue
} from "../webRunSettings";

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
    compositionSettings,
    fitMaxiter,
    setFitMaxiter,
    crossValidation,
    setCrossValidation,
    featureImportance,
    setFeatureImportance,
    settingsValid,
    setStep
  } = useWorkbench();
  const [featureMissingSettings, setFeatureMissingSettings] = useState<FeatureMissingSettingsValue>(
    () => loadFeatureMissingSettings()
  );

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
  const taskTypes = useMemo(
    () => targetColumns
      .map((target) => targetSettings[target]?.task_type)
      .filter((task): task is NonNullable<typeof task> => Boolean(task)),
    [targetColumns, targetSettings]
  );
  const allRegression = taskTypes.length > 0 && taskTypes.every((task) => task === "regression");
  const hasRegressionTargets = taskTypes.some((task) => task === "regression");
  const hasCategoricalFeatures = selectedVariables.some((variable) => variable.type === "categorical");
  const canUseMultitask = targetColumns.length > 1 && allRegression && !hasCategoricalFeatures;
  const canUseCrabNet = Boolean(
    targetColumns.length === 1 &&
    allRegression &&
    compositionSettings.enabled &&
    compositionSettings.column &&
    selectedVariables.some((variable) => variable.name === compositionSettings.column) &&
    compositionSettings.elements.length >= 2 &&
    !selectedVariables.some(
      (variable) => variable.type === "categorical" && variable.name !== compositionSettings.column
    )
  );
  const projectedModel = isProjectedModelType(modelType);
  const maxProjectionDimensions = Math.max(selectedVariables.length, 1);

  const availableModels = useMemo(
    () => MODEL_OPTIONS.filter((option) => (
      taskTypes.every((task) => modelSupportsTaskType(option.value, task)) &&
      (!isNonGaussianModelType(option.value) || allRegression) &&
      (!isMultitaskModelType(option.value) || canUseMultitask) &&
      (!isCrabNetModelType(option.value) || canUseCrabNet)
    )),
    [allRegression, canUseCrabNet, canUseMultitask, taskTypes]
  );
  const modelLikelihood = regressionLikelihoodFor(modelType);
  const modelFamily = modelFamilyFor(modelType);
  const likelihoodModels = useMemo(
    () => availableModels.filter(
      (option) => regressionLikelihoodFor(option.value) === modelLikelihood
    ),
    [availableModels, modelLikelihood]
  );
  const availableModelFamilies = useMemo(
    () => MODEL_FAMILY_OPTIONS.filter((family) => (
      likelihoodModels.some((model) => model.family === family.value)
    )),
    [likelihoodModels]
  );
  const modelOptions = useMemo(
    () => likelihoodModels.filter((option) => option.family === modelFamily),
    [likelihoodModels, modelFamily]
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
    const nextModelType = selectRegressionModelType(
      availableModels,
      modelLikelihood,
      regressionModelVariantFor(modelType),
      nextFamily
    );
    if (nextModelType) setModelType(nextModelType);
  }

  function updateFeatureMissingSettings(next: FeatureMissingSettingsValue) {
    setFeatureMissingSettings(next);
    saveFeatureMissingSettings(next);
  }

  const selectedModelDescription = MODEL_DESCRIPTIONS[modelType as WebModelType] ?? "";

  return (
    <>
      <SectionHeader
        step="3 · MODEL"
        title="目的変数とタスクを決める"
        text="目的変数とタスクに応じて、モデル・前処理・評価条件を設定します。"
        action={
          <button disabled={!settingsValid} onClick={() => setStep("optimize")}>
            候補提案へ
          </button>
        }
      />

      <TargetModelSettings
        columns={columns}
        preview={preview}
        targetColumns={targetColumns}
        targetSettings={targetSettings}
        patchTargetSetting={patchTargetSetting}
      />

      <article className="panel model-workbench-card">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">SURROGATE MODEL</span>
            <h3>モデルと基本設定</h3>
          </div>
          <span className="status-chip success">{modelType}</span>
        </div>

        <div className="model-config-grid">
          <section className="model-config-column model-selection-column">
            <div className="config-column-heading">
              <span className="panel-kicker">MODEL SELECTION</span>
              <h4>モデル選択</h4>
              <p>目的変数のタスクと応答分布に対応するモデルを選択します。</p>
            </div>
            <div className="model-config-fields">
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
                    <option key={option.value} value={option.value}>
                      {regressionModelVariantLabel(option.value)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p className="settings-note model-selection-note">
              {selectedModelDescription}
              {isMultitaskModelType(modelType)
                ? " 複数の回帰目的列をwide形式の相関付きモデルで学習します。"
                : null}
            </p>

            <div className="compact-setting-list model-analysis-toggles">
              <label className="compact-setting-row">
                <span>
                  <strong>交差検証（CV）</strong>
                  <small>モデル精度を交差検証で評価</small>
                </span>
                <input
                  type="checkbox"
                  checked={crossValidation.enabled}
                  onChange={(event) => setCrossValidation({
                    ...crossValidation,
                    enabled: event.target.checked
                  })}
                />
              </label>
              <label className="compact-setting-row">
                <span>
                  <strong>特徴量重要度</strong>
                  <small>Permutation Importanceなどを計算</small>
                </span>
                <input
                  type="checkbox"
                  checked={featureImportance.enabled}
                  onChange={(event) => setFeatureImportance({
                    ...featureImportance,
                    enabled: event.target.checked
                  })}
                />
              </label>
            </div>
          </section>

          <section className="model-config-column model-basic-settings">
            <div className="config-column-heading">
              <span className="panel-kicker">BASIC SETTINGS</span>
              <h4>基本設定</h4>
            </div>
            <div className="compact-setting-list">
              <label className="compact-setting-row">
                <span>
                  <strong>正規化</strong>
                  <small>探索boundsを使って入力を正規化</small>
                </span>
                <input
                  type="checkbox"
                  checked={normalize}
                  onChange={(event) => setNormalize(event.target.checked)}
                />
              </label>
              <label className="compact-setting-row">
                <span>
                  <strong>入力摂動</strong>
                  <small>入力ばらつきを考慮した頑健評価</small>
                </span>
                <input
                  type="checkbox"
                  checked={inputPerturbation}
                  onChange={(event) => setInputPerturbation(event.target.checked)}
                />
              </label>
              <FeatureMissingStrategySettings
                settings={featureMissingSettings}
                onChange={updateFeatureMissingSettings}
              />
            </div>
          </section>
        </div>

        <CrabNetModelSettings />

        <details className="model-card-details model-output-details" open={!settingsValid}>
          <summary>詳細設定（学習・頑健化・欠損値・観測ノイズ・評価・診断）</summary>

          <section className="model-advanced-section model-training-section">
            <div className="config-column-heading">
              <span className="panel-kicker">TRAINING</span>
              <h4>学習</h4>
              <p>モデル学習そのものに関わる反復数やモデル固有次元を設定します。</p>
            </div>
            <div className="model-settings-grid">
              <label>
                Fit maxiter
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={fitMaxiter}
                  onChange={(event) => setFitMaxiter(Number(event.target.value))}
                />
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
            </div>
          </section>

          <section className="model-advanced-section model-robustness-section">
            <div className="config-column-heading">
              <span className="panel-kicker">ROBUSTNESS</span>
              <h4>頑健化</h4>
              <p>入力ばらつきを考慮する場合のサンプリング条件を設定します。</p>
            </div>
            {inputPerturbation && (
              <div className="model-settings-grid">
                <label>
                  摂動サンプル数 n
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={nW}
                    onChange={(event) => setNW(Number(event.target.value))}
                  />
                </label>
                <label>
                  ばらつき（標準偏差）
                  <input
                    type="number"
                    min={0.000001}
                    step="any"
                    value={perturbationStd}
                    onChange={(event) => setPerturbationStd(Number(event.target.value))}
                  />
                </label>
              </div>
            )}
          </section>

          <CompositionModelSettings />

          <section className="model-advanced-section model-missing-values-section">
            <div className="config-column-heading">
              <span className="panel-kicker">MISSING VALUES</span>
              <h4>欠損値</h4>
              <p>説明変数を補完する場合の補完手法を設定します。</p>
            </div>
            <FeatureMissingImputationSettings
              settings={featureMissingSettings}
              onChange={updateFeatureMissingSettings}
            />
          </section>

          <section className="model-advanced-section model-observation-noise-section">
            <div className="config-column-heading">
              <span className="panel-kicker">OBSERVATION NOISE</span>
              <h4>観測ノイズ</h4>
              <p>対応する回帰モデルの観測ノイズ下限を設定します。</p>
            </div>
            <NoiseAlphaSettings
              modelType={modelType}
              hasRegressionTargets={hasRegressionTargets}
            />
          </section>

          <article className="panel model-evaluation-panel">
            <div className="panel-title">
              <div>
                <span className="panel-kicker">ACCURACY</span>
                <h3>精度評価</h3>
              </div>
              <span className={`status-chip ${crossValidation.enabled ? "success" : ""}`}>
                {crossValidation.enabled ? "ON" : "OFF"}
              </span>
            </div>
            {crossValidation.enabled && (
              <>
                <div className="model-settings-grid">
                  <label>
                    検証方法
                    <select
                      value={crossValidation.method}
                      onChange={(event) => setCrossValidation({
                        ...crossValidation,
                        method: event.target.value as "kfold" | "loo"
                      })}
                    >
                      <option value="kfold">K-fold</option>
                      <option value="loo">Leave-One-Out</option>
                    </select>
                  </label>
                  {crossValidation.method === "kfold" && (
                    <label>
                      分割数
                      <input
                        type="number"
                        min={2}
                        max={dataset.profile.n_rows}
                        value={crossValidation.nSplits}
                        onChange={(event) => setCrossValidation({
                          ...crossValidation,
                          nSplits: Number(event.target.value)
                        })}
                      />
                    </label>
                  )}
                </div>
                <p className="settings-note">
                  交差検証ではデータを分割してモデルを複数回学習するため、通常より時間がかかります。最終モデルは交差検証後に全データで別途学習されます。分類ではクラス比率を保つ層化分割を使用します。
                </p>
              </>
            )}
          </article>

          <article className="panel model-importance-panel">
            <div className="panel-title">
              <div>
                <span className="panel-kicker">INSPECTION</span>
                <h3>特徴量重要度</h3>
              </div>
              <span className={`status-chip ${featureImportance.enabled ? "success" : ""}`}>
                {featureImportance.enabled ? "ON" : "OFF"}
              </span>
            </div>
            {featureImportance.enabled && (
              <>
                <div className="model-settings-grid">
                  <label>
                    取得内容
                    <select
                      value={featureImportance.diagnosticAuto ? "permutation_and_model" : "permutation"}
                      onChange={(event) => setFeatureImportance({
                        ...featureImportance,
                        diagnosticAuto: event.target.value === "permutation_and_model"
                      })}
                    >
                      <option value="permutation">Permutation Importance（PI）のみ</option>
                      <option value="permutation_and_model">PI＋モデル固有診断</option>
                    </select>
                  </label>
                  <label>
                    評価方法
                    <select
                      value={featureImportance.source}
                      onChange={(event) => setFeatureImportance({
                        ...featureImportance,
                        source: event.target.value as "auto" | "training" | "cross_validation"
                      })}
                    >
                      <option value="auto">自動</option>
                      <option value="cross_validation">交差検証</option>
                      <option value="training">学習データ</option>
                    </select>
                  </label>
                  <label>
                    Permutation反復回数
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={featureImportance.nRepeats}
                      onChange={(event) => setFeatureImportance({
                        ...featureImportance,
                        nRepeats: Number(event.target.value)
                      })}
                    />
                  </label>
                  <label>
                    上位表示数
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={featureImportance.topK}
                      onChange={(event) => setFeatureImportance({
                        ...featureImportance,
                        topK: Number(event.target.value)
                      })}
                    />
                  </label>
                  <label>
                    順位基準
                    <select
                      value={featureImportance.rankBy}
                      onChange={(event) => setFeatureImportance({
                        ...featureImportance,
                        rankBy: event.target.value as "value" | "absolute"
                      })}
                    >
                      <option value="value">value</option>
                      <option value="absolute">absolute</option>
                    </select>
                  </label>
                </div>
                <label className="switch-field">
                  <input
                    type="checkbox"
                    checked={featureImportance.computeNoiseImportance}
                    onChange={(event) => setFeatureImportance({
                      ...featureImportance,
                      computeNoiseImportance: event.target.checked
                    })}
                  />
                  <span>入力依存ノイズの重要度も計算</span>
                </label>
                <label className="switch-field">
                  <input
                    type="checkbox"
                    checked={featureImportance.normalizeImportance}
                    onChange={(event) => setFeatureImportance({
                      ...featureImportance,
                      normalizeImportance: event.target.checked
                    })}
                  />
                  <span>正規化重要度を表示</span>
                </label>
                <label className="switch-field">
                  <input
                    type="checkbox"
                    checked={featureImportance.includeNegative}
                    onChange={(event) => setFeatureImportance({
                      ...featureImportance,
                      includeNegative: event.target.checked
                    })}
                  />
                  <span>負の重要度を表示</span>
                </label>
                <label className="switch-field">
                  <input
                    type="checkbox"
                    checked={featureImportance.showErrorBars}
                    onChange={(event) => setFeatureImportance({
                      ...featureImportance,
                      showErrorBars: event.target.checked
                    })}
                  />
                  <span>エラーバーを表示</span>
                </label>
                {featureImportance.source === "cross_validation" && !crossValidation.enabled && (
                  <div className="alert warning">
                    交差検証の特徴量重要度には、交差検証を有効にしてください。
                  </div>
                )}
                <p className="settings-note">
                  モデル固有診断を選択すると、学習モデルが提供するARD、PCA、マルチタスク相関などを取得します。計算回数は概ね 特徴量数 × 反復回数 × fold数 に比例します。
                </p>
              </>
            )}
          </article>
        </details>
      </article>

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
