import { useEffect, useMemo } from "react";
import { EmptyState, SectionHeader } from "../components/Common";
import CompositionModelSettings from "../components/CompositionModelSettings";
import FeatureMissingSettings from "../components/FeatureMissingSettings";
import NoiseAlphaSettings from "../components/NoiseAlphaSettings";
import TargetModelSettings from "../components/TargetModelSettings";
import { useWorkbench } from "../context/WorkbenchContext";
import {
  MODEL_DESCRIPTIONS,
  MODEL_FAMILY_OPTIONS,
  MODEL_OPTIONS,
  isMultitaskModelType,
  isNonGaussianModelType,
  isProjectedModelType,
  modelFamilyFor,
  type ModelFamily,
  type WebModelType
} from "../modelOptions";
import {
  regressionLikelihoodFor,
  regressionModelVariantFor,
  regressionModelVariantLabel,
  selectRegressionModelType
} from "../regressionLikelihoodOptions";

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
    crossValidation,
    setCrossValidation,
    featureImportance,
    setFeatureImportance,
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
  const hasRegressionTargets = taskTypes.some((task) => task === "regression");
  const hasCategoricalFeatures = selectedVariables.some((variable) => variable.type === "categorical");
  const canUseMultitask = targetColumns.length > 1 && allRegression && !hasCategoricalFeatures;
  const projectedModel = isProjectedModelType(modelType);
  const maxProjectionDimensions = Math.max(selectedVariables.length, 1);

  const availableModels = useMemo(
    () => MODEL_OPTIONS.filter((option) => (
      (!isNonGaussianModelType(option.value) || allRegression) &&
      (!isMultitaskModelType(option.value) || canUseMultitask)
    )),
    [allRegression, canUseMultitask]
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
              <p>目的変数側で選択した応答分布に対して、モデルの大分類と種類を設定します。</p>
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
                  <option key={option.value} value={option.value}>
                    {regressionModelVariantLabel(option.value)}
                  </option>
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
              <input
                type="number"
                min={1}
                step={1}
                value={fitMaxiter}
                onChange={(event) => setFitMaxiter(Number(event.target.value))}
              />
            </label>
          </div>
          <p className="settings-note">
            {selectedModelDescription}
            {isMultitaskModelType(modelType)
              ? " 複数の回帰目的列をwide形式の相関付きモデルで学習します。"
              : null}
          </p>
        </article>
      </div>

      <article className="panel feature-preprocessing-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">3 · INPUT TRANSFORM</span>
            <h3>説明変数の前処理</h3>
            <p>学習モデルへ入力する前の正規化、入力摂動、観測ノイズ下限、欠損値処理を設定します。</p>
          </div>
        </div>
        <div className="search-transform-grid">
          <section className="transform-card">
            <div className="transform-card-heading">
              <div><span className="panel-kicker">NORMALIZATION</span><h4>正規化</h4></div>
              <label className="switch-field">
                <input
                  type="checkbox"
                  checked={normalize}
                  onChange={(event) => setNormalize(event.target.checked)}
                />
                <span>使用する</span>
              </label>
            </div>
            <p>候補提案画面で設定する探索boundsを使って入力を正規化します。デフォルトは有効です。</p>
          </section>

          <section className="transform-card">
            <div className="transform-card-heading">
              <div><span className="panel-kicker">INPUT PERTURBATION</span><h4>入力摂動</h4></div>
              <label className="switch-field">
                <input
                  type="checkbox"
                  checked={inputPerturbation}
                  onChange={(event) => setInputPerturbation(event.target.checked)}
                />
                <span>使用する</span>
              </label>
            </div>
            <p>候補入力のばらつきをサンプリングし、頑健な候補評価へ反映します。デフォルトは無効です。</p>
            {inputPerturbation && (
              <div className="transform-fields">
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

          <NoiseAlphaSettings
            modelType={modelType}
            hasRegressionTargets={hasRegressionTargets}
          />

          <FeatureMissingSettings />
        </div>
      </article>

      <CompositionModelSettings />

      <article className="panel">
        <div className="panel-title">
          <div><span className="panel-kicker">4 · ACCURACY</span><h3>精度評価</h3></div>
        </div>
        <label className="switch-field">
          <input
            type="checkbox"
            checked={crossValidation.enabled}
            onChange={(event) => setCrossValidation({
              ...crossValidation,
              enabled: event.target.checked
            })}
          />
          <span>交差検証でモデル精度を評価する</span>
        </label>
        {crossValidation.enabled && (
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
        )}
        {crossValidation.enabled && (
          <p className="settings-note">
            交差検証ではデータを分割してモデルを複数回学習するため、通常より時間がかかります。最終モデルは交差検証後に全データで別途学習されます。分類ではクラス比率を保つ層化分割を使用します。
          </p>
        )}
      </article>

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">5 · INSPECTION</span>
            <h3>特徴量重要度</h3>
            <p>Permutation Importanceとモデル固有診断の取得内容を選択します。</p>
          </div>
        </div>
        <label className="switch-field">
          <input
            type="checkbox"
            checked={featureImportance.enabled}
            onChange={(event) => setFeatureImportance({
              ...featureImportance,
              enabled: event.target.checked
            })}
          />
          <span>特徴量重要度を計算する</span>
        </label>
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
