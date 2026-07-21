import { useEffect, useMemo, useState } from "react";
import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import type { AcquisitionFamily } from "../types";
import {
  loadSearchMethod,
  saveSearchMethod,
  type SearchMethod
} from "../webRunSettings";

type ModelFamily =
  | "standard_gp"
  | "deep_representation"
  | "high_dimensional"
  | "robust_noise"
  | "multitask";

type SearchMethodFamily =
  | "gradient"
  | "metaheuristic"
  | "sampling"
  | "multiobjective";

const MODEL_FAMILY_OPTIONS: Array<{ value: ModelFamily; label: string }> = [
  { value: "standard_gp", label: "標準ガウス過程" },
  { value: "deep_representation", label: "深層・表現学習" },
  { value: "high_dimensional", label: "高次元・次元削減" },
  { value: "robust_noise", label: "ノイズ・頑健" },
  { value: "multitask", label: "マルチタスク" }
];

const MODEL_OPTIONS = [
  { value: "base", label: "Base GP", family: "standard_gp" },
  { value: "deepgp", label: "Deep GP", family: "deep_representation" },
  { value: "deepkernel", label: "Deep Kernel", family: "deep_representation" },
  { value: "saas", label: "SAAS", family: "high_dimensional" },
  { value: "pca", label: "PCA", family: "high_dimensional" },
  { value: "rembo", label: "REMBO", family: "high_dimensional" },
  { value: "robust", label: "Robust (RRP)", family: "robust_noise" },
  { value: "hetero", label: "Heteroskedastic", family: "robust_noise" },
  { value: "multitask", label: "Multitask GP", family: "multitask" }
] as const satisfies ReadonlyArray<{
  value: string;
  label: string;
  family: ModelFamily;
}>;

type WebModelType = (typeof MODEL_OPTIONS)[number]["value"];

const MODEL_DESCRIPTIONS: Record<WebModelType, string> = {
  base: "標準的なガウス過程モデルです。",
  deepgp: "複数層のガウス過程で非線形な表現を学習します。",
  deepkernel: "ニューラルネットワークで特徴表現を学習し、ガウス過程へ接続します。",
  saas: "高次元入力のうち重要な少数次元を疎に選択します。",
  pca: "指定次元へPCA射影してモデル化します。",
  rembo: "指定次元の低次元空間から探索します。",
  robust: "内部ではRRPモデルを使用し、外れ値や頑健性を考慮します。",
  hetero: "入力位置によって異なる観測ノイズをモデル化します。",
  multitask: "回帰目的間の相関を学習して情報共有します。"
};

const FAMILY_OPTIONS: Array<{ value: AcquisitionFamily; label: string }> = [
  { value: "bayesian_optimization", label: "ベイズ最適化" },
  { value: "active_learning", label: "アクティブラーニング" },
  { value: "level_set_estimation", label: "レベルセット推定" }
];

const FAMILY_ACQUISITIONS: Record<AcquisitionFamily, string[]> = {
  bayesian_optimization: ["EI", "PI", "UCB", "EHVI", "NEHVI", "NParEGO"],
  active_learning: ["variance", "predictive_entropy", "BALD", "NIPV"],
  level_set_estimation: ["straddle", "boundary_variance", "ICU"]
};

const SEARCH_METHOD_FAMILY_OPTIONS: Array<{ value: SearchMethodFamily; label: string }> = [
  { value: "gradient", label: "勾配ベース" },
  { value: "metaheuristic", label: "メタヒューリスティクス" },
  { value: "sampling", label: "サンプリング" },
  { value: "multiobjective", label: "多目的専用" }
];

const SEARCH_METHOD_OPTIONS: Array<{
  value: SearchMethod;
  label: string;
  family: SearchMethodFamily;
}> = [
  { value: "normal", label: "通常（BoTorch）", family: "gradient" },
  { value: "torch", label: "Torch", family: "gradient" },
  { value: "ga", label: "GA", family: "metaheuristic" },
  { value: "sa", label: "SA", family: "metaheuristic" },
  { value: "pso", label: "PSO", family: "metaheuristic" },
  { value: "cmaes", label: "CMA-ES", family: "metaheuristic" },
  { value: "thompson_sampling", label: "Thompson sampling", family: "sampling" },
  { value: "nsgaii", label: "NSGA-II", family: "multiobjective" }
];

const SEARCH_FAMILY_DESCRIPTIONS: Record<SearchMethodFamily, string> = {
  gradient: "獲得関数の勾配を利用して連続探索空間を効率的に最適化します。",
  metaheuristic: "勾配を使わず、進化計算や確率的探索で候補を求めます。",
  sampling: "事後分布からサンプリングし、有限候補集合から候補を選択します。",
  multiobjective: "複数目的のベクトル値を直接扱う多目的専用探索です。"
};

function taskLabel(value: string): string {
  if (value === "classification") return "分類";
  if (value === "ordinal") return "順序回帰";
  return "回帰";
}

function familyLabel(value: AcquisitionFamily): string {
  return FAMILY_OPTIONS.find((option) => option.value === value)?.label ?? value;
}

function modelFamilyFor(modelType: string): ModelFamily {
  return MODEL_OPTIONS.find((option) => option.value === modelType)?.family ?? "standard_gp";
}

function searchMethodFamilyFor(searchMethod: SearchMethod): SearchMethodFamily {
  return SEARCH_METHOD_OPTIONS.find((option) => option.value === searchMethod)?.family ?? "gradient";
}

/** Configures the surrogate, acquisition family/function, and search backend. */
export default function OptimizePage() {
  const {
    dataset,
    settingsValid,
    optimizedTargetSettings,
    selectedVariables,
    modelType,
    setModelType,
    projectionDimensions,
    setProjectionDimensions,
    acquisitionFamily,
    setAcquisitionFamily,
    acquisition,
    setAcquisition,
    beta,
    setBeta,
    fitMaxiter,
    setFitMaxiter,
    q,
    setQ,
    numRestarts,
    setNumRestarts,
    rawSamples,
    setRawSamples,
    execute
  } = useWorkbench();
  const [searchMethod, setSearchMethod] = useState<SearchMethod>(() => loadSearchMethod());

  const optimizedCount = optimizedTargetSettings.length;
  const multiObjective = optimizedCount > 1;
  const hasCategoricalFeatures = selectedVariables.some((variable) => variable.type === "categorical");
  const taskTypes = optimizedTargetSettings.map((setting) => setting.task_type);
  const homogeneousTask = taskTypes.length > 0 && taskTypes.every((task) => task === taskTypes[0]);
  const allRegression = taskTypes.length > 0 && taskTypes.every((task) => task === "regression");
  const canUseMultitask = acquisitionFamily === "bayesian_optimization" &&
    multiObjective && allRegression && !hasCategoricalFeatures;
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

  const acquisitionOptions = useMemo(() => {
    if (acquisitionFamily === "bayesian_optimization") {
      return multiObjective
        ? ["EHVI", "NEHVI", "NParEGO"]
        : ["EI", "PI", "UCB"];
    }
    return FAMILY_ACQUISITIONS[acquisitionFamily];
  }, [acquisitionFamily, multiObjective]);

  const searchMethodOptions = useMemo(
    () => SEARCH_METHOD_OPTIONS.filter((option) => (
      option.value !== "nsgaii" || (
        acquisitionFamily === "bayesian_optimization" && multiObjective
      )
    )),
    [acquisitionFamily, multiObjective]
  );
  const searchMethodFamily = searchMethodFamilyFor(searchMethod);
  const availableSearchMethodFamilies = useMemo(
    () => SEARCH_METHOD_FAMILY_OPTIONS.filter((family) => (
      searchMethodOptions.some((method) => method.family === family.value)
    )),
    [searchMethodOptions]
  );
  const searchMethods = useMemo(
    () => searchMethodOptions.filter((option) => option.family === searchMethodFamily),
    [searchMethodFamily, searchMethodOptions]
  );

  useEffect(() => {
    if (!availableModels.some((option) => option.value === modelType)) {
      setModelType("base");
    }
  }, [availableModels, modelType, setModelType]);

  useEffect(() => {
    if (!acquisitionOptions.includes(acquisition)) {
      setAcquisition(acquisitionOptions[0]);
    }
  }, [acquisition, acquisitionOptions, setAcquisition]);

  useEffect(() => {
    if (!searchMethodOptions.some((option) => option.value === searchMethod)) {
      setSearchMethod("normal");
      saveSearchMethod("normal");
    }
  }, [searchMethod, searchMethodOptions]);

  useEffect(() => {
    if (projectionDimensions > maxProjectionDimensions) {
      setProjectionDimensions(Math.min(2, maxProjectionDimensions));
    }
  }, [maxProjectionDimensions, projectionDimensions, setProjectionDimensions]);

  function changeModelFamily(nextFamily: ModelFamily) {
    const firstModel = availableModels.find((option) => option.family === nextFamily);
    if (firstModel) setModelType(firstModel.value);
  }

  function changeFamily(nextFamily: AcquisitionFamily) {
    setAcquisitionFamily(nextFamily);
    if (nextFamily === "bayesian_optimization") {
      setAcquisition(multiObjective ? "EHVI" : "EI");
    } else if (nextFamily === "active_learning") {
      setAcquisition("variance");
    } else {
      setAcquisition("straddle");
    }
  }

  function changeSearchMethod(nextMethod: SearchMethod) {
    setSearchMethod(nextMethod);
    saveSearchMethod(nextMethod);
  }

  function changeSearchMethodFamily(nextFamily: SearchMethodFamily) {
    const firstMethod = searchMethodOptions.find((option) => option.family === nextFamily);
    if (firstMethod) changeSearchMethod(firstMethod.value);
  }

  const validationErrors = useMemo(() => {
    const errors: string[] = [];
    if (!settingsValid) errors.push("Settingsページの目的変数または探索変数設定を確認してください。");
    if (optimizedCount === 0) errors.push("最適化対象の目的変数を1つ以上選択してください。");
    if (modelType === "multitask" && !canUseMultitask) {
      errors.push("multitaskはベイズ最適化で、複数の回帰目的かつ説明変数がすべて数値の場合に選択できます。");
    }
    if (projectedModel && (
      !Number.isInteger(projectionDimensions) ||
      projectionDimensions < 1 ||
      projectionDimensions > maxProjectionDimensions
    )) {
      errors.push(`射影次元は1〜${maxProjectionDimensions}にしてください。`);
    }
    if (
      acquisitionFamily === "level_set_estimation" &&
      optimizedTargetSettings.some((setting) => setting.goal === "none")
    ) {
      errors.push("レベルセット推定では、最適化対象ごとに以上・以下・目標値のいずれかを設定してください。");
    }
    if (searchMethod === "nsgaii" && (!multiObjective || acquisitionFamily !== "bayesian_optimization")) {
      errors.push("NSGA-IIは多目的ベイズ最適化の場合にのみ選択できます。");
    }
    if (fitMaxiter < 1) errors.push("fit maxiterは1以上にしてください。");
    if (q < 1 || q > 20) errors.push("候補点数qは1〜20にしてください。");
    if (numRestarts < 1) errors.push("num_restartsは1以上にしてください。");
    if (rawSamples < 1) errors.push("raw_samplesは1以上にしてください。");
    return errors;
  }, [
    acquisitionFamily,
    canUseMultitask,
    fitMaxiter,
    maxProjectionDimensions,
    modelType,
    multiObjective,
    numRestarts,
    optimizedCount,
    optimizedTargetSettings,
    projectedModel,
    projectionDimensions,
    q,
    rawSamples,
    searchMethod,
    settingsValid
  ]);

  const canExecute = validationErrors.length === 0;
  const modeLabel = multiObjective ? "多目的" : "単目的";
  const taskSummary = homogeneousTask ? taskLabel(taskTypes[0] ?? "regression") : "混合タスク";
  const searchMethodLabel = searchMethodOptions.find((option) => option.value === searchMethod)?.label ?? searchMethod;
  const selectedModelDescription = MODEL_DESCRIPTIONS[modelType as WebModelType] ?? "";

  if (!dataset || !settingsValid) {
    return (
      <>
        <SectionHeader
          step="4 · OPTIMIZE"
          title="モデルと候補生成を設定する"
          text="先にSettingsページで目的変数と探索変数を設定してください。"
        />
        <EmptyState>モデル学習に必要な設定が完了していません。</EmptyState>
      </>
    );
  }

  return (
    <>
      <SectionHeader
        step="4 · OPTIMIZE"
        title="モデルと候補生成を設定する"
        text="モデル、獲得関数、探索手法、候補点数と計算量を設定します。"
        action={<button disabled={!canExecute} onClick={() => void execute()}>候補を生成</button>}
      />

      <div className="form-grid optimize-grid">
        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">SURROGATE</span><h3>モデル</h3></div></div>
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
          <p className="settings-note">
            {selectedModelDescription}
            {!homogeneousTask ? " 混合タスクでは目的変数ごとのサブモデルをhybrid wrapperに束ねます。" : null}
          </p>
          <label>Fit maxiter<input type="number" min={1} value={fitMaxiter} onChange={(event) => setFitMaxiter(Number(event.target.value))} /></label>
        </article>

        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">ACQUISITION</span><h3>獲得関数</h3></div></div>
          <label>
            大分類
            <select value={acquisitionFamily} onChange={(event) => changeFamily(event.target.value as AcquisitionFamily)}>
              {FAMILY_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
            </select>
          </label>
          <label>
            獲得関数
            <select value={acquisition} onChange={(event) => setAcquisition(event.target.value)} disabled={searchMethod === "nsgaii"}>
              {acquisitionOptions.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          {acquisition.toUpperCase().includes("UCB") && searchMethod !== "nsgaii" && (
            <label>Beta<input type="number" min={0} step="0.1" value={beta} onChange={(event) => setBeta(Number(event.target.value))} /></label>
          )}
          <p className="settings-note">
            {searchMethod === "nsgaii" && "NSGA-II選択時は、内部的にNSGA-II用のベクトル獲得戦略へ切り替えます。"}
            {searchMethod !== "nsgaii" && acquisitionFamily === "bayesian_optimization" && "目的値の改善を狙って候補を選びます。"}
            {acquisitionFamily === "active_learning" && "予測不確実性を減らすために情報量の高い候補を選びます。"}
            {acquisitionFamily === "level_set_estimation" && "設定した境界や目標付近を重点的に探索します。"}
          </p>
        </article>

        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">SEARCH METHOD</span><h3>探索手法</h3></div></div>
          <label>
            大分類
            <select value={searchMethodFamily} onChange={(event) => changeSearchMethodFamily(event.target.value as SearchMethodFamily)}>
              {availableSearchMethodFamilies.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
          <label>
            最適化手法
            <select value={searchMethod} onChange={(event) => changeSearchMethod(event.target.value as SearchMethod)}>
              {searchMethods.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
          <p className="settings-note search-method-note">
            {SEARCH_FAMILY_DESCRIPTIONS[searchMethodFamily]}
          </p>
        </article>

        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">CANDIDATES</span><h3>候補生成</h3></div></div>
          <label>q<input type="number" min={1} max={20} value={q} onChange={(event) => setQ(Number(event.target.value))} /></label>
          <label>num_restarts<input type="number" min={1} value={numRestarts} onChange={(event) => setNumRestarts(Number(event.target.value))} /></label>
          <label>raw_samples<input type="number" min={1} value={rawSamples} onChange={(event) => setRawSamples(Number(event.target.value))} /></label>
        </article>
      </div>

      <article className="panel compact-panel validation-panel">
        <div className="panel-title">
          <div><span className="panel-kicker">VALIDATION</span><h3>実行前チェック</h3><p>モデル、目的変数、獲得関数、探索手法を確認します。</p></div>
          <span className={`status-chip ${canExecute ? "success" : "warning"}`}>{canExecute ? "Ready" : `${validationErrors.length} issues`}</span>
        </div>
        {canExecute ? <p className="settings-note">設定に矛盾は見つかりませんでした。</p> : <ul>{validationErrors.map((message) => <li key={message}>{message}</li>)}</ul>}
        <div className="train-launcher">
          <div>
            <strong>{modelType}{projectedModel ? `(${projectionDimensions}D)` : ""} × {familyLabel(acquisitionFamily)} × {searchMethod === "nsgaii" ? "NSGA-II" : acquisition}</strong>
            <span>{modeLabel} · {taskSummary} · {searchMethodLabel} · 最適化対象 {optimizedCount}件 · q={q}</span>
          </div>
          <button disabled={!canExecute} onClick={() => void execute()}>学習して候補を生成</button>
        </div>
      </article>
    </>
  );
}
