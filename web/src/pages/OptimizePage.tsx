import { useEffect, useMemo, useState } from "react";
import { EmptyState, SectionHeader } from "../components/Common";
import FeatureConstraints from "../components/FeatureConstraints";
import InputPerturbationRiskSettingsControl from "../components/InputPerturbationRiskSettings";
import SearchVariableSettings from "../components/SearchVariableSettings";
import TargetProposalSettings from "../components/TargetProposalSettings";
import { useWorkbench } from "../context/WorkbenchContext";
import type { AcquisitionFamily } from "../types";
import {
  loadSearchMethod,
  saveSearchMethod,
  type SearchMethod
} from "../webRunSettings";

type SearchMethodFamily =
  | "gradient"
  | "metaheuristic"
  | "sampling"
  | "multiobjective";

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

function searchMethodFamilyFor(searchMethod: SearchMethod): SearchMethodFamily {
  return SEARCH_METHOD_OPTIONS.find((option) => option.value === searchMethod)?.family ?? "gradient";
}

function compactAcquisitionName(value: string): string {
  return value.replace(/[_\-\s]/g, "").toLowerCase();
}

function levelSetParameter(name: string): {
  label: string;
  min: number;
  defaultValue: number;
  help: string;
} {
  const key = compactAcquisitionName(name);
  if (key === "boundaryvariance") {
    return {
      label: "境界幅 τ",
      min: 1e-12,
      defaultValue: 1,
      help: "小さいほど境界しきい値の近傍を強く重視します。"
    };
  }
  if (key === "icu") {
    return {
      label: "Bandwidth",
      min: 0,
      defaultValue: 0,
      help: "0では予測標準偏差を自動的に帯域幅として使用します。"
    };
  }
  return {
    label: "Straddle β",
    min: 0,
    defaultValue: 1.96,
    help: "大きいほど不確実性を強く評価し、境界近傍から広めに探索します。"
  };
}

/** Configures objectives, search space, constraints, acquisition, and candidate search. */
export default function OptimizePage() {
  const {
    dataset,
    columns,
    settingsValid,
    candidateSettingsValid,
    modelReuseAvailable,
    targetColumns,
    targetSettings,
    patchTargetSetting,
    optimizedTargetSettings,
    selectedVariables,
    patchVariable,
    modelType,
    projectionDimensions,
    inputPerturbation,
    acquisitionFamily,
    setAcquisitionFamily,
    acquisition,
    setAcquisition,
    beta,
    setBeta,
    fitMaxiter,
    q,
    setQ,
    sequential,
    setSequential,
    minimumCandidateDistanceRatio,
    setMinimumCandidateDistanceRatio,
    numRestarts,
    setNumRestarts,
    rawSamples,
    setRawSamples,
    execute,
    numberOrUndefined
  } = useWorkbench();
  const [searchMethod, setSearchMethod] = useState<SearchMethod>(() => loadSearchMethod());

  const optimizedCount = optimizedTargetSettings.length;
  const multiObjective = optimizedCount > 1;
  const taskTypes = optimizedTargetSettings.map((setting) => setting.task_type);
  const homogeneousTask = taskTypes.length > 0 && taskTypes.every((task) => task === taskTypes[0]);
  const projectedModel = modelType === "pca" || modelType === "rembo";
  const regressionLocalUncertaintyEquivalent = (
    acquisitionFamily === "active_learning"
    && homogeneousTask
    && taskTypes[0] === "regression"
    && ["variance", "predictive_entropy", "bald"].includes(acquisition.toLowerCase())
  );
  const sequentialForced = q > 1 && (
    selectedVariables.some((variable) => variable.type === "categorical")
    || searchMethod === "cmaes"
    || (acquisitionFamily === "level_set_estimation" && inputPerturbation)
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
  const lseParameter = levelSetParameter(acquisition);
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
    if (!acquisitionOptions.includes(acquisition)) setAcquisition(acquisitionOptions[0]);
  }, [acquisition, acquisitionOptions, setAcquisition]);

  useEffect(() => {
    if (!searchMethodOptions.some((option) => option.value === searchMethod)) {
      setSearchMethod("normal");
      saveSearchMethod("normal");
    }
  }, [searchMethod, searchMethodOptions]);

  function changeFamily(nextFamily: AcquisitionFamily) {
    setAcquisitionFamily(nextFamily);
    if (nextFamily === "bayesian_optimization") {
      setAcquisition(multiObjective ? "EHVI" : "EI");
    } else if (nextFamily === "active_learning") {
      setAcquisition("variance");
    } else {
      setAcquisition("straddle");
      setBeta(levelSetParameter("straddle").defaultValue);
    }
  }

  function changeAcquisition(nextAcquisition: string) {
    setAcquisition(nextAcquisition);
    if (acquisitionFamily === "level_set_estimation") {
      setBeta(levelSetParameter(nextAcquisition).defaultValue);
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
    if (!settingsValid) errors.push("モデル設定画面のタスク、前処理、欠損処理、モデル設定を確認してください。");
    if (!candidateSettingsValid) errors.push("目的変数の候補条件または説明変数の探索範囲を確認してください。");
    if (optimizedCount === 0) errors.push("最適化対象の目的変数を1つ以上選択してください。");
    if (modelType === "multitask" && acquisitionFamily !== "bayesian_optimization") {
      errors.push("Multitask GPは現在ベイズ最適化で使用してください。");
    }
    if (
      acquisitionFamily === "level_set_estimation" &&
      optimizedTargetSettings.some((setting) => setting.goal === "none")
    ) {
      errors.push("レベルセット推定では、最適化対象ごとに以上・以下・目標値のいずれかを設定してください。");
    }
    if (acquisitionFamily === "level_set_estimation") {
      const key = compactAcquisitionName(acquisition);
      if (!Number.isFinite(beta) || beta < 0) {
        errors.push("LSEの獲得関数パラメータは0以上の有限値にしてください。");
      } else if (key === "boundaryvariance" && beta <= 0) {
        errors.push("Boundary Varianceのτは0より大きくしてください。");
      }
      const levelSetWeights = optimizedTargetSettings.map(
        (setting) => Number(setting.level_set_weight ?? 1)
      );
      if (levelSetWeights.some((weight) => !Number.isFinite(weight) || weight < 0)) {
        errors.push("LSEの境界重みは0以上の有限値にしてください。");
      } else if (levelSetWeights.reduce((sum, weight) => sum + weight, 0) <= 0) {
        errors.push("LSEの境界重みは少なくとも1つを0より大きくしてください。");
      }
    }
    if (searchMethod === "nsgaii" && (!multiObjective || acquisitionFamily !== "bayesian_optimization")) {
      errors.push("NSGA-IIは多目的ベイズ最適化の場合にのみ選択できます。");
    }
    if (q < 1 || q > 20) errors.push("候補点数qは1〜20にしてください。");
    if (numRestarts < 1) errors.push("num_restartsは1以上にしてください。");
    if (rawSamples < 1) errors.push("raw_samplesは1以上にしてください。");
    if (
      !Number.isFinite(minimumCandidateDistanceRatio)
      || minimumCandidateDistanceRatio < 0
      || minimumCandidateDistanceRatio > 1
    ) {
      errors.push("最小候補間距離は探索範囲比0〜100%で指定してください。");
    }
    return errors;
  }, [
    acquisition,
    acquisitionFamily,
    beta,
    candidateSettingsValid,
    minimumCandidateDistanceRatio,
    modelType,
    multiObjective,
    numRestarts,
    optimizedCount,
    optimizedTargetSettings,
    q,
    rawSamples,
    searchMethod,
    settingsValid
  ]);

  const canExecute = validationErrors.length === 0;
  const modeLabel = multiObjective ? "多目的" : "単目的";
  const taskSummary = homogeneousTask ? taskLabel(taskTypes[0] ?? "regression") : "混合タスク";
  const searchMethodLabel = searchMethodOptions.find((option) => option.value === searchMethod)?.label ?? searchMethod;

  function executionButtons() {
    if (!modelReuseAvailable) {
      return (
        <button
          disabled={!canExecute}
          onClick={() => void execute("retrain")}
          title="現在の設定でモデルを学習してから候補を生成します。"
        >
          モデルを学習して候補を生成
        </button>
      );
    }
    return (
      <div className="model-reuse-actions">
        <button
          className="secondary"
          disabled={!canExecute}
          onClick={() => void execute("retrain")}
          title="現在の設定でモデルを学習し直してから候補を生成します。"
        >
          再学習
        </button>
        <button
          disabled={!canExecute}
          onClick={() => void execute("reuse")}
          title="モデルの再学習を省略し、現在の候補提案条件で候補だけを生成します。"
        >
          学習済みモデルを使用
        </button>
      </div>
    );
  }

  if (!dataset || !settingsValid) {
    return (
      <>
        <SectionHeader
          step="4 · SUGGEST"
          title="候補提案条件を設定する"
          text="先にモデル設定画面でタスク、前処理、欠損処理、モデルを設定してください。"
        />
        <EmptyState>モデル作成に必要な設定が完了していません。</EmptyState>
      </>
    );
  }

  return (
    <>
      <SectionHeader
        step="4 · SUGGEST"
        title="候補提案条件を設定する"
        text="目的、獲得関数、探索手法、候補数、探索範囲、制約を設定します。"
        action={executionButtons()}
      />

      <TargetProposalSettings
        columns={columns}
        preview={dataset.preview}
        targetColumns={targetColumns}
        targetSettings={targetSettings}
        patchTargetSetting={patchTargetSetting}
        numberOrUndefined={numberOrUndefined}
      />

      <div className="form-grid optimize-grid suggestion-method-grid">
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
            <select value={acquisition} onChange={(event) => changeAcquisition(event.target.value)} disabled={searchMethod === "nsgaii"}>
              {acquisitionOptions.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          {acquisition.toUpperCase().includes("UCB") && searchMethod !== "nsgaii" && (
            <label>Beta<input type="number" min={0} step="any" value={beta} onChange={(event) => setBeta(Number(event.target.value))} /></label>
          )}
          {acquisitionFamily === "level_set_estimation" && (
            <>
              <label>
                {lseParameter.label}
                <input
                  type="number"
                  min={lseParameter.min}
                  step="any"
                  value={beta}
                  onChange={(event) => setBeta(Number(event.target.value))}
                />
              </label>
              <small className="settings-note">{lseParameter.help}</small>
            </>
          )}
          <p className="settings-note">
            {searchMethod === "nsgaii" && "NSGA-II選択時は、内部的にNSGA-II用のベクトル獲得戦略へ切り替えます。"}
            {searchMethod !== "nsgaii" && acquisitionFamily === "bayesian_optimization" && "目的値の改善を狙って候補を選びます。"}
            {acquisitionFamily === "active_learning" && "予測不確実性を減らすために情報量の高い候補を選びます。"}
            {acquisitionFamily === "level_set_estimation" && "設定した境界や目標付近を重点的に探索します。"}
          </p>
          {regressionLocalUncertaintyEquivalent && (
            <p className="settings-note">
              標準の等分散Gaussian回帰では、Variance・Predictive Entropy・BALDはposterior varianceの単調変換になるため、
              同じ候補順位になるのが正常です。異なる観点で実験点を選びたい場合は、領域全体の不確実性低減を評価するNIPVを使用してください。
            </p>
          )}
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
          <p className="settings-note search-method-note">{SEARCH_FAMILY_DESCRIPTIONS[searchMethodFamily]}</p>
        </article>

        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">CANDIDATES</span><h3>候補生成</h3></div></div>
          <label>q<input type="number" min={1} max={20} step={1} value={q} onChange={(event) => setQ(Number(event.target.value))} /></label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={q > 1 && (sequential || sequentialForced)}
              disabled={q <= 1 || sequentialForced}
              onChange={(event) => setSequential(event.target.checked)}
            />
            逐次候補生成
          </label>
          <p className="settings-note">
            q &gt; 1で有効にすると、選択済み候補をpendingとして次候補を順番に探索します。
            カテゴリ変数、CMA-ES、LSE + 入力摂動では自動的に有効になります。
          </p>
          <label>
            最小候補間距離（探索範囲比 %）
            <input
              type="number"
              min={0}
              max={100}
              step={0.01}
              value={minimumCandidateDistanceRatio * 100}
              onChange={(event) => setMinimumCandidateDistanceRatio(Number(event.target.value) / 100)}
            />
          </label>
          <p className="settings-note">
            連続変数は探索範囲比、step指定変数は実験分解能、カテゴリ変数はカテゴリ一致で重複を判定します。
          </p>
          <label>num_restarts<input type="number" min={1} step={1} value={numRestarts} onChange={(event) => setNumRestarts(Number(event.target.value))} /></label>
          <label>raw_samples<input type="number" min={1} step={1} value={rawSamples} onChange={(event) => setRawSamples(Number(event.target.value))} /></label>
        </article>
      </div>

      {inputPerturbation && (
        <article className="panel compact-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">INPUT PERTURBATION RISK</span>
              <h3>入力摂動の候補評価</h3>
              <p>探索範囲から生成した候補を、入力ばらつきに対してどのように集約して評価するかを設定します。</p>
            </div>
          </div>
          <div className="transform-fields">
            <InputPerturbationRiskSettingsControl
              acquisitionFamily={acquisitionFamily}
              disabled={acquisitionFamily === "active_learning"}
            />
          </div>
        </article>
      )}

      <SearchVariableSettings
        columns={columns}
        preview={dataset.preview}
        variables={selectedVariables}
        patchVariable={patchVariable}
        numberOrUndefined={numberOrUndefined}
      />

      <FeatureConstraints variables={selectedVariables} />

      <article className="panel compact-panel validation-panel">
        <div className="panel-title">
          <div><span className="panel-kicker">VALIDATION</span><h3>候補提案前チェック</h3><p>目的、獲得関数、探索手法、探索範囲、制約を確認します。</p></div>
          <span className={`status-chip ${canExecute ? "success" : "warning"}`}>{canExecute ? "Ready" : `${validationErrors.length} issues`}</span>
        </div>
        {canExecute ? <p className="settings-note">候補提案条件に矛盾は見つかりませんでした。</p> : <ul>{validationErrors.map((message) => <li key={message}>{message}</li>)}</ul>}
        <div className="train-launcher">
          <div>
            <strong>{modelType}{projectedModel ? `(${projectionDimensions}D)` : ""} × {familyLabel(acquisitionFamily)} × {searchMethod === "nsgaii" ? "NSGA-II" : acquisition}</strong>
            <span>{modeLabel} · {taskSummary} · {searchMethodLabel} · 最適化対象 {optimizedCount}件 · q={q} · fit={fitMaxiter}</span>
          </div>
          {executionButtons()}
        </div>
      </article>
    </>
  );
}
