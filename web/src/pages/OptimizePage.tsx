import { useEffect, useMemo, useState } from "react";
import { fetchCapabilities, type WebCapabilities } from "../api";
import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import type { AcquisitionFamily } from "../types";

const WEB_MODEL_TYPES = [
  "base",
  "deepgp",
  "deepkernel",
  "saas",
  "pca",
  "rembo",
  "robust",
  "hetero",
  "multitask"
] as const;

const FAMILY_OPTIONS: Array<{ value: AcquisitionFamily; label: string }> = [
  { value: "bayesian_optimization", label: "ベイズ最適化" },
  { value: "active_learning", label: "アクティブラーニング" },
  { value: "level_set_estimation", label: "レベルセット推定" }
];

const FAMILY_ACQUISITIONS: Record<AcquisitionFamily, string[]> = {
  bayesian_optimization: ["EI", "NEI", "UCB", "EHVI", "NEHVI"],
  active_learning: ["variance", "predictive_entropy", "BALD"],
  level_set_estimation: ["straddle", "boundary_variance"]
};

const FALLBACK_CAPABILITIES: WebCapabilities = {
  task_types: ["regression", "classification", "ordinal", "hybrid"],
  model_types: [...WEB_MODEL_TYPES],
  acquisitions: Object.values(FAMILY_ACQUISITIONS).flat(),
  acquisition_families: FAMILY_ACQUISITIONS,
  optimizers: ["optimize_acqf"],
  data_sources: ["csv", "excel"],
  visualizations: ["yyplot", "prediction-1d", "prediction-2d"]
};

function taskLabel(value: string): string {
  if (value === "classification") return "分類";
  if (value === "ordinal") return "順序回帰";
  return "回帰";
}

function familyLabel(value: AcquisitionFamily): string {
  return FAMILY_OPTIONS.find((option) => option.value === value)?.label ?? value;
}

/** Configures the surrogate, acquisition family, acquisition, and candidate-generation budget. */
export default function OptimizePage() {
  const {
    dataset,
    settingsValid,
    optimizedTargetSettings,
    selectedVariables,
    modelType,
    setModelType,
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
  const [capabilities, setCapabilities] = useState<WebCapabilities>(FALLBACK_CAPABILITIES);

  useEffect(() => {
    let active = true;
    fetchCapabilities()
      .then((response) => {
        if (active) setCapabilities(response);
      })
      .catch(() => {
        if (active) setCapabilities(FALLBACK_CAPABILITIES);
      });
    return () => {
      active = false;
    };
  }, []);

  const optimizedCount = optimizedTargetSettings.length;
  const hasCategoricalFeatures = selectedVariables.some((variable) => variable.type === "categorical");
  const taskTypes = optimizedTargetSettings.map((setting) => setting.task_type);
  const homogeneousTask = taskTypes.length > 0 && taskTypes.every((task) => task === taskTypes[0]);
  const allRegression = taskTypes.length > 0 && taskTypes.every((task) => task === "regression");
  const canUseMultitask = acquisitionFamily === "bayesian_optimization" &&
    optimizedCount > 1 && allRegression && !hasCategoricalFeatures;
  const availableModels = useMemo(
    () => WEB_MODEL_TYPES.filter((name) => name !== "multitask" || canUseMultitask),
    [canUseMultitask]
  );

  const supportedByApi = capabilities.acquisition_families?.[acquisitionFamily] ??
    capabilities.acquisitions ?? FAMILY_ACQUISITIONS[acquisitionFamily];
  const acquisitionOptions = useMemo(() => {
    let requested: string[];
    if (acquisitionFamily === "bayesian_optimization") {
      requested = optimizedCount > 1 ? ["NEHVI", "EHVI"] : ["EI", "NEI", "UCB"];
    } else {
      requested = FAMILY_ACQUISITIONS[acquisitionFamily];
    }
    const filtered = requested.filter((name) => supportedByApi.includes(name));
    return filtered.length ? filtered : requested;
  }, [acquisitionFamily, optimizedCount, supportedByApi.join("\u0000")]);

  useEffect(() => {
    if (!availableModels.includes(modelType as (typeof WEB_MODEL_TYPES)[number])) {
      setModelType("base");
    }
  }, [availableModels, modelType, setModelType]);

  useEffect(() => {
    if (!acquisitionOptions.includes(acquisition)) {
      setAcquisition(acquisitionOptions[0]);
    }
  }, [acquisition, acquisitionOptions, setAcquisition]);

  function changeFamily(nextFamily: AcquisitionFamily) {
    setAcquisitionFamily(nextFamily);
    if (nextFamily === "bayesian_optimization") {
      setAcquisition(optimizedCount > 1 ? "NEHVI" : "EI");
    } else if (nextFamily === "active_learning") {
      setAcquisition("variance");
    } else {
      setAcquisition("straddle");
    }
  }

  const validationErrors = useMemo(() => {
    const errors: string[] = [];
    if (!settingsValid) errors.push("Settingsページの目的変数または探索変数設定を確認してください。");
    if (optimizedCount === 0) errors.push("最適化対象の目的変数を1つ以上選択してください。");
    if (modelType === "multitask" && !canUseMultitask) {
      errors.push("multitaskはベイズ最適化で、複数の回帰目的かつ説明変数がすべて数値の場合に選択できます。");
    }
    if (
      acquisitionFamily === "level_set_estimation" &&
      optimizedTargetSettings.some((setting) => setting.goal === "none")
    ) {
      errors.push("レベルセット推定では、最適化対象ごとに以上・以下・目標値のいずれかを設定してください。");
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
    modelType,
    numRestarts,
    optimizedCount,
    optimizedTargetSettings,
    q,
    rawSamples,
    settingsValid
  ]);

  const canExecute = validationErrors.length === 0;
  const modeLabel = optimizedCount > 1 ? "多目的" : "単目的";
  const taskSummary = homogeneousTask
    ? taskLabel(taskTypes[0] ?? "regression")
    : "混合タスク";

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
        text="モデル、探索の大分類、獲得関数、候補点数と計算量を設定します。"
        action={<button disabled={!canExecute} onClick={() => void execute()}>候補を生成</button>}
      />

      <div className="form-grid optimize-grid">
        <article className="panel compact-panel">
          <div className="panel-title">
            <div><span className="panel-kicker">SURROGATE</span><h3>モデル</h3></div>
          </div>
          <label>
            Model type
            <select value={modelType} onChange={(event) => setModelType(event.target.value)}>
              {availableModels.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          <p className="settings-note">
            {modelType === "robust" ? "内部ではrrpモデルを使用します。" : null}
            {modelType === "multitask" ? "回帰目的間の相関を学習して情報共有します。" : null}
            {!homogeneousTask ? "混合タスクでは目的変数ごとのサブモデルをhybrid wrapperに束ねます。" : null}
          </p>
          <label>
            Fit maxiter
            <input type="number" min={1} value={fitMaxiter} onChange={(event) => setFitMaxiter(Number(event.target.value))} />
          </label>
        </article>

        <article className="panel compact-panel">
          <div className="panel-title">
            <div><span className="panel-kicker">ACQUISITION</span><h3>探索方法</h3></div>
          </div>
          <label>
            大分類
            <select
              value={acquisitionFamily}
              onChange={(event) => changeFamily(event.target.value as AcquisitionFamily)}
            >
              {FAMILY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
          <label>
            獲得関数
            <select value={acquisition} onChange={(event) => setAcquisition(event.target.value)}>
              {acquisitionOptions.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          {acquisition.toUpperCase().includes("UCB") && (
            <label>
              Beta
              <input type="number" min={0} step="0.1" value={beta} onChange={(event) => setBeta(Number(event.target.value))} />
            </label>
          )}
          <p className="settings-note">
            {acquisitionFamily === "bayesian_optimization" && "目的値の改善を狙って候補を選びます。"}
            {acquisitionFamily === "active_learning" && "予測不確実性を減らすために情報量の高い候補を選びます。"}
            {acquisitionFamily === "level_set_estimation" && "設定した境界や目標付近を重点的に探索します。"}
          </p>
        </article>

        <article className="panel compact-panel">
          <div className="panel-title">
            <div><span className="panel-kicker">CANDIDATES</span><h3>候補生成</h3></div>
          </div>
          <label>q<input type="number" min={1} max={20} value={q} onChange={(event) => setQ(Number(event.target.value))} /></label>
          <label>num_restarts<input type="number" min={1} value={numRestarts} onChange={(event) => setNumRestarts(Number(event.target.value))} /></label>
          <label>raw_samples<input type="number" min={1} value={rawSamples} onChange={(event) => setRawSamples(Number(event.target.value))} /></label>
        </article>
      </div>

      <article className="panel compact-panel validation-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">VALIDATION</span>
            <h3>実行前チェック</h3>
            <p>モデル適用条件、目的変数の役割、獲得関数、候補生成設定を確認します。</p>
          </div>
          <span className={`status-chip ${canExecute ? "success" : "warning"}`}>
            {canExecute ? "Ready" : `${validationErrors.length} issues`}
          </span>
        </div>
        {canExecute ? (
          <p className="settings-note">設定に矛盾は見つかりませんでした。</p>
        ) : (
          <ul>{validationErrors.map((message) => <li key={message}>{message}</li>)}</ul>
        )}
        <div className="train-launcher">
          <div>
            <strong>{modelType} × {familyLabel(acquisitionFamily)} × {acquisition}</strong>
            <span>{modeLabel} · {taskSummary} · 最適化対象 {optimizedCount}件 · q={q}</span>
          </div>
          <button disabled={!canExecute} onClick={() => void execute()}>学習して候補を生成</button>
        </div>
      </article>
    </>
  );
}
