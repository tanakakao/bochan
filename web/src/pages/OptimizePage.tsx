import { useEffect, useMemo, useState } from "react";
import { fetchCapabilities, type WebCapabilities } from "../api";
import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import type { TargetSetting } from "../types";

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

const FALLBACK_CAPABILITIES: WebCapabilities = {
  task_types: ["regression", "classification", "ordinal", "hybrid"],
  model_types: [...WEB_MODEL_TYPES],
  acquisitions: ["EI", "NEI", "UCB", "EHVI", "NEHVI"],
  optimizers: ["optimize_acqf"],
  data_sources: ["csv", "excel"],
  visualizations: ["yyplot", "prediction-1d", "prediction-2d"]
};

function taskLabel(value: string): string {
  if (value === "classification") return "分類";
  if (value === "ordinal") return "順序回帰";
  return "回帰";
}

function goalLabel(value: string): string {
  if (value === "none") return "制約なし";
  if (value === "below") return "以下";
  if (value === "target") return "目標値";
  return "以上";
}

function settingDetails(setting: TargetSetting): string {
  const details: string[] = [taskLabel(setting.task_type)];
  if (setting.task_type === "classification") {
    const classes = setting.target_class !== null && setting.target_class !== undefined
      ? [setting.target_class]
      : (setting.target_classes ?? []);
    details.push(`target class: ${classes.map(String).join(", ") || "—"}`);
  }
  if (setting.task_type === "ordinal") {
    details.push(`order: ${(setting.class_order ?? []).map(String).join(" < ") || "—"}`);
  }
  if (setting.goal === "none") {
    details.push(goalLabel(setting.goal));
  } else if (setting.goal === "target" && setting.task_type === "ordinal") {
    details.push(`${goalLabel(setting.goal)}: ${(setting.target_values ?? []).map(String).join(", ")}`);
  } else {
    details.push(`${goalLabel(setting.goal)} ${String(setting.value ?? "—")}`);
  }
  return details.join(" · ");
}

/** Configures the surrogate, acquisition, and candidate-generation budget. */
export default function OptimizePage() {
  const {
    dataset,
    settingsValid,
    targetColumns,
    selectedTargetSettings,
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

  useEffect(() => {
    if (acquisitionFamily !== "bayesian_optimization") {
      setAcquisitionFamily("bayesian_optimization");
    }
  }, [acquisitionFamily, setAcquisitionFamily]);

  const hasCategoricalFeatures = selectedVariables.some((variable) => variable.type === "categorical");
  const taskTypes = selectedTargetSettings.map((setting) => setting.task_type);
  const homogeneousTask = taskTypes.length > 0 && taskTypes.every((task) => task === taskTypes[0]);
  const allRegression = taskTypes.length > 0 && taskTypes.every((task) => task === "regression");
  const canUseMultitask = targetColumns.length > 1 && allRegression && !hasCategoricalFeatures;
  const availableModels = useMemo(
    () => WEB_MODEL_TYPES.filter((name) => name !== "multitask" || canUseMultitask),
    [canUseMultitask]
  );
  const acquisitionOptions = targetColumns.length > 1
    ? ["NEHVI", "EHVI"]
    : ["EI", "NEI", "UCB"];

  useEffect(() => {
    if (!availableModels.includes(modelType as (typeof WEB_MODEL_TYPES)[number])) {
      setModelType("base");
    }
  }, [availableModels, modelType, setModelType]);

  useEffect(() => {
    if (!acquisitionOptions.includes(acquisition)) {
      setAcquisition(targetColumns.length > 1 ? "NEHVI" : "EI");
    }
  }, [acquisition, setAcquisition, targetColumns.length]);

  const validationErrors = useMemo(() => {
    const errors: string[] = [];
    if (!settingsValid) errors.push("Settingsページの目的変数または探索変数設定を確認してください。");
    if (modelType === "multitask" && !canUseMultitask) {
      errors.push("multitaskは複数の回帰目的で、説明変数がすべて数値の場合に選択できます。");
    }
    if (fitMaxiter < 1) errors.push("fit maxiterは1以上にしてください。");
    if (q < 1 || q > 20) errors.push("候補点数qは1〜20にしてください。");
    if (numRestarts < 1) errors.push("num_restartsは1以上にしてください。");
    if (rawSamples < 1) errors.push("raw_samplesは1以上にしてください。");
    return errors;
  }, [canUseMultitask, fitMaxiter, modelType, numRestarts, q, rawSamples, settingsValid]);

  const canExecute = validationErrors.length === 0;
  const modeLabel = targetColumns.length > 1 ? "多目的最適化" : "単目的最適化";
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
        text="この画面ではモデル、獲得関数、候補点数と計算量だけを設定します。"
        action={<button disabled={!canExecute} onClick={() => void execute()}>候補を生成</button>}
      />

      <article className="panel compact-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">CONFIGURATION SUMMARY</span>
            <h3>目的変数設定</h3>
            <p>{modeLabel} · {taskSummary} · {targetColumns.length} targets</p>
          </div>
          <span className="status-chip success">Configured</span>
        </div>
        <div className="cards">
          {selectedTargetSettings.map((setting) => (
            <div className="settings-note" key={setting.target}>
              <strong>{setting.target}</strong>
              <span> · {settingDetails(setting)}</span>
            </div>
          ))}
        </div>
      </article>

      <div className="form-grid">
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
            <div><span className="panel-kicker">ACQUISITION</span><h3>獲得関数</h3></div>
          </div>
          <label>
            Acquisition
            <select value={acquisition} onChange={(event) => setAcquisition(event.target.value)}>
              {acquisitionOptions
                .filter((name) => capabilities.acquisitions.includes(name) || FALLBACK_CAPABILITIES.acquisitions.includes(name))
                .map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          {acquisition.toUpperCase().includes("UCB") && (
            <label>
              Beta
              <input type="number" min={0} step="0.1" value={beta} onChange={(event) => setBeta(Number(event.target.value))} />
            </label>
          )}
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

      <article className="panel compact-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">VALIDATION</span>
            <h3>実行前チェック</h3>
            <p>モデル適用条件と候補生成設定を確認します。</p>
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
            <strong>{modelType} × {acquisition}</strong>
            <span>{modeLabel} · {taskSummary} · q={q}</span>
          </div>
          <button disabled={!canExecute} onClick={() => void execute()}>学習して候補を生成</button>
        </div>
      </article>
    </>
  );
}
