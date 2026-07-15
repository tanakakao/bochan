import { useEffect, useMemo, useState } from "react";
import { fetchCapabilities, type WebCapabilities } from "../api";
import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import type { SearchVariable } from "../types";

const FALLBACK_CAPABILITIES: WebCapabilities = {
  task_types: ["regression"],
  model_types: ["base", "saas", "deepkernel"],
  acquisitions: ["EI", "NEI", "UCB"],
  optimizers: ["optimize_acqf"],
  data_sources: ["csv", "excel"],
  visualizations: ["yyplot", "prediction-1d", "prediction-2d"]
};

function finiteNumber(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function validateVariable(variable: SearchVariable): string[] {
  const errors: string[] = [];
  if (variable.type === "numeric") {
    const lower = finiteNumber(variable.lower);
    const upper = finiteNumber(variable.upper);
    if (lower === null || upper === null) {
      errors.push(`${variable.name}: 下限と上限を入力してください。`);
      return errors;
    }
    if (lower >= upper) errors.push(`${variable.name}: 下限は上限より小さくしてください。`);
    if (variable.step !== undefined) {
      const step = finiteNumber(variable.step);
      if (step === null || step <= 0) errors.push(`${variable.name}: 刻み幅は正の数にしてください。`);
      else if (upper > lower && step > upper - lower) errors.push(`${variable.name}: 刻み幅が探索範囲より大きいです。`);
    }
    if (variable.fixed) {
      const fixed = finiteNumber(variable.fixed_value);
      if (fixed === null) errors.push(`${variable.name}: 固定値を数値で入力してください。`);
      else if (fixed < lower || fixed > upper) errors.push(`${variable.name}: 固定値を探索範囲内にしてください。`);
    }
  } else {
    if (variable.fixed && (variable.fixed_value === undefined || String(variable.fixed_value).trim() === "")) {
      errors.push(`${variable.name}: 固定するカテゴリを入力してください。`);
    }
    if (
      variable.fixed &&
      variable.categories?.length &&
      !variable.categories.map(String).includes(String(variable.fixed_value))
    ) {
      errors.push(`${variable.name}: 固定値が既知カテゴリに含まれていません。`);
    }
  }
  return errors;
}

/** Renders the verified single-objective regression settings. */
export default function OptimizePage() {
  const {
    dataset,
    columns,
    canConfigure,
    targetColumns,
    taskType,
    setTaskType,
    direction,
    setDirection,
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
    selectedVariables,
    patchVariable,
    outcomeConstraints,
    removeOutcomeConstraint,
    linearConstraints,
    removeLinearConstraint,
    kSparse,
    setKSparse,
    execute,
    numberOrUndefined
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
    if (taskType !== "regression") setTaskType("regression");
    if (acquisitionFamily !== "bayesian_optimization") {
      setAcquisitionFamily("bayesian_optimization");
    }
  }, [acquisitionFamily, setAcquisitionFamily, setTaskType, taskType]);

  useEffect(() => {
    if (!capabilities.model_types.includes(modelType)) {
      setModelType(capabilities.model_types[0] ?? "base");
    }
  }, [capabilities.model_types, modelType, setModelType]);

  useEffect(() => {
    if (!capabilities.acquisitions.includes(acquisition)) {
      setAcquisition(capabilities.acquisitions[0] ?? "EI");
    }
  }, [acquisition, capabilities.acquisitions, setAcquisition]);

  const columnKind = useMemo(
    () => Object.fromEntries(columns.map((column) => [column.name, column.kind])),
    [columns]
  );
  const validationErrors = useMemo(() => {
    const errors = selectedVariables.flatMap(validateVariable);
    if (targetColumns.length !== 1) errors.unshift("目的変数は1列だけ選択してください。");
    if (fitMaxiter < 1) errors.push("fit maxiterは1以上にしてください。");
    if (q < 1 || q > 20) errors.push("候補点数qは1〜20にしてください。");
    if (numRestarts < 1) errors.push("num_restartsは1以上にしてください。");
    if (rawSamples < 1) errors.push("raw_samplesは1以上にしてください。");
    if (outcomeConstraints.length || linearConstraints.length || kSparse.enabled) {
      errors.push("以前の画面で設定した未対応の制約またはk-sparse設定が残っています。クリアしてください。");
    }
    return errors;
  }, [fitMaxiter, kSparse.enabled, linearConstraints.length, numRestarts, outcomeConstraints.length, q, rawSamples, selectedVariables, targetColumns.length]);

  const canExecute = validationErrors.length === 0;

  if (!dataset || !canConfigure) {
    return (
      <>
        <SectionHeader step="3 · OPTIMIZE" title="モデルと探索空間を設定する" text="先に目的変数と説明変数を設定してください。" />
        <EmptyState>探索に必要な変数設定が完了していません。</EmptyState>
      </>
    );
  }

  function resetUnsupportedSettings() {
    outcomeConstraints.forEach((constraint) => removeOutcomeConstraint(constraint.id));
    linearConstraints.forEach((constraint) => removeLinearConstraint(constraint.id));
    setKSparse({ enabled: false, k: 1, variables: [] });
  }

  function setVariableType(variable: SearchVariable, categorical: boolean) {
    const nextType = categorical ? "categorical" : "numeric";
    patchVariable(variable.name, {
      type: nextType,
      fixed: false,
      fixed_value: undefined,
      step: nextType === "categorical" ? undefined : variable.step
    });
  }

  return (
    <>
      <SectionHeader
        step="3 · OPTIMIZE"
        title="モデルと探索空間を設定する"
        text="FastAPIが公開する対応モデル・獲得関数に限定し、探索範囲を実行前に検証します。"
        action={<button disabled={!canExecute} onClick={() => void execute()}>候補を生成</button>}
      />

      <article className="panel compact-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">SUPPORTED SCOPE</span>
            <h3>現在のWeb API対応範囲</h3>
            <p>単目的回帰、数値・カテゴリ説明変数、最大化／最小化、範囲・刻み・固定値に対応します。</p>
          </div>
          <span className="status-chip success">Regression MVP</span>
        </div>
        <p className="settings-note">
          多目的、分類、順序回帰、目的変数制約、線形制約、k-sparse、Active Learning、Level-setは
          ライブラリ側には実装がありますが、このWeb APIにはまだ接続していません。
        </p>
        {(outcomeConstraints.length > 0 || linearConstraints.length > 0 || kSparse.enabled) && (
          <button className="secondary" onClick={resetUnsupportedSettings}>未対応設定をクリア</button>
        )}
      </article>

      <div className="form-grid">
        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">OBJECTIVE</span><h3>目的</h3></div></div>
          <label>Direction
            <select value={direction} onChange={(event) => setDirection(event.target.value as "maximize" | "minimize")}>
              <option value="maximize">最大化</option>
              <option value="minimize">最小化</option>
            </select>
          </label>
        </article>

        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">SURROGATE</span><h3>モデル</h3></div></div>
          <label>Model type
            <select value={modelType} onChange={(event) => setModelType(event.target.value)}>
              {capabilities.model_types.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          <label>Fit maxiter
            <input type="number" min={1} value={fitMaxiter} onChange={(event) => setFitMaxiter(Number(event.target.value))} />
          </label>
        </article>

        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">ACQUISITION</span><h3>獲得関数</h3></div></div>
          <label>Acquisition
            <select value={acquisition} onChange={(event) => setAcquisition(event.target.value)}>
              {capabilities.acquisitions.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          {acquisition.toUpperCase().includes("UCB") && (
            <label>Beta
              <input type="number" min={0} step="0.1" value={beta} onChange={(event) => setBeta(Number(event.target.value))} />
            </label>
          )}
        </article>
      </div>

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">CANDIDATE GENERATION</span>
            <h3>候補生成設定</h3>
            <p>候補点数と獲得関数最適化の計算量を設定します。</p>
          </div>
        </div>
        <div className="form-grid candidate-settings">
          <label>q<input type="number" min={1} max={20} value={q} onChange={(event) => setQ(Number(event.target.value))} /></label>
          <label>num_restarts<input type="number" min={1} value={numRestarts} onChange={(event) => setNumRestarts(Number(event.target.value))} /></label>
          <label>raw_samples<input type="number" min={1} value={rawSamples} onChange={(event) => setRawSamples(Number(event.target.value))} /></label>
        </div>
      </article>

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">SEARCH SPACE</span>
            <h3>探索変数</h3>
            <p>観測範囲を初期値として、カテゴリ扱い・下限・上限・刻み・固定値を編集できます。</p>
          </div>
          <span className="status-chip success">{selectedVariables.length} variables</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>変数</th><th>カテゴリ?</th><th>型</th><th>下限</th><th>上限</th><th>刻み</th><th>固定</th><th>固定値</th></tr></thead>
            <tbody>
              {selectedVariables.map((variable) => {
                const detectedCategorical = columnKind[variable.name] === "categorical";
                return (
                  <tr key={variable.name}>
                    <td><strong>{variable.name}</strong></td>
                    <td>
                      <input
                        className="table-checkbox"
                        type="checkbox"
                        checked={variable.type === "categorical"}
                        disabled={detectedCategorical}
                        title={detectedCategorical ? "文字列・カテゴリ列は数値変数へ変更できません。" : "数値列を離散カテゴリとして扱えます。"}
                        onChange={(event) => setVariableType(variable, event.target.checked)}
                      />
                    </td>
                    <td><span className="status-chip">{variable.type}</span></td>
                    <td>{variable.type === "numeric" ? <input type="number" value={variable.lower ?? ""} onChange={(event) => patchVariable(variable.name, { lower: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                    <td>{variable.type === "numeric" ? <input type="number" value={variable.upper ?? ""} onChange={(event) => patchVariable(variable.name, { upper: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                    <td>{variable.type === "numeric" ? <input type="number" min={0} value={variable.step ?? ""} placeholder="任意" onChange={(event) => patchVariable(variable.name, { step: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                    <td><input className="table-checkbox" type="checkbox" checked={variable.fixed} onChange={(event) => patchVariable(variable.name, { fixed: event.target.checked, fixed_value: event.target.checked ? variable.fixed_value : undefined })} /></td>
                    <td>
                      {variable.fixed && variable.type === "categorical" && variable.categories?.length ? (
                        <select value={String(variable.fixed_value ?? "")} onChange={(event) => patchVariable(variable.name, { fixed_value: event.target.value })}>
                          <option value="">選択</option>
                          {variable.categories.map((category) => <option key={category} value={category}>{category}</option>)}
                        </select>
                      ) : variable.fixed ? (
                        <input value={variable.fixed_value ?? ""} onChange={(event) => patchVariable(variable.name, { fixed_value: variable.type === "numeric" ? numberOrUndefined(event.target.value) : event.target.value })} />
                      ) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </article>

      <article className="panel compact-panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">VALIDATION</span>
            <h3>実行前チェック</h3>
            <p>探索範囲、刻み幅、固定値、候補生成設定を確認します。</p>
          </div>
          <span className={`status-chip ${canExecute ? "success" : "warning"}`}>
            {canExecute ? "Ready" : `${validationErrors.length} issues`}
          </span>
        </div>
        {canExecute ? (
          <p className="settings-note">設定に矛盾は見つかりませんでした。</p>
        ) : (
          <ul>
            {validationErrors.map((message) => <li key={message}>{message}</li>)}
          </ul>
        )}
        <div className="train-launcher">
          <div>
            <strong>{modelType} × {acquisition}</strong>
            <span>単目的回帰 · {direction === "maximize" ? "最大化" : "最小化"} · q={q}</span>
          </div>
          <button disabled={!canExecute} onClick={() => void execute()}>学習して候補を生成</button>
        </div>
      </article>
    </>
  );
}
