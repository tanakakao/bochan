import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import type { AcquisitionFamily, TaskType } from "../types";

const ACQUISITIONS: Record<AcquisitionFamily, string[]> = {
  bayesian_optimization: ["EI", "NEI", "LogEI", "PI", "UCB", "KG", "EHVI", "NEHVI"],
  active_learning: ["PosteriorVariance", "qNIPV", "PredictiveEntropy", "BALD", "qBALD"],
  level_set_estimation: ["Straddle", "qStraddle", "ProbabilityOfExceedance", "LevelSetUncertainty", "JointBoundaryVariance"]
};

/** Renders model, acquisition, constraint, and search-space settings. */
export default function OptimizePage() {
  const {
    dataset,
    canConfigure,
    targetColumns,
    targetCandidates,
    taskType,
    setTaskType,
    ordinalOrder,
    setOrdinalOrder,
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
    addOutcomeConstraint,
    patchOutcomeConstraint,
    removeOutcomeConstraint,
    linearConstraints,
    addLinearConstraint,
    patchLinearConstraint,
    patchLinearConstraintTerm,
    removeLinearConstraint,
    kSparse,
    setKSparse,
    execute,
    numberOrUndefined
  } = useWorkbench();

  if (!dataset || !canConfigure) {
    return (
      <>
        <SectionHeader step="3 · OPTIMIZE" title="モデルと探索空間を設定する" text="先に目的変数と説明変数を設定してください。" />
        <EmptyState>探索に必要な変数設定が完了していません。</EmptyState>
      </>
    );
  }

  return (
    <>
      <SectionHeader
        step="3 · OPTIMIZE"
        title="モデルと探索空間を設定する"
        text="タスク、モデル、獲得関数、目的変数制約、入力制約、k-sparse条件を設定します。"
        action={<button onClick={() => void execute()}>候補を生成</button>}
      />

      <div className="form-grid">
        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">OBJECTIVE</span><h3>タスクと目的</h3></div></div>
          <label>Task
            <select value={taskType} onChange={(event) => setTaskType(event.target.value as TaskType)}>
              <option value="regression">回帰</option>
              <option value="classification">分類</option>
              <option value="ordinal">順序回帰</option>
            </select>
          </label>
          <label>Direction
            <select value={direction} onChange={(event) => setDirection(event.target.value as "maximize" | "minimize")}>
              <option value="maximize">最大化</option>
              <option value="minimize">最小化</option>
            </select>
          </label>
          {taskType === "ordinal" && (
            <label>カテゴリ順序
              <input value={ordinalOrder.join(", ")} placeholder="low, medium, high" onChange={(event) => setOrdinalOrder(event.target.value.split(",").map((v) => v.trim()).filter(Boolean))} />
            </label>
          )}
        </article>

        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">SURROGATE</span><h3>モデル</h3></div></div>
          <label>Model type
            <select value={modelType} onChange={(event) => setModelType(event.target.value)}>
              <option value="base">base</option>
              <option value="deepgp">deepgp</option>
              <option value="deepkernel">deepkernel</option>
              <option value="saas">saas</option>
              <option value="pca">pca</option>
              <option value="rrp">rrp</option>
              <option value="hetero">hetero</option>
              {targetColumns.length > 1 && <option value="modellist">modellist</option>}
              {targetColumns.length > 1 && <option value="multitask">multitask</option>}
            </select>
          </label>
          <label>Fit maxiter<input type="number" min={1} value={fitMaxiter} onChange={(event) => setFitMaxiter(Number(event.target.value))} /></label>
        </article>

        <article className="panel compact-panel">
          <div className="panel-title"><div><span className="panel-kicker">ACQUISITION</span><h3>獲得関数</h3></div></div>
          <label>大分類
            <select value={acquisitionFamily} onChange={(event) => {
              const family = event.target.value as AcquisitionFamily;
              setAcquisitionFamily(family);
              setAcquisition(ACQUISITIONS[family][0]);
            }}>
              <option value="bayesian_optimization">bayes optimization</option>
              <option value="active_learning">active learning</option>
              <option value="level_set_estimation">levelset estimation</option>
            </select>
          </label>
          <label>Acquisition
            <select value={acquisition} onChange={(event) => setAcquisition(event.target.value)}>
              {ACQUISITIONS[acquisitionFamily].map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          {acquisition.includes("UCB") && <label>Beta<input type="number" step="0.1" value={beta} onChange={(event) => setBeta(Number(event.target.value))} /></label>}
        </article>
      </div>

      <article className="panel">
        <div className="panel-title"><div><span className="panel-kicker">OUTCOME CONSTRAINTS</span><h3>目的変数の制約</h3><p>目的変数に対するしきい値制約を任意の数だけ追加できます。</p></div><button className="secondary" onClick={addOutcomeConstraint}>追加</button></div>
        <div className="constraint-list">
          {outcomeConstraints.map((constraint) => (
            <div className="constraint-row" key={constraint.id}>
              <select value={constraint.target} onChange={(event) => patchOutcomeConstraint(constraint.id, { target: event.target.value })}>{targetCandidates.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}</select>
              <select value={constraint.operator} onChange={(event) => patchOutcomeConstraint(constraint.id, { operator: event.target.value as ">=" | "<=" })}><option value=">=">以上</option><option value="<=">以下</option></select>
              <input type="number" value={constraint.value} onChange={(event) => patchOutcomeConstraint(constraint.id, { value: Number(event.target.value) })} />
              <button className="ghost-button" onClick={() => removeOutcomeConstraint(constraint.id)}>削除</button>
            </div>
          ))}
          {outcomeConstraints.length === 0 && <p className="settings-note">目的変数制約は未設定です。</p>}
        </div>
      </article>

      <article className="panel">
        <div className="panel-title"><div><span className="panel-kicker">INPUT CONSTRAINTS</span><h3>制約とk-sparse</h3><p>等式・不等式の線形制約と、非ゼロ変数数の上限を設定します。</p></div><div><button className="secondary" onClick={() => addLinearConstraint("equality")}>等式追加</button><button className="secondary" onClick={() => addLinearConstraint("inequality")}>不等式追加</button></div></div>
        <div className="constraint-list">
          {linearConstraints.map((constraint) => (
            <div className="constraint-row wide" key={constraint.id}>
              <strong>{constraint.kind === "equality" ? "等式" : "不等式"}</strong>
              {selectedVariables.map((variable) => <label key={variable.name}>{variable.name}<input type="number" placeholder="係数" value={constraint.terms[variable.name] ?? ""} onChange={(event) => patchLinearConstraintTerm(constraint.id, variable.name, numberOrUndefined(event.target.value))} /></label>)}
              <select value={constraint.operator} onChange={(event) => patchLinearConstraint(constraint.id, { operator: event.target.value as "=" | "<=" | ">=" })}><option value="=">=</option><option value="<=">≤</option><option value=">=">≥</option></select>
              <input type="number" value={constraint.rhs} onChange={(event) => patchLinearConstraint(constraint.id, { rhs: Number(event.target.value) })} />
              <button className="ghost-button" onClick={() => removeLinearConstraint(constraint.id)}>削除</button>
            </div>
          ))}
        </div>
        <div className="form-grid candidate-settings">
          <label><input type="checkbox" checked={kSparse.enabled} onChange={(event) => setKSparse({ ...kSparse, enabled: event.target.checked })} /> k-sparseを有効化</label>
          <label>k<input type="number" min={1} value={kSparse.k} onChange={(event) => setKSparse({ ...kSparse, k: Number(event.target.value) })} /></label>
          <label>対象変数<select multiple value={kSparse.variables} onChange={(event) => setKSparse({ ...kSparse, variables: Array.from(event.target.selectedOptions).map((option) => option.value) })}>{selectedVariables.map((variable) => <option key={variable.name} value={variable.name}>{variable.name}</option>)}</select></label>
        </div>
      </article>

      <article className="panel">
        <div className="panel-title"><div><span className="panel-kicker">CANDIDATE GENERATION</span><h3>候補生成設定</h3><p>q個の候補を逐次最適化します。</p></div></div>
        <div className="form-grid candidate-settings">
          <label>q<input type="number" min={1} value={q} onChange={(event) => setQ(Number(event.target.value))} /></label>
          <label>num_restarts<input type="number" min={1} value={numRestarts} onChange={(event) => setNumRestarts(Number(event.target.value))} /></label>
          <label>raw_samples<input type="number" min={1} value={rawSamples} onChange={(event) => setRawSamples(Number(event.target.value))} /></label>
        </div>
      </article>

      <article className="panel">
        <div className="panel-title"><div><span className="panel-kicker">SEARCH SPACE</span><h3>探索変数</h3><p>観測範囲を初期値として、カテゴリ扱い・下限・上限・刻み・固定値を編集できます。</p></div><span className="status-chip success">{selectedVariables.length} variables</span></div>
        <div className="table-wrap">
          <table><thead><tr><th>変数</th><th>カテゴリ?</th><th>型</th><th>下限</th><th>上限</th><th>刻み</th><th>固定</th><th>固定値</th></tr></thead><tbody>
            {selectedVariables.map((variable) => (
              <tr key={variable.name}>
                <td><strong>{variable.name}</strong></td>
                <td><input className="table-checkbox" type="checkbox" checked={variable.type === "categorical"} onChange={(event) => patchVariable(variable.name, { type: event.target.checked ? "categorical" : "numeric" })} /></td>
                <td><span className="status-chip">{variable.type}</span></td>
                <td>{variable.type === "numeric" ? <input type="number" value={variable.lower ?? ""} onChange={(event) => patchVariable(variable.name, { lower: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                <td>{variable.type === "numeric" ? <input type="number" value={variable.upper ?? ""} onChange={(event) => patchVariable(variable.name, { upper: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                <td>{variable.type === "numeric" ? <input type="number" min={0} value={variable.step ?? ""} placeholder="任意" onChange={(event) => patchVariable(variable.name, { step: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                <td><input className="table-checkbox" type="checkbox" checked={variable.fixed} onChange={(event) => patchVariable(variable.name, { fixed: event.target.checked })} /></td>
                <td>{variable.fixed ? <input value={variable.fixed_value ?? ""} onChange={(event) => patchVariable(variable.name, { fixed_value: variable.type === "numeric" ? numberOrUndefined(event.target.value) : event.target.value })} /> : "—"}</td>
              </tr>
            ))}
          </tbody></table>
        </div>
        <div className="train-launcher"><div><strong>{modelType} × {acquisitionFamily} / {acquisition}</strong><span>{targetColumns.length}目的 · {direction === "maximize" ? "最大化" : "最小化"} · q={q}</span></div><button onClick={() => void execute()}>学習して候補を生成</button></div>
      </article>
    </>
  );
}
