import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";

export default function OptimizePage() {
  const {
    dataset,
    canConfigure,
    direction,
    setDirection,
    modelType,
    setModelType,
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
    execute,
    numberOrUndefined
  } = useWorkbench();

  if (!dataset || !canConfigure) {
    return (
      <>
        <SectionHeader
          step="3 · OPTIMIZE"
          title="モデルと探索空間を設定する"
          text="先に目的変数と説明変数を設定してください。"
        />
        <EmptyState>探索に必要な変数設定が完了していません。</EmptyState>
      </>
    );
  }

  return (
    <>
      <SectionHeader
        step="3 · OPTIMIZE"
        title="モデルと探索空間を設定する"
        text="GPモデル、獲得関数、候補数、各変数の探索範囲を設定します。"
        action={<button onClick={() => void execute()}>候補を生成</button>}
      />

      <div className="form-grid">
        <article className="panel compact-panel">
          <div className="panel-title">
            <div><span className="panel-kicker">OBJECTIVE</span><h3>最適化方向</h3></div>
          </div>
          <label>
            Direction
            <select value={direction} onChange={(event) => setDirection(event.target.value as "maximize" | "minimize")}>
              <option value="maximize">最大化</option>
              <option value="minimize">最小化</option>
            </select>
          </label>
        </article>

        <article className="panel compact-panel">
          <div className="panel-title">
            <div><span className="panel-kicker">SURROGATE</span><h3>GPモデル</h3></div>
          </div>
          <label>
            Model type
            <select value={modelType} onChange={(event) => setModelType(event.target.value)}>
              <option value="base">Base GP</option>
              <option value="saas">MAP-SAAS</option>
              <option value="deepkernel">Deep Kernel GP</option>
            </select>
          </label>
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
              <option value="EI">EI</option>
              <option value="NEI">NEI</option>
              <option value="UCB">UCB</option>
            </select>
          </label>
          {acquisition === "UCB" && (
            <label>
              Beta
              <input type="number" step="0.1" value={beta} onChange={(event) => setBeta(Number(event.target.value))} />
            </label>
          )}
        </article>
      </div>

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">CANDIDATE GENERATION</span>
            <h3>候補生成設定</h3>
            <p>q個の候補を逐次最適化します。</p>
          </div>
        </div>
        <div className="form-grid candidate-settings">
          <label>q<input type="number" min={1} value={q} onChange={(event) => setQ(Number(event.target.value))} /></label>
          <label>num_restarts<input type="number" min={1} value={numRestarts} onChange={(event) => setNumRestarts(Number(event.target.value))} /></label>
          <label>raw_samples<input type="number" min={1} value={rawSamples} onChange={(event) => setRawSamples(Number(event.target.value))} /></label>
        </div>
      </article>

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">SEARCH SPACE</span>
            <h3>探索変数</h3>
            <p>観測範囲を初期値として、下限・上限・刻み・固定値を編集できます。</p>
          </div>
          <span className="status-chip success">{selectedVariables.length} variables</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>変数</th><th>型</th><th>下限</th><th>上限</th><th>刻み</th><th>固定</th><th>固定値</th></tr>
            </thead>
            <tbody>
              {selectedVariables.map((variable) => (
                <tr key={variable.name}>
                  <td><strong>{variable.name}</strong></td>
                  <td><span className="status-chip">{variable.type}</span></td>
                  <td>{variable.type === "numeric" ? <input type="number" value={variable.lower ?? ""} onChange={(event) => patchVariable(variable.name, { lower: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                  <td>{variable.type === "numeric" ? <input type="number" value={variable.upper ?? ""} onChange={(event) => patchVariable(variable.name, { upper: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                  <td>{variable.type === "numeric" ? <input type="number" min={0} value={variable.step ?? ""} placeholder="任意" onChange={(event) => patchVariable(variable.name, { step: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                  <td><input className="table-checkbox" type="checkbox" checked={variable.fixed} onChange={(event) => patchVariable(variable.name, { fixed: event.target.checked })} /></td>
                  <td>
                    {variable.fixed ? (
                      variable.type === "categorical" ? (
                        <select value={String(variable.fixed_value ?? variable.categories?.[0] ?? "")} onChange={(event) => patchVariable(variable.name, { fixed_value: event.target.value })}>
                          {(variable.categories ?? []).map((value) => <option key={value} value={value}>{value}</option>)}
                        </select>
                      ) : (
                        <input type="number" value={variable.fixed_value ?? ""} onChange={(event) => patchVariable(variable.name, { fixed_value: numberOrUndefined(event.target.value) })} />
                      )
                    ) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="train-launcher">
          <div>
            <strong>{modelType} × {acquisition}</strong>
            <span>{direction === "maximize" ? "目的値を最大化" : "目的値を最小化"} · q={q}</span>
          </div>
          <button onClick={() => void execute()}>学習して候補を生成</button>
        </div>
      </article>
    </>
  );
}
