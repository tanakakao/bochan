import { useCallback, useEffect, useState } from "react";
import {
  COMPOSITION_SETTINGS_CHANGE_EVENT,
  loadCompositionSettings,
  saveCompositionSettings,
  type CompositionSettings
} from "../compositionExtension";

type ElementConstraint = CompositionSettings["constraints"][number];
type ConstraintTerm = ElementConstraint["terms"][number];
type SettingsUpdater = (settings: CompositionSettings) => CompositionSettings;

function newConstraintId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `composition-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function useCompositionSettings(): [CompositionSettings, (updater: SettingsUpdater) => void] {
  const [settings, setSettings] = useState<CompositionSettings>(() => loadCompositionSettings());

  useEffect(() => {
    const refresh = () => setSettings(loadCompositionSettings());
    window.addEventListener(COMPOSITION_SETTINGS_CHANGE_EVENT, refresh);
    return () => window.removeEventListener(COMPOSITION_SETTINGS_CHANGE_EVENT, refresh);
  }, []);

  const update = useCallback((updater: SettingsUpdater) => {
    saveCompositionSettings(updater(loadCompositionSettings()));
  }, []);

  return [settings, update];
}

function patchConstraint(
  settings: CompositionSettings,
  id: string,
  patch: Partial<ElementConstraint>
): CompositionSettings {
  return {
    ...settings,
    constraints: settings.constraints.map((constraint) => (
      constraint.id === id ? { ...constraint, ...patch } : constraint
    ))
  };
}

function patchTerm(
  settings: CompositionSettings,
  constraintId: string,
  termIndex: number,
  patch: Partial<ConstraintTerm>
): CompositionSettings {
  const constraint = settings.constraints.find((item) => item.id === constraintId);
  if (!constraint) return settings;
  return patchConstraint(settings, constraintId, {
    terms: constraint.terms.map((term, index) => (
      index === termIndex ? { ...term, ...patch } : term
    ))
  });
}

export function CompositionSearchSpaceConstraints() {
  const [settings, update] = useCompositionSettings();
  if (!settings.enabled || !settings.column) return null;

  const elementCount = Math.max(settings.elements.length, 1);
  const maximum = settings.maxComponents ?? (settings.elements.length || 1);
  const variableTotal = settings.totalMode === "variable";
  const totalLimit = variableTotal ? settings.totalUpper : settings.total;

  return (
    <section className="composition-search-space-constraints-react">
      <div className="composition-inline-heading">
        <div>
          <span className="panel-kicker">COMPOSITION SEARCH SPACE</span>
          <h4>組成候補の元素制約</h4>
          <p>説明変数の探索範囲と同じ場所で、組成合計・使用元素数・元素ごとの量を設定します。</p>
        </div>
        <span className="status-chip success">{settings.elements.length} elements</span>
      </div>

      <section className="constraint-section composition-total-constraints-react">
        <div className="constraint-section-heading">
          <div>
            <h4>組成合計</h4>
            <p>
              固定合計だけでなく、合計量そのものを探索変数として最適化できます。
              Variable totalでは各元素の絶対量の和が指定範囲に入るよう探索します。
            </p>
          </div>
          <span className={`status-chip ${variableTotal ? "success" : ""}`}>
            {variableTotal ? "VARIABLE TOTAL" : "FIXED TOTAL"}
          </span>
        </div>
        <div className="composition-basic-grid">
          <label>
            <span>合計の扱い</span>
            <select
              value={settings.totalMode}
              onChange={(event) => update((current) => ({
                ...current,
                totalMode: event.target.value === "variable" ? "variable" : "fixed"
              }))}
            >
              <option value="fixed">固定</option>
              <option value="variable">範囲内で最適化</option>
            </select>
          </label>
          {!variableTotal ? (
            <label>
              <span>固定合計</span>
              <input
                type="number"
                min={1e-12}
                step="any"
                value={settings.total}
                onChange={(event) => update((current) => ({
                  ...current,
                  total: Math.max(1e-12, Number(event.target.value) || 1e-12)
                }))}
              />
            </label>
          ) : (
            <>
              <label>
                <span>合計下限</span>
                <input
                  type="number"
                  min={1e-12}
                  step="any"
                  value={settings.totalLower}
                  onChange={(event) => {
                    const value = Math.max(1e-12, Number(event.target.value) || 1e-12);
                    update((current) => ({
                      ...current,
                      totalLower: value,
                      totalUpper: Math.max(current.totalUpper, value + 1e-6)
                    }));
                  }}
                />
              </label>
              <label>
                <span>合計上限</span>
                <input
                  type="number"
                  min={settings.totalLower + 1e-6}
                  step="any"
                  value={settings.totalUpper}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    update((current) => ({
                      ...current,
                      totalUpper: Number.isFinite(value) && value > current.totalLower
                        ? value
                        : current.totalLower + 1e-6
                    }));
                  }}
                />
              </label>
            </>
          )}
        </div>
        {variableTotal && (
          <p className="settings-note">
            Variable totalでは学習モデルにも組成合計featureを追加し、元素supportと合計量を同じ候補最適化で決定します。
          </p>
        )}
      </section>

      <section className="constraint-section composition-count-constraints-react">
        <div className="constraint-section-heading">
          <div>
            <h4>元素数の制約</h4>
            <p>候補組成に含める元素種類数の最小値と最大値を指定します。</p>
          </div>
        </div>
        <div className="composition-basic-grid composition-count-grid">
          <label>
            <span>最小使用元素数</span>
            <input
              type="number"
              min={1}
              max={elementCount}
              value={settings.minComponents}
              onChange={(event) => {
                const value = Math.max(1, Math.min(Number(event.target.value), elementCount));
                update((current) => ({
                  ...current,
                  minComponents: value,
                  maxComponents: Math.max(
                    value,
                    current.maxComponents ?? (current.elements.length || value)
                  )
                }));
              }}
            />
          </label>
          <label>
            <span>最大使用元素数</span>
            <input
              type="number"
              min={settings.minComponents}
              max={elementCount}
              value={maximum}
              onChange={(event) => {
                const value = Math.max(
                  settings.minComponents,
                  Math.min(Number(event.target.value), elementCount)
                );
                update((current) => ({ ...current, maxComponents: value }));
              }}
            />
          </label>
        </div>
      </section>

      <section className="constraint-section composition-element-section composition-ratio-constraints-react">
        <div className="constraint-section-heading">
          <div>
            <h4>{variableTotal ? "元素ごとの絶対量制約" : "元素ごとの量制約"}</h4>
            <p>
              {variableTotal
                ? `上下限と刻みは各元素の絶対量で指定します。元素量の合計は ${settings.totalLower}〜${settings.totalUpper} の範囲です。`
                : `上下限と刻みは固定合計 ${settings.total} と同じ量基準で指定します。`}
            </p>
          </div>
        </div>
        {settings.elements.length === 0 ? (
          <div className="constraint-empty">候補元素をModel画面で指定してください。</div>
        ) : (
          <div className="table-wrap">
            <table className="composition-element-table">
              <thead>
                <tr><th>元素</th><th>下限</th><th>上限</th><th>刻み</th><th>必須</th></tr>
              </thead>
              <tbody>
                {settings.elements.map((element) => {
                  const pair = settings.bounds[element] ?? [0, totalLimit];
                  const step = settings.steps[element];
                  const required = settings.requiredComponents.includes(element);
                  return (
                    <tr key={element}>
                      <td><strong>{element}</strong></td>
                      <td>
                        <input
                          type="number"
                          min={0}
                          max={totalLimit}
                          step="any"
                          value={pair[0]}
                          onChange={(event) => {
                            const value = Math.max(0, Number(event.target.value));
                            update((current) => {
                              const currentLimit = current.totalMode === "variable"
                                ? current.totalUpper
                                : current.total;
                              const currentPair = current.bounds[element] ?? [0, currentLimit];
                              return {
                                ...current,
                                bounds: {
                                  ...current.bounds,
                                  [element]: [value, Math.max(value, currentPair[1])]
                                }
                              };
                            });
                          }}
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          min={0}
                          max={totalLimit}
                          step="any"
                          value={pair[1]}
                          onChange={(event) => {
                            const value = Math.max(0, Number(event.target.value));
                            update((current) => {
                              const currentLimit = current.totalMode === "variable"
                                ? current.totalUpper
                                : current.total;
                              const currentPair = current.bounds[element] ?? [0, currentLimit];
                              return {
                                ...current,
                                bounds: {
                                  ...current.bounds,
                                  [element]: [currentPair[0], Math.max(currentPair[0], value)]
                                }
                              };
                            });
                          }}
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          min={0}
                          max={totalLimit}
                          step="any"
                          value={step ?? ""}
                          placeholder="任意"
                          onChange={(event) => {
                            const raw = event.target.value.trim();
                            const value = raw ? Math.max(0, Number(raw)) : null;
                            update((current) => ({
                              ...current,
                              steps: { ...current.steps, [element]: value }
                            }));
                          }}
                        />
                      </td>
                      <td>
                        <input
                          className="table-checkbox"
                          type="checkbox"
                          checked={required}
                          onChange={(event) => update((current) => ({
                            ...current,
                            requiredComponents: event.target.checked
                              ? [...new Set([...current.requiredComponents, element])]
                              : current.requiredComponents.filter((value) => value !== element)
                          }))}
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </section>
  );
}

export function CompositionLinearConstraints() {
  const [settings, update] = useCompositionSettings();
  if (!settings.enabled || !settings.column) return null;

  function addConstraint(): void {
    update((current) => {
      const first = current.elements[0] ?? "";
      const second = current.elements[1] ?? first;
      return {
        ...current,
        constraints: [
          ...current.constraints,
          {
            id: newConstraintId(),
            terms: [
              { element: first, coefficient: 1 },
              { element: second, coefficient: -1 }
            ].filter((term) => term.element),
            operator: "=",
            rhs: 0,
            basis: "atomic_amount"
          }
        ]
      };
    });
  }

  return (
    <section className="constraint-section composition-element-section composition-linear-constraints-react">
      <div className="constraint-section-heading">
        <div>
          <h4>元素間の線形制約</h4>
          <p>例: Sr − 0.5 × La = 0 とすると、Sr量をLa量の半分に固定できます。</p>
        </div>
        <button
          type="button"
          className="secondary"
          disabled={settings.elements.length === 0}
          onClick={addConstraint}
        >
          制約を追加
        </button>
      </div>

      {settings.constraints.length === 0 ? (
        <div className="constraint-empty">元素間制約は設定されていません。</div>
      ) : (
        <div className="constraint-list">
          {settings.constraints.map((constraint, constraintIndex) => (
            <div className="constraint-card" key={constraint.id}>
              <div className="constraint-card-heading">
                <div>
                  <span className="constraint-index">{constraintIndex + 1}</span>
                  <strong>Σ（係数 × 元素量）</strong>
                </div>
                <button
                  type="button"
                  className="constraint-delete"
                  aria-label={`元素制約${constraintIndex + 1}を削除`}
                  onClick={() => update((current) => ({
                    ...current,
                    constraints: current.constraints.filter((item) => item.id !== constraint.id)
                  }))}
                >×</button>
              </div>

              <div className="composition-term-list">
                {constraint.terms.map((term, termIndex) => (
                  <div className="composition-term-row" key={`${constraint.id}-${termIndex}`}>
                    <input
                      type="number"
                      step="any"
                      value={term.coefficient}
                      onChange={(event) => update((current) => patchTerm(
                        current,
                        constraint.id,
                        termIndex,
                        { coefficient: Number(event.target.value) }
                      ))}
                    />
                    <span>×</span>
                    <select
                      value={term.element}
                      onChange={(event) => update((current) => patchTerm(
                        current,
                        constraint.id,
                        termIndex,
                        { element: event.target.value }
                      ))}
                    >
                      {settings.elements.map((element) => (
                        <option key={element} value={element}>{element}</option>
                      ))}
                    </select>
                    <button
                      type="button"
                      className="secondary compact"
                      disabled={constraint.terms.length <= 1}
                      onClick={() => update((current) => patchConstraint(current, constraint.id, {
                        terms: constraint.terms.filter((_term, index) => index !== termIndex)
                      }))}
                    >
                      削除
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  className="secondary compact"
                  onClick={() => update((current) => patchConstraint(current, constraint.id, {
                    terms: [
                      ...constraint.terms,
                      { element: current.elements[0] ?? "", coefficient: 1 }
                    ]
                  }))}
                >
                  項を追加
                </button>
              </div>

              <div className="composition-constraint-relation">
                <select
                  value={constraint.operator}
                  onChange={(event) => update((current) => patchConstraint(current, constraint.id, {
                    operator: event.target.value as ElementConstraint["operator"]
                  }))}
                >
                  <option value="=">=</option>
                  <option value="<=">≤</option>
                  <option value=">=">≥</option>
                </select>
                <input
                  type="number"
                  step="any"
                  value={constraint.rhs}
                  onChange={(event) => update((current) => patchConstraint(current, constraint.id, {
                    rhs: Number(event.target.value)
                  }))}
                />
                <select
                  value={constraint.basis}
                  onChange={(event) => update((current) => patchConstraint(current, constraint.id, {
                    basis: event.target.value as ElementConstraint["basis"]
                  }))}
                >
                  <option value="atomic_amount">原子量・mol量基準</option>
                  <option value="weight_amount">重量基準</option>
                </select>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
