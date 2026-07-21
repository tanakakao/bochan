import { useEffect, useMemo, useState } from "react";
import type { SearchVariable } from "../types";
import {
  loadFeatureConstraints,
  loadFeatureMissingSettings,
  loadSelectionCountConstraint,
  newConstraintId,
  saveFeatureConstraints,
  saveFeatureMissingSettings,
  saveSelectionCountConstraint,
  type FeatureConstraint,
  type FeatureConstraintOperator,
  type FeatureMissingSettings,
  type SelectionCountConstraint
} from "../webRunSettings";

interface Props {
  variables: SearchVariable[];
}

function expression(constraint: FeatureConstraint): string {
  if (!constraint.variables.length) return "説明変数を選択";
  return constraint.variables
    .map((name) => `${constraint.coefficients[name] ?? 1} × ${name}`)
    .join(" + ");
}

/** Edits feature missing-value handling, linear constraints, and a k-sparse constraint. */
export default function FeatureConstraints({ variables }: Props) {
  const numericVariables = useMemo(
    () => variables.filter((variable) => variable.type === "numeric"),
    [variables]
  );
  const numericNames = useMemo(
    () => new Set(numericVariables.map((variable) => variable.name)),
    [numericVariables]
  );
  const [featureMissing, setFeatureMissing] = useState<FeatureMissingSettings>(
    () => loadFeatureMissingSettings()
  );
  const [constraints, setConstraints] = useState<FeatureConstraint[]>(() => loadFeatureConstraints());
  const [selectionCount, setSelectionCount] = useState<SelectionCountConstraint>(
    () => loadSelectionCountConstraint()
  );

  useEffect(() => {
    setConstraints((current) => {
      const next = current.map((constraint) => {
        const selected = constraint.variables.filter((name) => numericNames.has(name));
        return {
          ...constraint,
          variables: selected,
          coefficients: Object.fromEntries(
            selected.map((name) => [name, constraint.coefficients[name] ?? 1])
          )
        };
      });
      saveFeatureConstraints(next);
      return next;
    });
    setSelectionCount((current) => {
      const variables = current.variables.filter((name) => numericNames.has(name));
      const next = {
        ...current,
        variables,
        k: Math.max(1, Math.min(current.k, Math.max(variables.length, 1)))
      };
      saveSelectionCountConstraint(next);
      return next;
    });
  }, [numericNames]);

  function updateMissing(next: FeatureMissingSettings) {
    setFeatureMissing(next);
    saveFeatureMissingSettings(next);
  }

  function update(next: FeatureConstraint[]) {
    setConstraints(next);
    saveFeatureConstraints(next);
  }

  function updateSelectionCount(next: SelectionCountConstraint) {
    setSelectionCount(next);
    saveSelectionCountConstraint(next);
  }

  function addConstraint() {
    const variable = numericVariables[0];
    update([
      ...constraints,
      {
        id: newConstraintId(),
        variables: variable ? [variable.name] : [],
        coefficients: variable ? { [variable.name]: 1 } : {},
        operator: "<",
        value: variable?.upper ?? 0
      }
    ]);
  }

  function patchConstraint(id: string, patch: Partial<FeatureConstraint>) {
    update(constraints.map((constraint) => (
      constraint.id === id ? { ...constraint, ...patch } : constraint
    )));
  }

  function toggleConstraintVariable(constraint: FeatureConstraint, name: string) {
    const selected = constraint.variables.includes(name);
    const variables = selected
      ? constraint.variables.filter((value) => value !== name)
      : [...constraint.variables, name];
    const coefficients = { ...constraint.coefficients };
    if (selected) delete coefficients[name];
    else coefficients[name] = 1;
    patchConstraint(constraint.id, { variables, coefficients });
  }

  function removeConstraint(id: string) {
    update(constraints.filter((constraint) => constraint.id !== id));
  }

  function toggleSelectionVariable(name: string) {
    const selected = selectionCount.variables.includes(name);
    const nextVariables = selected
      ? selectionCount.variables.filter((value) => value !== name)
      : [...selectionCount.variables, name];
    updateSelectionCount({
      ...selectionCount,
      variables: nextVariables,
      k: Math.max(1, Math.min(selectionCount.k, Math.max(nextVariables.length, 1)))
    });
  }

  return (
    <article className="panel feature-constraint-panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">3 · DATA & CONSTRAINTS</span>
          <h3>欠損値と制約</h3>
          <p>説明変数の欠損処理、重み付き和の制約、有効にする変数数を設定します。</p>
        </div>
        <span className={`status-chip ${constraints.length || selectionCount.enabled || featureMissing.strategy === "impute" ? "success" : ""}`}>
          {featureMissing.strategy === "impute" ? "impute" : "drop"}
        </span>
      </div>

      <section className="constraint-section feature-missing-section">
        <div className="constraint-section-heading">
          <div>
            <h4>説明変数の欠損値</h4>
            <p>目的変数の欠損処理とは独立して、説明変数を欠損した行の扱いを選択します。</p>
          </div>
        </div>
        <div className="constraint-variable-picker" role="group" aria-label="説明変数の欠損値処理">
          <button
            type="button"
            className={featureMissing.strategy === "drop" ? "selected" : ""}
            aria-pressed={featureMissing.strategy === "drop"}
            onClick={() => updateMissing({ ...featureMissing, strategy: "drop" })}
          >
            欠損行を削除
          </button>
          <button
            type="button"
            className={featureMissing.strategy === "impute" ? "selected" : ""}
            aria-pressed={featureMissing.strategy === "impute"}
            onClick={() => updateMissing({ ...featureMissing, strategy: "impute" })}
          >
            欠損値を補完
          </button>
        </div>

        {featureMissing.strategy === "impute" && (
          <div className="transform-fields">
            <label>
              数値変数の補完
              <select
                value={featureMissing.continuousStrategy}
                onChange={(event) => updateMissing({
                  ...featureMissing,
                  continuousStrategy: event.target.value === "iterative" ? "iterative" : "mean"
                })}
              >
                <option value="mean">平均値</option>
                <option value="iterative">IterativeImputer</option>
              </select>
            </label>
            <label>
              カテゴリ変数の補完
              <select value="mode" disabled>
                <option value="mode">最頻値</option>
              </select>
            </label>
            {featureMissing.continuousStrategy === "iterative" && (
              <label>
                最大反復回数
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={featureMissing.imputeMaxIter}
                  onChange={(event) => updateMissing({
                    ...featureMissing,
                    imputeMaxIter: Math.max(1, Math.trunc(Number(event.target.value) || 1))
                  })}
                />
              </label>
            )}
          </div>
        )}
        <p className="settings-note">
          補完では、数値変数に平均値またはIterativeImputer、カテゴリ変数に最頻値を使用します。目的変数はここでは補完しません。
        </p>
      </section>

      <section className="constraint-section">
        <div className="constraint-section-heading">
          <div>
            <h4>線形和制約</h4>
            <p>Σ（係数 × 説明変数）に対して、等式または不等式を設定します。</p>
          </div>
          <div className="button-row constraint-actions">
            <button type="button" className="secondary" disabled={!numericVariables.length} onClick={addConstraint}>
              制約を追加
            </button>
            <button type="button" className="secondary" disabled={!constraints.length} onClick={() => update([])}>
              全削除
            </button>
          </div>
        </div>

        {!numericVariables.length ? (
          <p className="settings-note">数値型の説明変数がないため、線形制約は追加できません。</p>
        ) : constraints.length === 0 ? (
          <div className="constraint-empty">制約は設定されていません。</div>
        ) : (
          <div className="constraint-list">
            {constraints.map((constraint, index) => (
              <div className="constraint-card" key={constraint.id}>
                <div className="constraint-card-heading">
                  <div>
                    <span className="constraint-index">{index + 1}</span>
                    <strong>{expression(constraint)}</strong>
                  </div>
                  <button
                    type="button"
                    className="constraint-delete"
                    aria-label={`制約${index + 1}を削除`}
                    title="この制約を削除"
                    onClick={() => removeConstraint(constraint.id)}
                  >×</button>
                </div>

                <div className="constraint-relation-row">
                  <span className="constraint-expression">Σ（係数 × 変数）</span>
                  <select
                    aria-label={`制約${index + 1}の演算子`}
                    value={constraint.operator}
                    onChange={(event) => patchConstraint(constraint.id, {
                      operator: event.target.value as FeatureConstraintOperator
                    })}
                  >
                    <option value=">">&gt;</option>
                    <option value="<">&lt;</option>
                    <option value="=">=</option>
                  </select>
                  <input
                    type="number"
                    aria-label={`制約${index + 1}の値`}
                    value={constraint.value}
                    onChange={(event) => patchConstraint(constraint.id, { value: Number(event.target.value) })}
                  />
                </div>

                <details className="constraint-expander" open>
                  <summary>説明変数と係数を設定</summary>
                  <div className="constraint-variable-picker" role="group" aria-label={`制約${index + 1}の説明変数`}>
                    {numericVariables.map((variable) => {
                      const selected = constraint.variables.includes(variable.name);
                      return (
                        <button
                          key={variable.name}
                          type="button"
                          className={selected ? "selected" : ""}
                          aria-pressed={selected}
                          onClick={() => toggleConstraintVariable(constraint, variable.name)}
                        >
                          {variable.name}
                        </button>
                      );
                    })}
                  </div>
                  {constraint.variables.length > 0 && (
                    <div className="coefficient-grid">
                      {constraint.variables.map((name) => (
                        <label key={name}>
                          <span>{name} の係数</span>
                          <input
                            type="number"
                            step="any"
                            value={constraint.coefficients[name] ?? 1}
                            onChange={(event) => patchConstraint(constraint.id, {
                              coefficients: {
                                ...constraint.coefficients,
                                [name]: Number(event.target.value)
                              }
                            })}
                          />
                        </label>
                      ))}
                    </div>
                  )}
                </details>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="constraint-section selection-count-section">
        <div className="constraint-section-heading">
          <div>
            <h4>有効変数数制約</h4>
            <p>選択した数値変数のうち、候補ごとに非ゼロで採用する最大数を指定します。</p>
          </div>
          <label className="switch-field">
            <input
              type="checkbox"
              checked={selectionCount.enabled}
              aria-label="スパース制約を使用する"
              onChange={(event) => updateSelectionCount({
                ...selectionCount,
                enabled: event.target.checked
              })}
            />
            <span>使用する</span>
          </label>
        </div>

        {selectionCount.enabled && (
          <div className="selection-count-editor">
            <div className="constraint-variable-picker" role="group" aria-label="スパース制約の候補変数">
              {numericVariables.map((variable) => {
                const selected = selectionCount.variables.includes(variable.name);
                return (
                  <button
                    key={variable.name}
                    type="button"
                    className={selected ? "selected" : ""}
                    aria-pressed={selected}
                    onClick={() => toggleSelectionVariable(variable.name)}
                  >
                    {variable.name}
                  </button>
                );
              })}
            </div>
            <label className="selection-count-input">
              <span>採用する変数数</span>
              <input
                type="number"
                min={1}
                max={Math.max(selectionCount.variables.length, 1)}
                value={selectionCount.k}
                onChange={(event) => updateSelectionCount({
                  ...selectionCount,
                  k: Math.max(1, Math.min(Number(event.target.value), Math.max(selectionCount.variables.length, 1)))
                })}
              />
              <small>選択候補 {selectionCount.variables.length} 変数中</small>
            </label>
          </div>
        )}
      </section>

      <p className="settings-note">
        「&gt;」「&lt;」は数値最適化上、それぞれ「以上」「以下」として扱います。有効変数数制約では0を未採用として扱います。
      </p>
    </article>
  );
}
