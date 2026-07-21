import { useEffect, useMemo, useState } from "react";
import type { SearchVariable } from "../types";
import {
  loadFeatureConstraints,
  newConstraintId,
  saveFeatureConstraints,
  type FeatureConstraint,
  type FeatureConstraintOperator
} from "../webRunSettings";

interface Props {
  variables: SearchVariable[];
}

/** Edits single-term linear constraints on numeric explanatory variables. */
export default function FeatureConstraints({ variables }: Props) {
  const numericVariables = useMemo(
    () => variables.filter((variable) => variable.type === "numeric"),
    [variables]
  );
  const numericNames = useMemo(
    () => new Set(numericVariables.map((variable) => variable.name)),
    [numericVariables]
  );
  const [constraints, setConstraints] = useState<FeatureConstraint[]>(() => loadFeatureConstraints());

  useEffect(() => {
    setConstraints((current) => {
      const next = current.filter((constraint) => numericNames.has(constraint.variable));
      saveFeatureConstraints(next);
      return next;
    });
  }, [numericNames]);

  function update(next: FeatureConstraint[]) {
    setConstraints(next);
    saveFeatureConstraints(next);
  }

  function addConstraint() {
    const variable = numericVariables[0];
    if (!variable) return;
    update([
      ...constraints,
      {
        id: newConstraintId(),
        variable: variable.name,
        coefficient: 1,
        operator: "<",
        value: variable.upper ?? 0
      }
    ]);
  }

  function patchConstraint(id: string, patch: Partial<FeatureConstraint>) {
    update(constraints.map((constraint) => (
      constraint.id === id ? { ...constraint, ...patch } : constraint
    )));
  }

  function removeConstraint(id: string) {
    update(constraints.filter((constraint) => constraint.id !== id));
  }

  return (
    <article className="panel feature-constraint-panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">3 · CONSTRAINTS</span>
          <h3>制約</h3>
          <p>説明変数に対する線形制約を設定します。各行は「係数 × 説明変数 演算子 値」を表します。</p>
        </div>
        <span className={`status-chip ${constraints.length ? "success" : ""}`}>
          {constraints.length} constraints
        </span>
      </div>

      <div className="button-row constraint-actions">
        <button type="button" className="secondary" disabled={!numericVariables.length} onClick={addConstraint}>
          制約を追加
        </button>
        <button type="button" className="secondary" disabled={!constraints.length} onClick={() => update([])}>
          全削除
        </button>
      </div>

      {!numericVariables.length ? (
        <p className="settings-note">数値型の説明変数がないため、線形制約は追加できません。</p>
      ) : constraints.length === 0 ? (
        <div className="constraint-empty">制約は設定されていません。</div>
      ) : (
        <div className="constraint-list">
          {constraints.map((constraint, index) => (
            <div className="constraint-row" key={constraint.id}>
              <span className="constraint-index">{index + 1}</span>
              <input
                type="number"
                aria-label={`制約${index + 1}の係数`}
                value={constraint.coefficient}
                onChange={(event) => patchConstraint(constraint.id, { coefficient: Number(event.target.value) })}
              />
              <span className="constraint-times">×</span>
              <select
                aria-label={`制約${index + 1}の説明変数`}
                value={constraint.variable}
                onChange={(event) => patchConstraint(constraint.id, { variable: event.target.value })}
              >
                {numericVariables.map((variable) => (
                  <option key={variable.name} value={variable.name}>{variable.name}</option>
                ))}
              </select>
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
              <button
                type="button"
                className="constraint-delete"
                aria-label={`制約${index + 1}を削除`}
                title="この制約を削除"
                onClick={() => removeConstraint(constraint.id)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      <p className="settings-note">
        「&gt;」「&lt;」は数値最適化上、それぞれ「以上」「以下」として扱います。複数変数の和を制約したい場合は、今後1つの制約に複数項を追加できる形式へ拡張できます。
      </p>
    </article>
  );
}
