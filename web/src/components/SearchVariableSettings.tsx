import { getColumnClassValues } from "../targetSettingUtils";
import type { ColumnProfile, SearchVariable, TargetClassValue } from "../types";

interface Props {
  columns: ColumnProfile[];
  preview: Record<string, unknown>[];
  variables: SearchVariable[];
  patchVariable: (name: string, patch: Partial<SearchVariable>) => void;
  numberOrUndefined: (value: string) => number | undefined;
}

/** Edits candidate-search bounds, steps, and fixed values without changing feature types. */
export default function SearchVariableSettings({
  columns,
  preview,
  variables,
  patchVariable,
  numberOrUndefined
}: Props) {
  function categoriesForVariable(name: string): TargetClassValue[] {
    const variable = variables.find((candidate) => candidate.name === name);
    if (variable?.categories?.length) return variable.categories;
    const column = columns.find((candidate) => candidate.name === name);
    return column ? getColumnClassValues(column, preview) : [];
  }

  return (
    <article className="panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">2 · SEARCH SPACE</span>
          <h3>説明変数の探索範囲</h3>
          <p>候補生成で使用する下限・上限・刻み・固定値を設定します。型はSelect画面で変更します。</p>
        </div>
        <span className="status-chip success">{variables.length} variables</span>
      </div>
      <div className="table-wrap">
        <table className="search-variable-table">
          <thead>
            <tr><th>変数</th><th>型</th><th>下限</th><th>上限</th><th>刻み</th><th>固定</th><th>固定値</th></tr>
          </thead>
          <tbody>
            {variables.map((variable) => {
              const categories = categoriesForVariable(variable.name);
              return (
                <tr key={variable.name}>
                  <td><strong>{variable.name}</strong></td>
                  <td><span className={`status-chip ${variable.type === "categorical" ? "categorical-chip" : ""}`}>{variable.type}</span></td>
                  <td>{variable.type === "numeric" ? (
                    <input
                      type="number"
                      step="any"
                      value={variable.lower ?? ""}
                      onChange={(event) => patchVariable(variable.name, { lower: numberOrUndefined(event.target.value) })}
                    />
                  ) : "—"}</td>
                  <td>{variable.type === "numeric" ? (
                    <input
                      type="number"
                      step="any"
                      value={variable.upper ?? ""}
                      onChange={(event) => patchVariable(variable.name, { upper: numberOrUndefined(event.target.value) })}
                    />
                  ) : "—"}</td>
                  <td>{variable.type === "numeric" ? (
                    <input
                      type="number"
                      min={0}
                      step="any"
                      value={variable.step ?? ""}
                      placeholder="任意"
                      onChange={(event) => patchVariable(variable.name, { step: numberOrUndefined(event.target.value) })}
                    />
                  ) : "—"}</td>
                  <td>
                    <input
                      className="table-checkbox"
                      type="checkbox"
                      checked={variable.fixed}
                      onChange={(event) => patchVariable(variable.name, {
                        fixed: event.target.checked,
                        fixed_value: event.target.checked ? variable.fixed_value : undefined
                      })}
                    />
                  </td>
                  <td>
                    {variable.fixed && variable.type === "categorical" ? (
                      <select
                        value={String(variable.fixed_value ?? "")}
                        onChange={(event) => patchVariable(variable.name, { fixed_value: event.target.value })}
                      >
                        <option value="">選択</option>
                        {categories.map((category) => (
                          <option key={String(category)} value={String(category)}>{String(category)}</option>
                        ))}
                      </select>
                    ) : variable.fixed ? (
                      <input
                        type="number"
                        step="any"
                        value={variable.fixed_value ?? ""}
                        onChange={(event) => patchVariable(variable.name, {
                          fixed_value: numberOrUndefined(event.target.value)
                        })}
                      />
                    ) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </article>
  );
}
