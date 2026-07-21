import { EmptyState, SectionHeader } from "../components/Common";
import FeatureConstraints from "../components/FeatureConstraints";
import { useWorkbench } from "../context/WorkbenchContext";
import { getColumnClassValues } from "../targetSettingUtils";
import type {
  Direction,
  SearchVariable,
  TargetClassValue,
  TargetGoal,
  TargetSetting,
  TaskType
} from "../types";

function selectedValues(select: HTMLSelectElement): string[] {
  return Array.from(select.selectedOptions, (option) => option.value);
}

function moveItem(values: TargetClassValue[], index: number, offset: number): TargetClassValue[] {
  const nextIndex = index + offset;
  if (nextIndex < 0 || nextIndex >= values.length) return values;
  const next = [...values];
  [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
  return next;
}

function taskLabel(taskType: TaskType): string {
  if (taskType === "classification") return "分類";
  if (taskType === "ordinal") return "順序回帰";
  return "回帰";
}

function numericInputStep(variable: SearchVariable): number | "any" {
  return Number(variable.upper) === 1 ? 0.1 : "any";
}

/** Configures targets, search variables, transforms, and explanatory-variable constraints. */
export default function SettingsPage() {
  const {
    dataset,
    columns,
    targetColumns,
    targetSettings,
    patchTargetSetting,
    selectedVariables,
    patchVariable,
    normalize,
    setNormalize,
    inputPerturbation,
    setInputPerturbation,
    nW,
    setNW,
    perturbationStd,
    setPerturbationStd,
    settingsValid,
    setStep,
    numberOrUndefined
  } = useWorkbench();

  if (!dataset || targetColumns.length === 0 || selectedVariables.length === 0) {
    return (
      <>
        <SectionHeader
          step="3 · SETTINGS"
          title="目的変数と探索変数を設定する"
          text="先にSelectページで目的変数と説明変数を選択してください。"
        />
        <EmptyState>設定対象の変数が選択されていません。</EmptyState>
      </>
    );
  }

  const preview = dataset.preview;

  function classesFor(target: string): TargetClassValue[] {
    const column = columns.find((candidate) => candidate.name === target);
    return column ? getColumnClassValues(column, preview) : [];
  }

  function categoriesForVariable(name: string): TargetClassValue[] {
    const column = columns.find((candidate) => candidate.name === name);
    return column ? getColumnClassValues(column, preview) : [];
  }

  function setVariableType(variable: SearchVariable, categorical: boolean) {
    const nextType = categorical ? "categorical" : "numeric";
    const column = columns.find((candidate) => candidate.name === variable.name);
    patchVariable(variable.name, {
      type: nextType,
      fixed: false,
      fixed_value: undefined,
      categories: nextType === "categorical" ? categoriesForVariable(variable.name) : undefined,
      lower: nextType === "numeric" ? variable.lower ?? column?.min ?? undefined : undefined,
      upper: nextType === "numeric" ? variable.upper ?? column?.max ?? undefined : undefined,
      step: nextType === "categorical" ? undefined : variable.step
    });
  }

  function changeTask(target: string, nextTask: TaskType) {
    const column = columns.find((candidate) => candidate.name === target);
    const current = targetSettings[target];
    if (!column || !current) return;
    const classes = classesFor(target);
    const common = {
      task_type: nextTask,
      optimize: current.optimize,
      direction: current.direction,
      goal: "none" as TargetGoal,
      value: null,
      target_class: null,
      target_classes: [] as TargetClassValue[],
      class_order: [] as TargetClassValue[],
      target_values: [] as TargetClassValue[]
    };

    if (nextTask === "regression") {
      patchTargetSetting(target, common);
    } else if (nextTask === "classification") {
      const initialClass = classes.length === 2 ? classes[1] : classes[0];
      patchTargetSetting(target, {
        ...common,
        target_class: classes.length === 2 ? initialClass ?? null : null,
        target_classes: initialClass === undefined ? [] : [initialClass]
      });
    } else {
      patchTargetSetting(target, { ...common, class_order: [...classes] });
    }
  }

  function changeGoal(target: string, nextGoal: TargetGoal) {
    const column = columns.find((candidate) => candidate.name === target);
    const setting = targetSettings[target];
    if (!column || !setting) return;
    const classes = setting.class_order?.length ? setting.class_order : classesFor(target);

    if (nextGoal === "none") {
      patchTargetSetting(target, { goal: nextGoal, value: null, target_values: [] });
    } else if (setting.task_type === "regression") {
      patchTargetSetting(target, {
        goal: nextGoal,
        optimize: nextGoal === "target" ? true : setting.optimize,
        direction: nextGoal === "target" ? "maximize" : setting.direction,
        value: column.mean ?? column.min ?? 0,
        target_values: []
      });
    } else if (setting.task_type === "classification") {
      patchTargetSetting(target, {
        goal: nextGoal,
        value: nextGoal === "above" || nextGoal === "below" ? 0.5 : null,
        target_values: []
      });
    } else if (nextGoal === "target") {
      patchTargetSetting(target, {
        goal: nextGoal,
        optimize: true,
        direction: "maximize",
        value: null,
        target_values: classes.length ? [classes[0]] : []
      });
    } else {
      patchTargetSetting(target, {
        goal: nextGoal,
        value: classes[0] ?? null,
        target_values: []
      });
    }
  }

  function targetValueControl(target: string, setting: TargetSetting, classes: TargetClassValue[]) {
    if (setting.goal === "none") return <span className="muted-cell">制約なし</span>;
    if (setting.task_type === "regression") {
      return (
        <input
          type="number"
          step="any"
          value={setting.value ?? ""}
          onChange={(event) => patchTargetSetting(target, {
            value: numberOrUndefined(event.target.value) ?? null
          })}
        />
      );
    }
    if (setting.task_type === "classification") {
      return (
        <input
          type="number"
          min={0}
          max={1}
          step={0.01}
          value={setting.value ?? ""}
          onChange={(event) => patchTargetSetting(target, {
            value: numberOrUndefined(event.target.value) ?? null
          })}
        />
      );
    }
    if (setting.goal === "target") {
      return (
        <select
          multiple
          size={Math.min(Math.max(classes.length, 2), 5)}
          value={(setting.target_values ?? []).map(String)}
          onChange={(event) => patchTargetSetting(target, {
            target_values: selectedValues(event.target)
          })}
        >
          {classes.map((value) => (
            <option key={String(value)} value={String(value)}>{String(value)}</option>
          ))}
        </select>
      );
    }
    return (
      <select
        value={String(setting.value ?? "")}
        onChange={(event) => patchTargetSetting(target, { value: event.target.value })}
      >
        {classes.map((value) => (
          <option key={String(value)} value={String(value)}>{String(value)}</option>
        ))}
      </select>
    );
  }

  function classControl(target: string, setting: TargetSetting, classes: TargetClassValue[]) {
    if (setting.task_type === "regression") return <span className="muted-cell">—</span>;
    if (setting.task_type === "classification") {
      if (classes.length === 2) {
        return (
          <label className="table-field">
            <span>target_class</span>
            <select
              value={String(setting.target_class ?? "")}
              onChange={(event) => patchTargetSetting(target, {
                target_class: event.target.value,
                target_classes: [event.target.value]
              })}
            >
              {classes.map((value) => (
                <option key={String(value)} value={String(value)}>{String(value)}</option>
              ))}
            </select>
          </label>
        );
      }
      return (
        <label className="table-field">
          <span>ターゲットクラス（複数可）</span>
          <select
            multiple
            size={Math.min(Math.max(classes.length, 2), 6)}
            value={(setting.target_classes ?? []).map(String)}
            onChange={(event) => patchTargetSetting(target, {
              target_class: null,
              target_classes: selectedValues(event.target)
            })}
          >
            {classes.map((value) => (
              <option key={String(value)} value={String(value)}>{String(value)}</option>
            ))}
          </select>
        </label>
      );
    }

    const order = setting.class_order?.length ? setting.class_order : classes;
    return (
      <div className="class-order-editor">
        {order.map((value, index) => (
          <div className="class-order-item" key={String(value)}>
            <span className="order-index">{index + 1}</span>
            <strong>{String(value)}</strong>
            <button
              type="button"
              className="secondary order-button"
              disabled={index === 0}
              onClick={() => patchTargetSetting(target, { class_order: moveItem(order, index, -1) })}
              aria-label={`${String(value)}を上へ移動`}
            >↑</button>
            <button
              type="button"
              className="secondary order-button"
              disabled={index === order.length - 1}
              onClick={() => patchTargetSetting(target, { class_order: moveItem(order, index, 1) })}
              aria-label={`${String(value)}を下へ移動`}
            >↓</button>
          </div>
        ))}
      </div>
    );
  }

  function directionControl(target: string, setting: TargetSetting) {
    if (setting.goal === "target") return <span className="direction-note">目標へ近づける</span>;
    if (!setting.optimize) return <span className="muted-cell">対象外</span>;
    return (
      <select
        value={setting.direction}
        onChange={(event) => patchTargetSetting(target, {
          direction: event.target.value as Direction
        })}
      >
        <option value="maximize">最大化</option>
        <option value="minimize">最小化</option>
      </select>
    );
  }

  return (
    <>
      <SectionHeader
        step="3 · SETTINGS"
        title="目的変数と探索変数を設定する"
        text="目的変数、探索範囲、入力変換、説明変数の制約を設定します。"
        action={
          <button disabled={!settingsValid} onClick={() => setStep("optimize")}>
            モデル設定へ
          </button>
        }
      />

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">1 · TARGET SETTINGS</span>
            <h3>目的変数の設定</h3>
            <p>タスク、最適化対象、方向、任意の制約、ターゲットクラス、順序を設定します。</p>
          </div>
          <span className="status-chip success">{targetColumns.length} targets</span>
        </div>
        <div className="table-wrap target-settings-wrap">
          <table className="target-settings-table">
            <thead>
              <tr>
                <th>目的変数</th><th>最適化対象</th><th>タスク</th><th>方向</th>
                <th>制約</th><th>しきい値／目標</th><th>クラス設定／順序</th>
              </tr>
            </thead>
            <tbody>
              {targetColumns.map((target) => {
                const column = columns.find((candidate) => candidate.name === target);
                const setting = targetSettings[target];
                if (!column || !setting) return null;
                const classes = classesFor(target);
                const orderedClasses = setting.class_order?.length ? setting.class_order : classes;
                return (
                  <tr key={target} className={setting.optimize ? "objective-row" : "constraint-only-row"}>
                    <td className="target-name-cell"><strong>{target}</strong><span>{taskLabel(setting.task_type)}</span></td>
                    <td>
                      <input
                        className="table-checkbox"
                        type="checkbox"
                        checked={setting.optimize}
                        disabled={setting.goal === "target"}
                        title={setting.goal === "target" ? "目標値は最適化目的として扱います。" : "制約専用にする場合はチェックを外します。"}
                        onChange={(event) => patchTargetSetting(target, { optimize: event.target.checked })}
                      />
                    </td>
                    <td>
                      <select value={setting.task_type} onChange={(event) => changeTask(target, event.target.value as TaskType)}>
                        <option value="regression" disabled={column.kind !== "numeric"}>回帰</option>
                        <option value="classification">分類</option>
                        <option value="ordinal">順序回帰</option>
                      </select>
                    </td>
                    <td>{directionControl(target, setting)}</td>
                    <td>
                      <select value={setting.goal} onChange={(event) => changeGoal(target, event.target.value as TargetGoal)}>
                        <option value="none">なし</option>
                        <option value="above">以上</option>
                        <option value="below">以下</option>
                        {setting.task_type !== "classification" && <option value="target">目標値</option>}
                      </select>
                    </td>
                    <td>{targetValueControl(target, setting, orderedClasses)}</td>
                    <td className="class-config-cell">{classControl(target, setting, classes)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="settings-note">
          「方向」は最適化の向き、「以上／以下」は実行可能性の制約です。分類制約は選択クラスの予測確率に対して適用します。
        </p>
      </article>

      <article className="panel">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">2 · SEARCH VARIABLES</span>
            <h3>探索変数</h3>
            <p>観測範囲を初期値として、カテゴリ扱い・下限・上限・刻み・固定値を編集します。</p>
          </div>
          <span className="status-chip success">{selectedVariables.length} variables</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>変数</th><th>カテゴリ?</th><th>型</th><th>下限</th><th>上限</th><th>刻み</th><th>固定</th><th>固定値</th></tr>
            </thead>
            <tbody>
              {selectedVariables.map((variable) => {
                const column = columns.find((candidate) => candidate.name === variable.name);
                const detectedCategorical = column?.kind === "categorical";
                const inputStep = numericInputStep(variable);
                const categories = variable.categories?.length
                  ? variable.categories
                  : categoriesForVariable(variable.name);
                return (
                  <tr key={variable.name}>
                    <td><strong>{variable.name}</strong></td>
                    <td><input className="table-checkbox" type="checkbox" checked={variable.type === "categorical"} disabled={detectedCategorical} onChange={(event) => setVariableType(variable, event.target.checked)} /></td>
                    <td><span className="status-chip">{variable.type}</span></td>
                    <td>{variable.type === "numeric" ? <input type="number" step={inputStep} value={variable.lower ?? ""} onChange={(event) => patchVariable(variable.name, { lower: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                    <td>{variable.type === "numeric" ? <input type="number" step={inputStep} value={variable.upper ?? ""} onChange={(event) => patchVariable(variable.name, { upper: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                    <td>{variable.type === "numeric" ? <input type="number" min={0} step={inputStep} value={variable.step ?? ""} placeholder="任意" onChange={(event) => patchVariable(variable.name, { step: numberOrUndefined(event.target.value) })} /> : "—"}</td>
                    <td><input className="table-checkbox" type="checkbox" checked={variable.fixed} onChange={(event) => patchVariable(variable.name, { fixed: event.target.checked, fixed_value: event.target.checked ? variable.fixed_value : undefined })} /></td>
                    <td>
                      {variable.fixed && variable.type === "categorical" ? (
                        <select value={String(variable.fixed_value ?? "")} onChange={(event) => patchVariable(variable.name, { fixed_value: event.target.value })}>
                          <option value="">選択</option>
                          {categories.map((category) => <option key={String(category)} value={String(category)}>{String(category)}</option>)}
                        </select>
                      ) : variable.fixed ? (
                        <input type="number" step={inputStep} value={variable.fixed_value ?? ""} onChange={(event) => patchVariable(variable.name, { fixed_value: numberOrUndefined(event.target.value) })} />
                      ) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="search-transform-grid">
          <section className="transform-card">
            <div className="transform-card-heading">
              <div><span className="panel-kicker">NORMALIZATION</span><h4>正規化</h4></div>
              <label className="switch-field">
                <input type="checkbox" checked={normalize} onChange={(event) => setNormalize(event.target.checked)} />
                <span>{normalize ? "使用する" : "使用しない"}</span>
              </label>
            </div>
            <p>各探索変数に設定した下限・上限を使って入力を正規化します。デフォルトは有効です。</p>
          </section>

          <section className="transform-card">
            <div className="transform-card-heading">
              <div><span className="panel-kicker">INPUT PERTURBATION</span><h4>入力摂動</h4></div>
              <label className="switch-field">
                <input type="checkbox" checked={inputPerturbation} onChange={(event) => setInputPerturbation(event.target.checked)} />
                <span>{inputPerturbation ? "使用する" : "使用しない"}</span>
              </label>
            </div>
            <p>候補入力のばらつきをサンプリングし、頑健な候補評価へ反映します。デフォルトは無効です。</p>
            {inputPerturbation && (
              <div className="transform-fields">
                <label>摂動サンプル数 n<input type="number" min={1} step={1} value={nW} onChange={(event) => setNW(Number(event.target.value))} /></label>
                <label>ばらつき（標準偏差）<input type="number" min={0.000001} step={0.01} value={perturbationStd} onChange={(event) => setPerturbationStd(Number(event.target.value))} /></label>
              </div>
            )}
          </section>
        </div>
      </article>

      <FeatureConstraints variables={selectedVariables} />

      {!settingsValid && (
        <article className="panel compact-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">VALIDATION</span>
              <h3>設定を確認してください</h3>
              <p>少なくとも1つの最適化対象、有効な探索範囲、カテゴリ候補、入力摂動のサンプル数と標準偏差を確認してください。</p>
            </div>
            <span className="status-chip warning">Not ready</span>
          </div>
        </article>
      )}
    </>
  );
}
