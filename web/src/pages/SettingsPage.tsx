import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import { getColumnClassValues } from "../targetSettingUtils";
import type {
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

/** Configures target tasks/constraints and the complete search-variable definition. */
export default function SettingsPage() {
  const {
    dataset,
    columns,
    targetColumns,
    targetSettings,
    patchTargetSetting,
    selectedVariables,
    patchVariable,
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

  function classesFor(target: string): TargetClassValue[] {
    const column = columns.find((candidate) => candidate.name === target);
    return column ? getColumnClassValues(column, dataset.preview) : [];
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

  function changeTask(target: string, nextTask: TaskType) {
    const column = columns.find((candidate) => candidate.name === target);
    if (!column) return;
    const classes = classesFor(target);

    if (nextTask === "regression") {
      patchTargetSetting(target, {
        task_type: nextTask,
        goal: "none",
        value: null,
        target_class: null,
        target_classes: [],
        class_order: [],
        target_values: []
      });
      return;
    }

    if (nextTask === "classification") {
      const initialClass = classes.length === 2 ? classes[1] : classes[0];
      patchTargetSetting(target, {
        task_type: nextTask,
        goal: "none",
        value: null,
        target_class: classes.length === 2 ? initialClass ?? null : null,
        target_classes: initialClass === undefined ? [] : [initialClass],
        class_order: [],
        target_values: []
      });
      return;
    }

    patchTargetSetting(target, {
      task_type: nextTask,
      goal: "none",
      value: null,
      target_class: null,
      target_classes: [],
      class_order: [...classes],
      target_values: []
    });
  }

  function changeGoal(target: string, nextGoal: TargetGoal) {
    const column = columns.find((candidate) => candidate.name === target);
    const setting = targetSettings[target];
    if (!column || !setting) return;
    const classes = setting.class_order?.length ? setting.class_order : classesFor(target);

    if (nextGoal === "none") {
      patchTargetSetting(target, { goal: nextGoal, value: null, target_values: [] });
      return;
    }
    if (setting.task_type === "regression") {
      patchTargetSetting(target, {
        goal: nextGoal,
        value: column.mean ?? column.min ?? 0,
        target_values: []
      });
      return;
    }
    if (setting.task_type === "classification") {
      patchTargetSetting(target, {
        goal: nextGoal,
        value: nextGoal === "above" || nextGoal === "below" ? 0.5 : null,
        target_values: []
      });
      return;
    }
    if (nextGoal === "target") {
      patchTargetSetting(target, {
        goal: nextGoal,
        value: null,
        target_values: classes.length ? [classes[0]] : []
      });
      return;
    }
    patchTargetSetting(target, {
      goal: nextGoal,
      value: classes[0] ?? null,
      target_values: []
    });
  }

  function targetValueControl(target: string, setting: TargetSetting, classes: TargetClassValue[]) {
    if (setting.goal === "none") return <span className="muted-cell">制約なし</span>;

    if (setting.task_type === "regression") {
      return (
        <input
          type="number"
          value={setting.value ?? ""}
          onChange={(event) => patchTargetSetting(target, { value: numberOrUndefined(event.target.value) ?? null })}
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
          onChange={(event) => patchTargetSetting(target, { value: numberOrUndefined(event.target.value) ?? null })}
        />
      );
    }

    if (setting.goal === "target") {
      return (
        <select
          multiple
          size={Math.min(Math.max(classes.length, 2), 5)}
          value={(setting.target_values ?? []).map(String)}
          onChange={(event) => patchTargetSetting(target, { target_values: selectedValues(event.target) })}
        >
          {classes.map((value) => <option key={String(value)} value={String(value)}>{String(value)}</option>)}
        </select>
      );
    }

    return (
      <select
        value={String(setting.value ?? "")}
        onChange={(event) => patchTargetSetting(target, { value: event.target.value })}
      >
        {classes.map((value) => <option key={String(value)} value={String(value)}>{String(value)}</option>)}
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
              {classes.map((value) => <option key={String(value)} value={String(value)}>{String(value)}</option>)}
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
            {classes.map((value) => <option key={String(value)} value={String(value)}>{String(value)}</option>)}
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
            >
              ↑
            </button>
            <button
              type="button"
              className="secondary order-button"
              disabled={index === order.length - 1}
              onClick={() => patchTargetSetting(target, { class_order: moveItem(order, index, 1) })}
              aria-label={`${String(value)}を下へ移動`}
            >
              ↓
            </button>
          </div>
        ))}
      </div>
    );
  }

  return (
    <>
      <SectionHeader
        step="3 · SETTINGS"
        title="目的変数と探索変数を設定する"
        text="目的変数はタスクとクラスを指定します。制約は任意で、設定しない目的変数もそのまま最適化対象として利用できます。"
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
            <p>探索変数と同じ表形式で、タスク、任意の制約、ターゲットクラス、順序を設定します。</p>
          </div>
          <span className="status-chip success">{targetColumns.length} targets</span>
        </div>

        <div className="table-wrap target-settings-wrap">
          <table className="target-settings-table">
            <thead>
              <tr>
                <th>目的変数</th>
                <th>タスク</th>
                <th>制約</th>
                <th>しきい値／目標</th>
                <th>クラス設定／順序</th>
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
                  <tr key={target}>
                    <td className="target-name-cell">
                      <strong>{target}</strong>
                      <span>{taskLabel(setting.task_type)}</span>
                    </td>
                    <td>
                      <select
                        value={setting.task_type}
                        onChange={(event) => changeTask(target, event.target.value as TaskType)}
                      >
                        <option value="regression" disabled={column.kind !== "numeric"}>回帰</option>
                        <option value="classification">分類</option>
                        <option value="ordinal">順序回帰</option>
                      </select>
                    </td>
                    <td>
                      <select
                        value={setting.goal}
                        onChange={(event) => changeGoal(target, event.target.value as TargetGoal)}
                      >
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
          分類の以上／以下は、選択したターゲットクラス群の合計予測確率に対する制約です。順序回帰の目標値は複数クラスを選択できます。
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
              <tr>
                <th>変数</th><th>カテゴリ?</th><th>型</th><th>下限</th><th>上限</th><th>刻み</th><th>固定</th><th>固定値</th>
              </tr>
            </thead>
            <tbody>
              {selectedVariables.map((variable) => {
                const detectedCategorical = columns.find((column) => column.name === variable.name)?.kind === "categorical";
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

      {!settingsValid && (
        <article className="panel compact-panel">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">VALIDATION</span>
              <h3>設定を確認してください</h3>
              <p>分類ではターゲットクラス、順序回帰では全クラスの順序、数値探索変数では有効な上下限が必要です。制約自体は「なし」で構いません。</p>
            </div>
            <span className="status-chip warning">Not ready</span>
          </div>
        </article>
      )}
    </>
  );
}
