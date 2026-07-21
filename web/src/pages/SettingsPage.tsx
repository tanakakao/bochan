import { EmptyState, SectionHeader } from "../components/Common";
import { useWorkbench } from "../context/WorkbenchContext";
import type { SearchVariable, TargetGoal, TaskType } from "../types";

function taskLabel(taskType: TaskType): string {
  if (taskType === "classification") return "分類";
  if (taskType === "ordinal") return "順序回帰";
  return "回帰";
}

function goalLabel(goal: TargetGoal): string {
  if (goal === "below") return "以下";
  if (goal === "target") return "目標値";
  return "以上";
}

/** Configures one target rule per target and the full search-variable definition. */
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
    const current = targetSettings[target];
    if (!column || !current) return;
    if (nextTask === "regression") {
      patchTargetSetting(target, {
        task_type: nextTask,
        goal: "above",
        value: column.mean ?? column.min ?? 0
      });
      return;
    }
    if (nextTask === "classification") {
      patchTargetSetting(target, {
        task_type: nextTask,
        goal: "target",
        value: column.values?.[0] ?? column.min ?? 1
      });
      return;
    }
    patchTargetSetting(target, {
      task_type: nextTask,
      goal: "above",
      value: column.values?.[0] ?? column.min ?? 0
    });
  }

  function changeGoal(target: string, nextGoal: TargetGoal) {
    const column = columns.find((candidate) => candidate.name === target);
    const setting = targetSettings[target];
    if (!column || !setting) return;
    if (setting.task_type === "classification" && column.unique_count > 2 && nextGoal !== "target") {
      return;
    }
    let value: string | number;
    if (setting.task_type === "classification") {
      value = nextGoal === "target" ? (column.values?.[0] ?? column.min ?? 1) : 0.5;
    } else if (setting.task_type === "ordinal") {
      value = column.values?.[0] ?? column.min ?? 0;
    } else {
      value = column.mean ?? column.min ?? 0;
    }
    patchTargetSetting(target, { goal: nextGoal, value });
  }

  return (
    <>
      <SectionHeader
        step="3 · SETTINGS"
        title="目的変数と探索変数を設定する"
        text="目的変数には1列につき1つのタスク・条件を設定し、説明変数には探索範囲と固定条件を設定します。"
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
            <p>各目的変数に対して、タスク種別と「以上・以下・目標値」のいずれか1つを設定します。</p>
          </div>
          <span className="status-chip success">{targetColumns.length} targets</span>
        </div>

        <div className="cards">
          {targetColumns.map((target) => {
            const column = columns.find((candidate) => candidate.name === target);
            const setting = targetSettings[target];
            if (!column || !setting) return null;
            const categoryValues = column.values ?? [];
            const multiclass = setting.task_type === "classification" && column.unique_count > 2;
            const useValueSelector =
              categoryValues.length > 0 &&
              (setting.task_type === "ordinal" ||
                (setting.task_type === "classification" && setting.goal === "target"));
            return (
              <article className="panel compact-panel" key={target}>
                <div className="panel-title">
                  <div>
                    <span className="panel-kicker">TARGET</span>
                    <h3>{target}</h3>
                    <p>{column.kind} · {column.unique_count} unique</p>
                  </div>
                  <span className="status-chip">{taskLabel(setting.task_type)}</span>
                </div>

                <div className="form-grid candidate-settings">
                  <label>
                    タスク種別
                    <select
                      value={setting.task_type}
                      onChange={(event) => changeTask(target, event.target.value as TaskType)}
                    >
                      <option value="regression" disabled={column.kind !== "numeric"}>回帰</option>
                      <option value="classification">分類</option>
                      <option value="ordinal">順序回帰</option>
                    </select>
                  </label>
                  <label>
                    条件
                    <select
                      value={setting.goal}
                      onChange={(event) => changeGoal(target, event.target.value as TargetGoal)}
                    >
                      <option value="above" disabled={multiclass}>以上</option>
                      <option value="below" disabled={multiclass}>以下</option>
                      <option value="target">目標値</option>
                    </select>
                  </label>
                  <label>
                    {setting.goal === "target" ? "目標値" : "しきい値"}
                    {useValueSelector ? (
                      <select
                        value={String(setting.value)}
                        onChange={(event) => patchTargetSetting(target, { value: event.target.value })}
                      >
                        {categoryValues.map((value) => <option key={value} value={value}>{value}</option>)}
                      </select>
                    ) : (
                      <input
                        type="number"
                        min={setting.task_type === "classification" && setting.goal !== "target" ? 0 : undefined}
                        max={setting.task_type === "classification" && setting.goal !== "target" ? 1 : undefined}
                        step={setting.task_type === "classification" && setting.goal !== "target" ? 0.01 : "any"}
                        value={setting.value}
                        onChange={(event) => patchTargetSetting(target, { value: Number(event.target.value) })}
                      />
                    )}
                  </label>
                </div>

                <p className="settings-note">
                  {setting.task_type === "classification" && setting.goal !== "target"
                    ? "二値分類の以上・以下では、昇順で2番目のクラス確率に対するしきい値として扱います。"
                    : null}
                  {multiclass
                    ? "3クラス以上の分類では、探索対象クラスを目標値として指定します。"
                    : null}
                  {setting.task_type === "classification" && setting.goal === "target"
                    ? "指定クラスの予測確率が高い候補を探索します。"
                    : null}
                  {setting.task_type === "ordinal"
                    ? "順序はデータ中のカテゴリ値または数値の昇順として解釈し、以上・以下は予測期待順位で判定します。"
                    : null}
                  {setting.task_type === "regression" && setting.goal === "target"
                    ? "予測値と目標値の絶対偏差が小さい候補を探索します。"
                    : null}
                </p>
                <p className="settings-note">
                  現在の設定: <strong>{taskLabel(setting.task_type)} · {goalLabel(setting.goal)} {String(setting.value)}</strong>
                </p>
              </article>
            );
          })}
        </div>
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
              <p>回帰目的は数値列、二値分類のしきい値は0〜1、数値探索変数は下限より上限を大きく設定してください。</p>
            </div>
            <span className="status-chip warning">Not ready</span>
          </div>
        </article>
      )}
    </>
  );
}
