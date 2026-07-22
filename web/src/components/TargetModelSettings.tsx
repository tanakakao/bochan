import { getColumnClassValues } from "../targetSettingUtils";
import type {
  ColumnProfile,
  TargetClassValue,
  TargetGoal,
  TargetSetting,
  TaskType
} from "../types";

interface Props {
  columns: ColumnProfile[];
  preview: Record<string, unknown>[];
  targetColumns: string[];
  targetSettings: Record<string, TargetSetting>;
  patchTargetSetting: (target: string, patch: Partial<TargetSetting>) => void;
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

/** Defines target tasks and class/rank encodings used while fitting the model. */
export default function TargetModelSettings({
  columns,
  preview,
  targetColumns,
  targetSettings,
  patchTargetSetting
}: Props) {
  function classesFor(target: string): TargetClassValue[] {
    const column = columns.find((candidate) => candidate.name === target);
    return column ? getColumnClassValues(column, preview) : [];
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
      return;
    }
    if (nextTask === "classification") {
      const initialClass = classes.length === 2 ? classes[1] : classes[0];
      patchTargetSetting(target, {
        ...common,
        target_class: classes.length === 2 ? initialClass ?? null : null,
        target_classes: initialClass === undefined ? [] : [initialClass]
      });
      return;
    }
    patchTargetSetting(target, { ...common, class_order: [...classes] });
  }

  function classModelControl(target: string, setting: TargetSetting, classes: TargetClassValue[]) {
    if (setting.task_type === "regression") {
      return <span className="muted-cell">連続値</span>;
    }
    if (setting.task_type === "classification") {
      if (classes.length === 2) {
        return (
          <label className="table-field">
            <span>1として扱うクラス（Binary）</span>
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
        <div className="multiclass-summary">
          <span className="status-chip success">Multiclass</span>
          <strong>{classes.length} classes</strong>
          <small>{classes.map(String).join(" / ")}</small>
        </div>
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

  return (
    <article className="panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">1 · TARGET MODEL</span>
          <h3>目的変数とタスク</h3>
          <p>モデル学習に必要なタスク種別、Binaryの1クラス、順序回帰のクラス順を定義します。</p>
        </div>
        <span className="status-chip success">{targetColumns.length} targets</span>
      </div>
      <div className="table-wrap target-model-settings-wrap">
        <table className="target-model-settings-table">
          <thead>
            <tr><th>目的変数</th><th>タスク</th><th>モデル上のクラス設定／順序</th></tr>
          </thead>
          <tbody>
            {targetColumns.map((target) => {
              const column = columns.find((candidate) => candidate.name === target);
              const setting = targetSettings[target];
              if (!column || !setting) return null;
              const classes = classesFor(target);
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
                  <td className="class-config-cell">{classModelControl(target, setting, classes)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="settings-note">
        多値分類は自動的にMulticlassとして学習します。どのクラスを候補提案で狙うかは「候補提案」画面で設定します。
      </p>
    </article>
  );
}
