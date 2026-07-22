import { getColumnClassValues } from "../targetSettingUtils";
import type {
  ColumnProfile,
  Direction,
  TargetClassValue,
  TargetGoal,
  TargetSetting
} from "../types";

type OptimizationDirection = Direction | "target";

interface Props {
  columns: ColumnProfile[];
  preview: Record<string, unknown>[];
  targetColumns: string[];
  targetSettings: Record<string, TargetSetting>;
  patchTargetSetting: (target: string, patch: Partial<TargetSetting>) => void;
  numberOrUndefined: (value: string) => number | undefined;
}

function selectedValues(select: HTMLSelectElement): string[] {
  return Array.from(select.selectedOptions, (option) => option.value);
}

/** Defines objective roles, directions, desired classes, and outcome constraints. */
export default function TargetProposalSettings({
  columns,
  preview,
  targetColumns,
  targetSettings,
  patchTargetSetting,
  numberOrUndefined
}: Props) {
  function classesFor(target: string): TargetClassValue[] {
    const column = columns.find((candidate) => candidate.name === target);
    return column ? getColumnClassValues(column, preview) : [];
  }

  function changeGoal(target: string, nextGoal: Exclude<TargetGoal, "target">) {
    const column = columns.find((candidate) => candidate.name === target);
    const setting = targetSettings[target];
    if (!column || !setting) return;
    const classes = setting.class_order?.length ? setting.class_order : classesFor(target);

    if (nextGoal === "none") {
      patchTargetSetting(target, { goal: nextGoal, value: null, target_values: [] });
    } else if (setting.task_type === "regression") {
      patchTargetSetting(target, {
        goal: nextGoal,
        value: column.mean ?? column.min ?? 0,
        target_values: []
      });
    } else if (setting.task_type === "classification") {
      patchTargetSetting(target, {
        goal: nextGoal,
        value: 0.5,
        target_values: []
      });
    } else {
      patchTargetSetting(target, {
        goal: nextGoal,
        value: classes[0] ?? null,
        target_values: []
      });
    }
  }

  function changeDirection(target: string, nextDirection: OptimizationDirection) {
    const column = columns.find((candidate) => candidate.name === target);
    const setting = targetSettings[target];
    if (!column || !setting) return;

    if (nextDirection === "target") {
      if (setting.task_type === "classification") return;
      const classes = setting.class_order?.length ? setting.class_order : classesFor(target);
      if (setting.task_type === "regression") {
        patchTargetSetting(target, {
          optimize: true,
          direction: "maximize",
          goal: "target",
          value: setting.goal === "target" ? setting.value : column.mean ?? column.min ?? 0,
          target_values: []
        });
      } else {
        patchTargetSetting(target, {
          optimize: true,
          direction: "maximize",
          goal: "target",
          value: null,
          target_values: setting.goal === "target" && setting.target_values?.length
            ? setting.target_values
            : classes.length
              ? [classes[0]]
              : []
        });
      }
      return;
    }

    patchTargetSetting(target, {
      direction: nextDirection,
      ...(setting.goal === "target" ? {
        goal: "none",
        value: null,
        target_values: []
      } : {})
    });
  }

  function directionControl(target: string, setting: TargetSetting) {
    if (!setting.optimize) return <span className="muted-cell">対象外</span>;
    const directionValue: OptimizationDirection = setting.goal === "target"
      ? "target"
      : setting.direction;
    return (
      <select
        value={directionValue}
        onChange={(event) => changeDirection(target, event.target.value as OptimizationDirection)}
      >
        <option value="maximize">最大化</option>
        <option value="minimize">最小化</option>
        {setting.task_type !== "classification" && <option value="target">目標値</option>}
      </select>
    );
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

  function desiredClassControl(target: string, setting: TargetSetting, classes: TargetClassValue[]) {
    if (setting.task_type === "regression") return <span className="muted-cell">—</span>;
    if (setting.task_type === "ordinal") return <span className="muted-cell">モデル設定の順序を使用</span>;
    if (classes.length === 2) {
      return (
        <div className="binary-target-summary">
          <span>1として扱うクラス</span>
          <strong>{String(setting.target_class ?? "—")}</strong>
        </div>
      );
    }
    return (
      <label className="table-field">
        <span>候補提案で狙うクラス（複数可）</span>
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

  return (
    <article className="panel">
      <div className="panel-title">
        <div>
          <span className="panel-kicker">1 · TARGET PROPOSAL</span>
          <h3>目的変数の候補提案条件</h3>
          <p>最適化対象、方向、目標値、実行可能性制約、狙うクラスを設定します。</p>
        </div>
        <span className="status-chip success">{targetColumns.length} targets</span>
      </div>
      <div className="table-wrap target-settings-wrap">
        <table className="target-settings-table proposal-target-table">
          <thead>
            <tr>
              <th>目的変数</th><th>最適化対象</th><th>方向</th><th>制約</th>
              <th>しきい値／目標値</th><th>対象クラス</th>
            </tr>
          </thead>
          <tbody>
            {targetColumns.map((target) => {
              const setting = targetSettings[target];
              if (!setting) return null;
              const classes = setting.class_order?.length ? setting.class_order : classesFor(target);
              const targetMode = setting.goal === "target";
              return (
                <tr key={target} className={setting.optimize ? "objective-row" : "constraint-only-row"}>
                  <td className="target-name-cell"><strong>{target}</strong><span>{setting.task_type}</span></td>
                  <td>
                    <input
                      className="table-checkbox"
                      type="checkbox"
                      checked={setting.optimize}
                      disabled={targetMode}
                      title={targetMode ? "目標値は最適化目的として扱います。" : "制約専用にする場合はチェックを外します。"}
                      onChange={(event) => patchTargetSetting(target, { optimize: event.target.checked })}
                    />
                  </td>
                  <td>{directionControl(target, setting)}</td>
                  <td>
                    <select
                      value={targetMode ? "none" : setting.goal}
                      disabled={targetMode}
                      title={targetMode ? "目標値は方向で設定されています。" : undefined}
                      onChange={(event) => changeGoal(
                        target,
                        event.target.value as Exclude<TargetGoal, "target">
                      )}
                    >
                      <option value="none">なし</option>
                      <option value="above">以上</option>
                      <option value="below">以下</option>
                    </select>
                  </td>
                  <td>{targetValueControl(target, setting, classes)}</td>
                  <td className="class-config-cell">{desiredClassControl(target, setting, classesFor(target))}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="settings-note">
        方向・制約・対象クラスは候補提案時にのみ使用します。ここを変更しても、互換性のある学習済みモデルは再利用できます。
      </p>
    </article>
  );
}
