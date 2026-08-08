import { useWorkbench } from "../context/WorkbenchContext";
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
  const { acquisitionFamily } = useWorkbench();
  const isLevelSet = acquisitionFamily === "level_set_estimation";

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

  function changeConstraintGoal(target: string, nextGoal: TargetGoal) {
    if (nextGoal === "target") {
      changeDirection(target, "target");
      return;
    }
    changeGoal(target, nextGoal);
  }

  function directionControl(target: string, setting: TargetSetting) {
    if (!setting.optimize) return <span className="muted-cell">対象外</span>;
    if (isLevelSet) {
      return (
        <div className="table-field">
          <span className="muted-cell" title="レベルセット推定では最大化・最小化方向を使用しません。">
            境界推定
          </span>
          {targetColumns.length > 1 && (
            <label className="table-field">
              <span>境界重み</span>
              <input
                type="number"
                min={0}
                step={0.1}
                value={setting.level_set_weight ?? 1}
                aria-label={`${target}のレベルセット重み`}
                onChange={(event) => patchTargetSetting(target, {
                  level_set_weight: numberOrUndefined(event.target.value) ?? 0
                })}
              />
            </label>
          )}
        </div>
      );
    }
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
          aria-label={isLevelSet ? "対象クラス確率のレベルセットしきい値" : undefined}
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
        aria-label={isLevelSet ? "順序回帰のレベルセット境界クラス" : undefined}
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
    if (setting.task_type === "ordinal") {
      return (
        <span className="muted-cell">
          {isLevelSet ? "モデル設定のクラス順を境界尺度として使用" : "モデル設定の順序を使用"}
        </span>
      );
    }
    if (classes.length === 2) {
      return (
        <div className="binary-target-summary">
          <span>{isLevelSet ? "境界判定の対象クラス" : "1として扱うクラス"}</span>
          <strong>{String(setting.target_class ?? "—")}</strong>
        </div>
      );
    }
    return (
      <label className="table-field">
        <span>{isLevelSet ? "境界を推定する対象クラス（複数可）" : "候補提案で狙うクラス（複数可）"}</span>
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
          <h3>{isLevelSet ? "目的変数のレベルセット条件" : "目的変数の候補提案条件"}</h3>
          <p>
            {isLevelSet
              ? "各最適化対象に境界を設定します。最大化・最小化方向は使用せず、分類は対象クラス確率、順序回帰は設定済みのクラス順を目的尺度として扱います。"
              : "最適化対象、方向、目標値、実行可能性制約、狙うクラスを設定します。"}
          </p>
        </div>
        <span className="status-chip success">{targetColumns.length} targets</span>
      </div>
      <div className="table-wrap target-settings-wrap">
        <table className="target-settings-table proposal-target-table">
          <thead>
            <tr>
              <th>目的変数</th><th>最適化対象</th><th>{isLevelSet ? "探索" : "方向"}</th>
              <th>{isLevelSet ? "境界条件 / 制約" : "制約"}</th>
              <th>{isLevelSet ? "境界しきい値／目標値" : "しきい値／目標値"}</th><th>対象クラス</th>
            </tr>
          </thead>
          <tbody>
            {targetColumns.map((target) => {
              const setting = targetSettings[target];
              if (!setting) return null;
              const classes = setting.class_order?.length ? setting.class_order : classesFor(target);
              const targetMode = setting.goal === "target";
              const constraintValue = isLevelSet
                ? setting.goal
                : targetMode
                  ? "none"
                  : setting.goal;
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
                      value={constraintValue}
                      disabled={!isLevelSet && targetMode}
                      title={isLevelSet
                        ? setting.optimize
                          ? "最適化対象ではLSEの境界条件として使用し、候補のhard constraintにはしません。"
                          : "最適化対象外では候補の実行可能性制約として使用します。"
                        : targetMode
                          ? "目標値は方向で設定されています。"
                          : undefined}
                      onChange={(event) => changeConstraintGoal(
                        target,
                        event.target.value as TargetGoal
                      )}
                    >
                      <option value="none">なし</option>
                      <option value="above">以上</option>
                      <option value="below">以下</option>
                      {isLevelSet && setting.task_type !== "classification" && (
                        <option value="target">目標値</option>
                      )}
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
        {isLevelSet
          ? "レベルセット推定では最大化・最小化方向を設定しません。最適化対象行の「以上／以下／目標値」は探索する境界を定義し、hard constraintにはしません。チェックを外した行の「以上／以下」は実行可能性制約として使用します。Multiclass は選択クラス群の合計確率、順序回帰は設定済みクラス順を境界尺度として扱います。複数出力の境界重みは相対値として正規化されます。"
          : "方向・制約・対象クラスは候補提案時にのみ使用します。ここを変更しても、互換性のある学習済みモデルは再利用できます。"}
      </p>
    </article>
  );
}
