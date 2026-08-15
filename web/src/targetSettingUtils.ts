import type { ColumnProfile, TargetClassValue, TargetSetting } from "./types";

/** Returns observed class values from the profile or the uploaded preview. */
export function getColumnClassValues(
  column: ColumnProfile,
  preview: Record<string, unknown>[]
): TargetClassValue[] {
  const values: TargetClassValue[] = [];
  const seen = new Set<string>();

  function append(value: unknown) {
    if (value === null || value === undefined || value === "") return;
    if (typeof value !== "string" && typeof value !== "number") return;
    const key = `${typeof value}:${String(value)}`;
    if (seen.has(key)) return;
    seen.add(key);
    values.push(value);
  }

  (column.values ?? []).forEach(append);
  if (values.length === 0) {
    preview.forEach((row) => append(row[column.name]));
  }
  return values;
}

export function targetClassValues(setting: TargetSetting): TargetClassValue[] {
  if (setting.target_class !== null && setting.target_class !== undefined) {
    return [setting.target_class];
  }
  return setting.target_classes ?? [];
}

function sameClassValue(left: TargetClassValue, right: TargetClassValue): boolean {
  return left === right || String(left) === String(right);
}

function includesClass(values: TargetClassValue[], candidate: unknown): boolean {
  if (candidate === null || candidate === undefined) return false;
  if (typeof candidate !== "string" && typeof candidate !== "number") return false;
  return values.some((value) => sameClassValue(value, candidate));
}

function sameClassList(left: TargetClassValue[] | undefined, right: TargetClassValue[]): boolean {
  if (!left || left.length !== right.length) return false;
  return left.every((value, index) => sameClassValue(value, right[index]));
}

function finiteNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function regressionLevelSetThreshold(column: ColumnProfile, setting: TargetSetting): number {
  const configured = finiteNumber(setting.value);
  if (configured !== null) return configured;
  const mean = finiteNumber(column.mean);
  if (mean !== null) return mean;
  const lower = finiteNumber(column.min);
  const upper = finiteNumber(column.max);
  if (lower !== null && upper !== null) return (lower + upper) / 2;
  return lower ?? upper ?? 0;
}

/**
 * Return the smallest patch that initializes one target for Web LSE.
 *
 * Regression LSE estimates the contour y = threshold, so its objective is always
 * expressed as a target-value objective. Classification estimates a selected
 * class-probability contour and defaults to p=0.5. Ordinal LSE uses the configured
 * class order as a rank scale and defaults to the middle class as its boundary.
 * Once a valid LSE role has been initialized, editable threshold values are left
 * untouched so users can temporarily clear an input while typing a replacement.
 */
export function levelSetTargetDefaultPatch(
  column: ColumnProfile,
  setting: TargetSetting,
  preview: Record<string, unknown>[]
): Partial<TargetSetting> | null {
  if (!setting.optimize) {
    return setting.goal === "target"
      ? { goal: "none", value: null, target_values: [] }
      : null;
  }

  const patch: Partial<TargetSetting> = {};
  const weight = finiteNumber(setting.level_set_weight);
  if (weight === null || weight < 0) patch.level_set_weight = 1;

  if (setting.task_type === "regression") {
    if (setting.goal !== "target") {
      patch.goal = "target";
      patch.value = regressionLevelSetThreshold(column, setting);
    }
    if (setting.direction !== "maximize") patch.direction = "maximize";
    if (setting.target_values?.length) patch.target_values = [];
    return Object.keys(patch).length ? patch : null;
  }

  const observedClasses = getColumnClassValues(column, preview);
  const classes = setting.task_type === "ordinal" && setting.class_order?.length
    ? setting.class_order
    : observedClasses;

  if (setting.task_type === "classification") {
    if (setting.goal !== "above" && setting.goal !== "below") {
      patch.goal = "above";
      patch.value = 0.5;
    }
    if (setting.target_values?.length) patch.target_values = [];

    if (classes.length === 2) {
      const preferred = includesClass(classes, setting.target_class)
        ? setting.target_class as TargetClassValue
        : classes[1] ?? classes[0];
      if (preferred !== undefined) {
        if (!includesClass([preferred], setting.target_class)) patch.target_class = preferred;
        if (!sameClassList(setting.target_classes, [preferred])) patch.target_classes = [preferred];
      }
    } else {
      const selected = (setting.target_classes ?? []).filter((value) => includesClass(classes, value));
      const preferred = selected.length ? selected : classes.length ? [classes[0]] : [];
      if (setting.target_class !== null && setting.target_class !== undefined) patch.target_class = null;
      if (!sameClassList(setting.target_classes, preferred)) patch.target_classes = preferred;
    }
    return Object.keys(patch).length ? patch : null;
  }

  if (!setting.class_order?.length && observedClasses.length) {
    patch.class_order = [...observedClasses];
  }
  const validTargetValues = (setting.target_values ?? []).filter((value) => includesClass(classes, value));
  if (setting.goal === "target" && validTargetValues.length) {
    if (!sameClassList(setting.target_values, validTargetValues)) patch.target_values = validTargetValues;
    return Object.keys(patch).length ? patch : null;
  }
  if (setting.goal === "above" || setting.goal === "below") {
    if (setting.target_values?.length) patch.target_values = [];
    return Object.keys(patch).length ? patch : null;
  }

  patch.goal = "above";
  patch.value = classes.length ? classes[Math.floor(classes.length / 2)] : null;
  if (setting.target_values?.length) patch.target_values = [];
  return patch;
}
