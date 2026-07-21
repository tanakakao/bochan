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
