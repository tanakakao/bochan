import { targetClassValues } from "../../targetSettingUtils";
import type { AcquisitionFamily, TargetSetting } from "../../types";

export const API_STATUS_LABELS = {
  loading: "確認中",
  ready: "接続済み",
  error: "エラー"
} as const;

export function formatBestObserved(value: number | Record<string, number>): string {
  if (typeof value === "number") return Number.isFinite(value) ? value.toPrecision(5) : "—";
  return Object.entries(value)
    .map(([target, observed]) => `${target}: ${Number.isFinite(observed) ? observed.toPrecision(5) : "—"}`)
    .join(" / ");
}

function goalLabel(value: string): string {
  if (value === "none") return "制約なし";
  if (value === "below") return "≤";
  if (value === "target") return "目標";
  return "≥";
}

export function familyLabel(value: AcquisitionFamily): string {
  if (value === "active_learning") return "アクティブラーニング";
  if (value === "level_set_estimation") return "レベルセット推定";
  return "ベイズ最適化";
}

export function summarizeTargetSetting(setting: TargetSetting): string {
  const roleText = setting.optimize
    ? setting.goal === "target"
      ? "目標最適化"
      : setting.direction === "minimize"
        ? "最小化"
        : "最大化"
    : "制約専用";
  const classText = setting.task_type === "classification"
    ? `class=${targetClassValues(setting).map(String).join("|") || "—"}`
    : setting.task_type === "ordinal" && setting.goal === "target"
      ? `target=${(setting.target_values ?? []).map(String).join("|") || "—"}`
      : "";
  const constraintText = setting.goal === "none"
    ? goalLabel(setting.goal)
    : `${goalLabel(setting.goal)} ${setting.goal === "target" ? (setting.target_values ?? []).map(String).join("|") : String(setting.value ?? "—")}`;
  return [setting.target, roleText, classText, constraintText].filter(Boolean).join(": ");
}
