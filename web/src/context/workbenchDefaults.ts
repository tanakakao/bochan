import { getColumnClassValues } from "../targetSettingUtils";
import type {
  ColumnProfile,
  DatasetResponse,
  Direction,
  FeatureImportanceSettings,
  SearchVariable,
  TargetSetting
} from "../types";

export const DEFAULT_FEATURE_IMPORTANCE: FeatureImportanceSettings = {
  enabled: false,
  source: "auto",
  nRepeats: 10,
  randomState: 0,
  diagnosticAuto: true,
  computeNoiseImportance: true,
  normalizeImportance: false,
  topK: 15,
  rankBy: "value",
  includeNegative: true,
  showErrorBars: true
};

export function numberOrUndefined(value: string): number | undefined {
  if (value.trim() === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function createVariable(
  column: ColumnProfile,
  preview: Record<string, unknown>[]
): SearchVariable {
  if (column.kind === "categorical") {
    return {
      name: column.name,
      type: "categorical",
      fixed: false,
      categories: getColumnClassValues(column, preview)
    };
  }
  return {
    name: column.name,
    type: "numeric",
    lower: column.min ?? undefined,
    upper: column.max ?? undefined,
    fixed: false
  };
}

export function createTargetSetting(
  column: ColumnProfile,
  preview: Record<string, unknown>[]
): TargetSetting {
  const common = {
    target: column.name,
    optimize: true,
    direction: "maximize" as Direction,
    goal: "none" as const,
    value: null
  };
  if (column.kind === "numeric") {
    return { ...common, task_type: "regression" };
  }

  const classes = getColumnClassValues(column, preview);
  if (classes.length === 2) {
    return {
      ...common,
      task_type: "classification",
      target_class: classes[1],
      target_classes: [classes[1]]
    };
  }
  return {
    ...common,
    task_type: "classification",
    target_class: null,
    target_classes: classes.length ? [classes[0]] : []
  };
}

export interface InitialSelectionState {
  dataset: DatasetResponse;
  featureColumns: string[];
  targetColumns: string[];
  targetSettings: Record<string, TargetSetting>;
  variables: Record<string, SearchVariable>;
  projectionDimensions: number;
}

export function createInitialSelectionState(dataset: DatasetResponse): InitialSelectionState {
  const candidates = dataset.profile.columns.filter(
    (column) => column.kind === "numeric" || column.kind === "categorical"
  );
  const initialTarget = candidates.at(-1);
  const featureColumns = candidates
    .filter((column) => column.name !== initialTarget?.name)
    .map((column) => column.name);

  return {
    dataset,
    featureColumns,
    targetColumns: initialTarget ? [initialTarget.name] : [],
    targetSettings: initialTarget ? {
      [initialTarget.name]: createTargetSetting(initialTarget, dataset.preview)
    } : {},
    variables: Object.fromEntries(
      candidates.map((column) => [column.name, createVariable(column, dataset.preview)])
    ),
    projectionDimensions: Math.min(2, Math.max(featureColumns.length, 1))
  };
}
