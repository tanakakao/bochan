import {
  MODEL_OPTIONS,
  isCrabNetMixedModelType,
  isCrabNetModelType,
  isCrabNetMultitaskModelType
} from "../modelOptions";
import type { CompositionSettings } from "../compositionExtension";
import { getColumnClassValues } from "../targetSettingUtils";
import type {
  ColumnProfile,
  DatasetResponse,
  Direction,
  SearchVariable,
  TargetClassValue,
  TargetSetting
} from "../types";

function finiteNumber(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function classKey(value: TargetClassValue): string {
  return String(value);
}

function containsClass(
  values: TargetClassValue[],
  value: TargetClassValue | null | undefined
): boolean {
  if (value === null || value === undefined) return false;
  const requested = classKey(value);
  return values.some((candidate) => classKey(candidate) === requested);
}

function sameClassSet(left: TargetClassValue[], right: TargetClassValue[]): boolean {
  if (left.length !== right.length) return false;
  const leftKeys = new Set(left.map(classKey));
  const rightKeys = new Set(right.map(classKey));
  return leftKeys.size === left.length && rightKeys.size === right.length &&
    [...leftKeys].every((value) => rightKeys.has(value));
}

function validateVariableModelSetting(variable: SearchVariable): boolean {
  return variable.type !== "categorical" || Boolean(variable.categories?.length);
}

function validateVariableCandidateSetting(variable: SearchVariable): boolean {
  if (variable.type === "categorical") {
    if (!variable.categories?.length) return false;
    if (!variable.fixed) return true;
    return variable.fixed_value !== undefined &&
      variable.categories.some((value) => String(value) === String(variable.fixed_value));
  }
  const lower = finiteNumber(variable.lower);
  const upper = finiteNumber(variable.upper);
  if (lower === null || upper === null || lower >= upper) return false;
  if (variable.step !== undefined && (finiteNumber(variable.step) ?? 0) <= 0) return false;
  if (!variable.fixed) return true;
  const fixed = finiteNumber(variable.fixed_value);
  return fixed !== null && fixed >= lower && fixed <= upper;
}

function validateTargetModelSetting(
  setting: TargetSetting,
  column: ColumnProfile | undefined,
  preview: Record<string, unknown>[]
): boolean {
  if (!column) return false;
  if (setting.task_type === "regression") return column.kind === "numeric";

  const classes = getColumnClassValues(column, preview);
  if (classes.length < 2) return false;
  if (setting.task_type === "classification") {
    return classes.length !== 2 || containsClass(classes, setting.target_class);
  }
  return sameClassSet(setting.class_order ?? [], classes);
}

function validateTargetCandidateSetting(
  setting: TargetSetting,
  column: ColumnProfile | undefined,
  preview: Record<string, unknown>[]
): boolean {
  if (!column) return false;
  if (setting.goal === "target" && !setting.optimize) return false;

  if (setting.task_type === "regression") {
    return setting.goal === "none" || finiteNumber(setting.value) !== null;
  }

  const classes = getColumnClassValues(column, preview);
  if (classes.length < 2) return false;
  if (setting.task_type === "classification") {
    if (classes.length === 2) {
      if (!containsClass(classes, setting.target_class)) return false;
    } else {
      const selected = setting.target_classes ?? [];
      if (selected.length === 0 || !selected.every((value) => containsClass(classes, value))) return false;
    }
    if (setting.goal === "above" || setting.goal === "below") {
      const threshold = finiteNumber(setting.value);
      return threshold !== null && threshold >= 0 && threshold <= 1;
    }
    return setting.goal === "none";
  }

  const order = setting.class_order ?? [];
  if (!sameClassSet(order, classes)) return false;
  if (setting.goal === "none") return true;
  if (setting.goal === "above" || setting.goal === "below") {
    return containsClass(order, setting.value as TargetClassValue);
  }
  const targets = setting.target_values ?? [];
  return targets.length > 0 && targets.every((value) => containsClass(order, value));
}

export interface WorkbenchValidationInput {
  dataset: DatasetResponse | null;
  featureColumns: string[];
  targetColumns: string[];
  targetSettings: Record<string, TargetSetting>;
  variables: Record<string, SearchVariable>;
  inputPerturbation: boolean;
  nW: number;
  perturbationStd: number;
  projectionDimensions: number;
  modelType: string;
  compositionSettings: CompositionSettings;
  fitMaxiter: number;
}

export interface WorkbenchDerivedState {
  columns: ColumnProfile[];
  selectableColumns: ColumnProfile[];
  targetCandidates: ColumnProfile[];
  selectedTargetSettings: TargetSetting[];
  optimizedTargetSettings: TargetSetting[];
  targetColumn: string;
  selectedVariables: SearchVariable[];
  targetDirections: Record<string, Direction>;
  direction: Direction;
  canConfigure: boolean;
  settingsValid: boolean;
  candidateSettingsValid: boolean;
}

export function deriveWorkbenchState(input: WorkbenchValidationInput): WorkbenchDerivedState {
  const {
    dataset,
    featureColumns,
    targetColumns,
    targetSettings,
    variables,
    inputPerturbation,
    nW,
    perturbationStd,
    projectionDimensions,
    modelType,
    compositionSettings,
    fitMaxiter
  } = input;
  const columns = dataset?.profile.columns ?? [];
  const preview = dataset?.preview ?? [];
  const selectableColumns = columns.filter(
    (column) => column.kind === "numeric" || column.kind === "categorical"
  );
  const selectedTargetSettings = targetColumns
    .map((name) => targetSettings[name])
    .filter((setting): setting is TargetSetting => Boolean(setting));
  const optimizedTargetSettings = selectedTargetSettings.filter((setting) => setting.optimize);
  const targetColumn = optimizedTargetSettings[0]?.target ?? targetColumns[0] ?? "";
  const selectedVariables = featureColumns
    .map((name) => variables[name])
    .filter((value): value is SearchVariable => Boolean(value));
  const targetDirections = Object.fromEntries(selectedTargetSettings.map((setting) => [
    setting.target,
    setting.goal === "target" ? "maximize" : setting.direction
  ])) as Record<string, Direction>;
  const direction = targetDirections[targetColumn] ?? "maximize";
  const canConfigure = Boolean(
    dataset &&
    targetColumns.length > 0 &&
    featureColumns.length > 0 &&
    targetColumns.every((target) => !featureColumns.includes(target))
  );
  const allRegression = selectedTargetSettings.length > 0 &&
    selectedTargetSettings.every((setting) => setting.task_type === "regression");
  const hasCategoricalFeatures = selectedVariables.some((variable) => variable.type === "categorical");
  const canUseMultitask = targetColumns.length > 1 && allRegression && !hasCategoricalFeatures;
  const projectedModel = modelType === "pca" || modelType === "rembo";
  const modelTypeKnown = MODEL_OPTIONS.some((option) => option.value === modelType);
  const crabnetModel = isCrabNetModelType(modelType);
  const crabnetMixedModel = isCrabNetMixedModelType(modelType);
  const crabnetMultitaskModel = isCrabNetMultitaskModelType(modelType);
  const compositionColumn = compositionSettings.enabled
    ? compositionSettings.column
    : "";
  const crabnetCompositionReady = Boolean(
    compositionColumn &&
    featureColumns.includes(compositionColumn) &&
    compositionSettings.elements.length >= 2
  );
  const hasCategoricalProcessFeatures = selectedVariables.some(
    (variable) => variable.type === "categorical" && variable.name !== compositionColumn
  );
  const crabnetOutputCountValid = crabnetMultitaskModel
    ? targetColumns.length > 1
    : targetColumns.length > 0;
  const crabnetProcessTypeValid = crabnetMixedModel
    ? hasCategoricalProcessFeatures
    : !hasCategoricalProcessFeatures;
  const crabnetSettingsValid = !crabnetModel || Boolean(
    crabnetOutputCountValid &&
    allRegression &&
    crabnetCompositionReady &&
    crabnetProcessTypeValid &&
    !inputPerturbation
  );
  const settingsValid = Boolean(
    canConfigure &&
    selectedTargetSettings.length === targetColumns.length &&
    selectedTargetSettings.every((setting) => validateTargetModelSetting(
      setting,
      columns.find((column) => column.name === setting.target),
      preview
    )) &&
    selectedVariables.every(validateVariableModelSetting) &&
    (!inputPerturbation || (Number.isInteger(nW) && nW >= 1 && perturbationStd > 0)) &&
    modelTypeKnown &&
    (modelType !== "multitask" || canUseMultitask) &&
    crabnetSettingsValid &&
    (!projectedModel || (
      Number.isInteger(projectionDimensions) &&
      projectionDimensions >= 1 &&
      projectionDimensions <= Math.max(selectedVariables.length, 1)
    )) &&
    Number.isInteger(fitMaxiter) && fitMaxiter >= 1
  );
  const candidateSettingsValid = Boolean(
    settingsValid &&
    optimizedTargetSettings.length > 0 &&
    selectedTargetSettings.every((setting) => validateTargetCandidateSetting(
      setting,
      columns.find((column) => column.name === setting.target),
      preview
    )) &&
    selectedVariables.every(validateVariableCandidateSetting)
  );

  return {
    columns,
    selectableColumns,
    targetCandidates: selectableColumns,
    selectedTargetSettings,
    optimizedTargetSettings,
    targetColumn,
    selectedVariables,
    targetDirections,
    direction,
    canConfigure,
    settingsValid,
    candidateSettingsValid
  };
}
