import type { RunRegressionInput } from "./api";
import { loadFeatureMissingSettings } from "./webRunSettings";

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, canonicalValue(item)])
    );
  }
  return value;
}

/** Fingerprint of model fitting settings; proposal-only settings are intentionally excluded. */
export function buildModelReuseSignature(input: RunRegressionInput): string {
  const featureMissing = loadFeatureMissingSettings();
  const targetModelSettings = input.targetSettings.map((setting) => ({
    target: setting.target,
    task_type: setting.task_type,
    target_class: setting.target_class ?? null,
    class_order: setting.class_order ?? []
  }));
  const modelSearchSpace = input.searchSpace.map((variable) => ({
    name: variable.name,
    type: variable.type,
    lower: variable.lower,
    upper: variable.upper,
    categories: variable.categories ?? []
  }));
  return JSON.stringify(canonicalValue({
    datasetId: input.datasetId,
    featureColumns: input.featureColumns,
    targetColumns: input.targetColumns,
    targetModelSettings,
    modelType: input.modelType,
    projectionDimensions: input.projectionDimensions,
    fitMaxiter: input.fitMaxiter,
    normalize: input.normalize,
    inputPerturbation: input.inputPerturbation,
    nW: input.nW,
    perturbationStd: input.perturbationStd,
    searchSpace: modelSearchSpace,
    featureMissing
  }));
}
