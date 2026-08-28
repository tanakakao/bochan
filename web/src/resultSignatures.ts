import { buildModelReuseSignature, type RunRegressionInput } from "./api";
import { compositionSettingsToBackend } from "./compositionExtension";
import {
  loadFeatureConstraints,
  loadInputPerturbationRiskSettings,
  loadSearchMethod,
  loadSelectionCountConstraint
} from "./webRunSettings";

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

function normalizedFeatureConstraints() {
  return loadFeatureConstraints()
    .map((constraint) => ({
      operator: constraint.operator,
      value: constraint.value,
      terms: [...constraint.variables]
        .sort((left, right) => left.localeCompare(right))
        .map((name) => ({ name, coefficient: constraint.coefficients[name] ?? 1 }))
    }))
    .sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right)));
}

function normalizedSelectionCountConstraint() {
  const setting = loadSelectionCountConstraint();
  if (!setting.enabled) return { enabled: false };
  return {
    enabled: true,
    variables: [...setting.variables].sort((left, right) => left.localeCompare(right)),
    k: setting.k
  };
}

/**
 * Fingerprint for the candidate/result artifact produced from an already fitted model.
 *
 * It intentionally contains the model signature plus objective, acquisition, optimizer,
 * constraint, and candidate-space settings. Candidate-only changes therefore invalidate
 * Results without automatically disabling fitted-model reuse.
 */
export function buildSuggestionSignature(input: RunRegressionInput): string {
  const searchMethod = loadSearchMethod();
  const riskSettings = loadInputPerturbationRiskSettings();
  const riskSupported = input.inputPerturbation && (
    input.acquisitionFamily === "bayesian_optimization" ||
    input.acquisitionFamily === "level_set_estimation"
  );
  const perturbationRisk = riskSupported && riskSettings.riskType !== "none"
    ? riskSettings
    : { riskType: "none" };
  const effectiveSequential = Boolean(input.sequential ?? true) ||
    input.searchSpace.some((variable) => variable.type === "categorical") ||
    searchMethod === "cmaes" ||
    (input.acquisitionFamily === "level_set_estimation" && input.inputPerturbation);
  const composition = input.compositionSettings.enabled
    ? compositionSettingsToBackend(input.compositionSettings)
    : null;

  return JSON.stringify(canonicalValue({
    modelSignature: buildModelReuseSignature(input),
    targetColumn: input.targetColumn,
    targetDirections: input.targetDirections,
    targetSettings: input.targetSettings.map((setting) => ({
      target: setting.target,
      optimize: setting.optimize,
      direction: setting.direction,
      goal: setting.goal,
      value: setting.value ?? null,
      target_class: setting.target_class ?? null,
      target_classes: setting.target_classes ?? [],
      class_order: setting.class_order ?? [],
      target_values: setting.target_values ?? [],
      level_set_weight: setting.level_set_weight ?? 1
    })),
    acquisition: {
      family: input.acquisitionFamily,
      name: searchMethod === "nsgaii" ? "nsgaii" : input.acquisition,
      beta: input.beta,
      perturbationRisk
    },
    optimizer: {
      name: searchMethod,
      q: input.q,
      numRestarts: input.numRestarts,
      rawSamples: input.rawSamples,
      sequential: effectiveSequential,
      minimumCandidateDistanceRatio: input.minimumCandidateDistanceRatio ?? 1e-3
    },
    searchSpace: input.searchSpace.map((variable) => ({
      name: variable.name,
      type: variable.type,
      lower: variable.lower ?? null,
      upper: variable.upper ?? null,
      step: variable.step ?? null,
      fixed: Boolean(variable.fixed),
      fixedValue: variable.fixed_value ?? null,
      categories: variable.categories ?? []
    })),
    featureConstraints: normalizedFeatureConstraints(),
    selectionCountConstraint: normalizedSelectionCountConstraint(),
    compositionElementConstraints: composition?.element_constraints ?? null,
    featureImportance: input.featureImportance ?? null
  }));
}
