import { buildModelReuseSignature, type RunRegressionInput } from "./api";
import { getColumnClassValues } from "./targetSettingUtils";
import type {
  AcquisitionFamily,
  ColumnProfile,
  Direction,
  ModelArtifactImportResponse,
  SearchVariable,
  TargetSetting
} from "./types";
import {
  saveFeatureConstraints,
  saveFeatureMissingSettings,
  saveInputPerturbationRiskSettings,
  saveSearchMethod,
  saveSelectionCountConstraint,
  type FeatureConstraint,
  type InputPerturbationRiskType,
  type SearchMethod
} from "./webRunSettings";

interface RestoredWorkbenchState {
  featureColumns: string[];
  targetColumns: string[];
  targetSettings: Record<string, TargetSetting>;
  variables: Record<string, SearchVariable>;
  normalize: boolean;
  inputPerturbation: boolean;
  nW: number;
  perturbationStd: number;
  projectionDimensions: number;
  modelType: string;
  acquisitionFamily: AcquisitionFamily;
  acquisition: string;
  beta: number;
  fitMaxiter: number;
  q: number;
  sequential: boolean;
  minimumCandidateDistanceRatio: number;
  numRestarts: number;
  rawSamples: number;
  modelSignature: string;
}

function asRecord(value: unknown): Record<string, any> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, any>
    : {};
}

function finiteNumber(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function createVariable(
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

function createTargetSetting(
  column: ColumnProfile,
  preview: Record<string, unknown>[],
  direction: Direction
): TargetSetting {
  const common = {
    target: column.name,
    optimize: true,
    direction,
    goal: "none" as const,
    value: null
  };
  if (column.kind === "numeric") return { ...common, task_type: "regression" };
  const classes = getColumnClassValues(column, preview);
  const selected = classes.length === 2 ? classes[1] : classes[0];
  return {
    ...common,
    task_type: "classification",
    target_class: classes.length === 2 ? selected ?? null : null,
    target_classes: selected === undefined ? [] : [selected]
  };
}

function restoreFeatureSettings(request: Record<string, any>): void {
  const constraints: FeatureConstraint[] = (Array.isArray(request.constraints) ? request.constraints : [])
    .map((raw: unknown, index: number) => {
      const constraint = asRecord(raw);
      const terms = Array.isArray(constraint.terms) ? constraint.terms.map(asRecord) : [];
      const variables = terms
        .map((term) => String(term.column ?? ""))
        .filter(Boolean);
      const sense = String(constraint.sense ?? "eq");
      return {
        id: String(constraint.name ?? `artifact-constraint-${index + 1}`),
        variables,
        coefficients: Object.fromEntries(terms.map((term) => [
          String(term.column ?? ""),
          finiteNumber(term.coefficient, 1)
        ])),
        operator: sense === "ge" ? ">" : sense === "le" ? "<" : "=",
        value: finiteNumber(constraint.rhs, 0)
      } as FeatureConstraint;
    })
    .filter((constraint) => constraint.variables.length > 0);
  saveFeatureConstraints(constraints);

  const sparse = asRecord(request.k_sparse);
  const sparseColumns = Array.isArray(sparse.columns)
    ? sparse.columns.map(String).filter(Boolean)
    : [];
  saveSelectionCountConstraint({
    enabled: Boolean(sparse.enabled) && sparseColumns.length > 0,
    variables: sparseColumns,
    k: Math.max(1, Math.trunc(finiteNumber(sparse.k, 1)))
  });

  const modelKwargs = asRecord(request.model_kwargs);
  const missing = asRecord(modelKwargs.web_feature_missing);
  saveFeatureMissingSettings({
    strategy: missing.strategy === "impute" ? "impute" : "drop",
    continuousStrategy: missing.continuous_impute_strategy === "iterative" ? "iterative" : "mean",
    categoricalStrategy: "mode",
    imputeMaxIter: Math.max(1, Math.trunc(finiteNumber(missing.impute_max_iter, 10))),
    imputeRandomState: missing.impute_random_state === null || missing.impute_random_state === undefined
      ? null
      : Math.trunc(finiteNumber(missing.impute_random_state, 0)),
    multipleImputeSamplePosterior: Boolean(missing.multiple_impute_sample_posterior)
  });

  const acquisitionKwargs = asRecord(asRecord(request.acquisition).acqf_kwargs);
  const rawRiskType = String(acquisitionKwargs.web_risk_type ?? "none").toLowerCase();
  const riskType: InputPerturbationRiskType = rawRiskType === "var" || rawRiskType === "cvar"
    ? rawRiskType
    : "none";
  const riskAlpha = finiteNumber(acquisitionKwargs.web_risk_alpha, 0.2);
  saveInputPerturbationRiskSettings({
    riskType,
    alpha: riskAlpha > 0 && riskAlpha <= 1 ? riskAlpha : 0.2
  });

  const validMethods: SearchMethod[] = [
    "normal",
    "torch",
    "ga",
    "sa",
    "pso",
    "cmaes",
    "thompson_sampling",
    "nsgaii"
  ];
  const requestedMethod = String(asRecord(request.optimizer).name ?? "normal");
  const searchMethod = requestedMethod === "optimize_acqf" ? "normal" : requestedMethod;
  saveSearchMethod(validMethods.includes(searchMethod as SearchMethod)
    ? searchMethod as SearchMethod
    : "normal");
}

/** Restore all UI settings needed to inspect and reuse an imported fitted model. */
export function restoreWorkbenchFromArtifact(
  imported: ModelArtifactImportResponse
): RestoredWorkbenchState {
  const { dataset, result } = imported;
  const request = asRecord(imported.request);
  const modelKwargs = asRecord(request.model_kwargs);
  const acquisitionSettings = asRecord(request.acquisition);
  const acquisitionKwargs = asRecord(acquisitionSettings.acqf_kwargs);
  const optimizerSettings = asRecord(request.optimizer);
  const preview = dataset.preview;
  const columns = dataset.profile.columns;
  const featureColumns = result.feature_columns.length
    ? [...result.feature_columns]
    : (Array.isArray(request.feature_columns) ? request.feature_columns.map(String) : []);
  const targetColumns = result.target_columns.length
    ? [...result.target_columns]
    : (Array.isArray(request.target_columns) ? request.target_columns.map(String) : []);
  const directions = asRecord(request.directions) as Record<string, Direction>;

  const rawTargetSettings = result.target_settings?.length
    ? result.target_settings
    : Array.isArray(modelKwargs.web_target_settings)
      ? modelKwargs.web_target_settings as TargetSetting[]
      : [];
  const targetSettings = Object.fromEntries(targetColumns.map((name) => {
    const existing = rawTargetSettings.find((setting) => setting.target === name);
    if (existing) return [name, { ...existing, target: name }];
    const column = columns.find((candidate) => candidate.name === name);
    if (!column) throw new Error(`保存モデルの目的変数がデータにありません: ${name}`);
    const direction = directions[name] ?? result.directions?.[name] ?? "maximize";
    return [name, createTargetSetting(column, preview, direction)];
  }));

  const searchSpace = Array.isArray(request.search_space)
    ? request.search_space.map(asRecord)
    : [];
  const variables = Object.fromEntries(columns
    .filter((column) => column.kind === "numeric" || column.kind === "categorical")
    .map((column) => {
      const fallback = createVariable(column, preview);
      const saved = searchSpace.find((item) => String(item.name) === column.name);
      if (!saved) return [column.name, fallback];
      const type = saved.type === "categorical" ? "categorical" : "numeric";
      return [column.name, {
        ...fallback,
        name: column.name,
        type,
        lower: type === "numeric" && saved.lower !== null && saved.lower !== undefined
          ? finiteNumber(saved.lower, fallback.lower ?? 0)
          : undefined,
        upper: type === "numeric" && saved.upper !== null && saved.upper !== undefined
          ? finiteNumber(saved.upper, fallback.upper ?? 1)
          : undefined,
        step: type === "numeric" && saved.step !== null && saved.step !== undefined
          ? finiteNumber(saved.step, 0)
          : undefined,
        fixed: Boolean(saved.fixed),
        fixed_value: saved.fixed_value ?? undefined,
        categories: type === "categorical"
          ? (Array.isArray(saved.categories) ? saved.categories : fallback.categories)
          : undefined
      } as SearchVariable];
    }));

  restoreFeatureSettings(request);

  const normalize = request.normalize === undefined ? true : Boolean(request.normalize);
  const inputPerturbation = Boolean(request.input_perturbation);
  const nW = Math.max(1, Math.trunc(finiteNumber(request.n_w, 16)));
  const perturbationStd = finiteNumber(request.perturbation_std, 0.1);
  const backendModelType = String(request.model_type ?? result.model_type ?? "base");
  const modelType = backendModelType === "rrp" ? "robust" : backendModelType;
  const projectionDimensions = Math.max(1, Math.trunc(finiteNumber(
    modelKwargs.n_components,
    Math.min(2, Math.max(featureColumns.length, 1))
  )));
  const fitMaxiter = Math.max(1, Math.trunc(finiteNumber(request.fit_maxiter, 128)));
  const acquisitionFamily = ["active_learning", "level_set_estimation"].includes(
    String(acquisitionKwargs.web_family)
  )
    ? String(acquisitionKwargs.web_family) as AcquisitionFamily
    : "bayesian_optimization";
  const acquisition = String(acquisitionSettings.name ?? "EI");
  const acquisitionKey = acquisition.replace(/[_\-\s]/g, "").toLowerCase();
  const defaultLevelSetParameter = acquisitionKey === "boundaryvariance"
    ? 1
    : acquisitionKey === "icu"
      ? 0
      : 1.96;
  const savedLevelSetParameter = acquisitionKwargs.web_level_set_parameter;
  const beta = acquisitionFamily === "level_set_estimation"
    ? savedLevelSetParameter === null || savedLevelSetParameter === undefined
      ? defaultLevelSetParameter
      : finiteNumber(savedLevelSetParameter, defaultLevelSetParameter)
    : finiteNumber(acquisitionSettings.beta, 2);
  const q = Math.max(1, Math.trunc(finiteNumber(optimizerSettings.q, 3)));
  const sequential = optimizerSettings.sequential === undefined
    ? true
    : Boolean(optimizerSettings.sequential);
  const minimumCandidateDistanceRatio = Math.min(1, Math.max(
    0,
    finiteNumber(optimizerSettings.minimum_candidate_distance_ratio, 1e-3)
  ));
  const numRestarts = Math.max(1, Math.trunc(finiteNumber(optimizerSettings.num_restarts, 10)));
  const rawSamples = Math.max(1, Math.trunc(finiteNumber(optimizerSettings.raw_samples, 256)));
  const selectedSettings = targetColumns.map((name) => targetSettings[name]);
  const targetDirections = Object.fromEntries(selectedSettings.map((setting) => [
    setting.target,
    setting.goal === "target" ? "maximize" : setting.direction
  ])) as Record<string, Direction>;
  const targetColumn = selectedSettings.find((setting) => setting.optimize)?.target
    ?? targetColumns[0]
    ?? "";
  const signatureInput: RunRegressionInput = {
    datasetId: dataset.dataset_id,
    featureColumns,
    targetColumn,
    targetColumns,
    targetSettings: selectedSettings,
    targetDirections,
    direction: targetDirections[targetColumn] ?? "maximize",
    modelType,
    projectionDimensions,
    fitMaxiter,
    normalize,
    inputPerturbation,
    nW,
    perturbationStd,
    acquisitionFamily,
    acquisition,
    beta,
    q,
    sequential,
    minimumCandidateDistanceRatio,
    numRestarts,
    rawSamples,
    searchSpace: featureColumns.map((name) => variables[name]).filter(Boolean)
  };

  return {
    featureColumns,
    targetColumns,
    targetSettings,
    variables,
    normalize,
    inputPerturbation,
    nW,
    perturbationStd,
    projectionDimensions,
    modelType,
    acquisitionFamily,
    acquisition,
    beta,
    fitMaxiter,
    q,
    sequential,
    minimumCandidateDistanceRatio,
    numRestarts,
    rawSamples,
    modelSignature: buildModelReuseSignature(signatureInput)
  };
}
