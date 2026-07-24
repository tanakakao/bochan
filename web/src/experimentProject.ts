import type { RunRegressionInput } from "./api";
import type { RegressionResult } from "./types";
import {
  loadFeatureConstraints,
  loadFeatureMissingSettings,
  loadSearchMethod,
  loadSelectionCountConstraint
} from "./webRunSettings";

const RAW_API_BASE = String(import.meta.env.VITE_API_BASE ?? "/api/v1").trim();
const API_BASE = (RAW_API_BASE || "/api/v1").replace(/\/+$/, "");

export interface ExperimentProjectExportOptions {
  includeLatestModel?: boolean;
  includePastModels?: boolean;
}

async function responsePayload(response: Response): Promise<any> {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function errorFromResponse(response: Response, payload: any): Error {
  const detail = payload?.detail ?? payload ?? `HTTP ${response.status}`;
  const message = typeof detail === "string" ? detail : JSON.stringify(detail);
  const requestId = response.headers.get("X-Request-ID");
  return new Error(requestId ? `${message} [request_id=${requestId}]` : message);
}

function artifactFilename(response: Response, fallback: string): string {
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) return decodeURIComponent(utf8Match[1]);
  const plainMatch = disposition.match(/filename="?([^";]+)"?/i);
  return plainMatch?.[1] || fallback;
}

function buildProjectRunRequest(input: RunRegressionInput): Record<string, unknown> {
  const featureMissing = loadFeatureMissingSettings();
  const featureConstraints = loadFeatureConstraints();
  const selectionCount = loadSelectionCountConstraint();
  const searchMethod = loadSearchMethod();
  const backendModelType = input.modelType === "robust" ? "rrp" : input.modelType;
  const acquisitionName = searchMethod === "nsgaii" ? "nsgaii" : input.acquisition;
  const constraints = featureConstraints.map((constraint, index) => ({
    name: `feature-constraint-${index + 1}`,
    terms: constraint.variables.map((column) => ({
      column,
      coefficient: constraint.coefficients[column] ?? 1
    })),
    sense: constraint.operator === ">" ? "ge" : constraint.operator === "<" ? "le" : "eq",
    rhs: constraint.value,
    enabled: true
  }));
  const modelKwargs: Record<string, unknown> = {
    web_target_settings: input.targetSettings,
    web_target_roles: Object.fromEntries(input.targetSettings.map((setting) => [setting.target, {
      optimize: setting.optimize,
      direction: setting.direction
    }])),
    web_feature_missing: {
      strategy: featureMissing.strategy,
      continuous_impute_strategy: featureMissing.continuousStrategy,
      categorical_impute_strategy: featureMissing.categoricalStrategy,
      impute_random_state: featureMissing.imputeRandomState,
      impute_max_iter: featureMissing.imputeMaxIter,
      multiple_impute_sample_posterior: featureMissing.multipleImputeSamplePosterior
    }
  };
  if (input.modelType === "pca" || input.modelType === "rembo") {
    modelKwargs.n_components = input.projectionDimensions;
  }

  return {
    dataset_id: input.datasetId,
    feature_columns: input.featureColumns,
    target_column: input.targetColumn,
    target_columns: input.targetColumns,
    direction: input.targetDirections[input.targetColumn] ?? input.direction,
    directions: input.targetDirections,
    model_type: backendModelType,
    model_kwargs: modelKwargs,
    fit_maxiter: input.fitMaxiter,
    normalize: input.normalize,
    outcome_transform: true,
    input_perturbation: input.inputPerturbation,
    n_w: input.nW,
    perturbation_std: input.perturbationStd,
    search_space: input.searchSpace,
    constraints,
    outcome_constraints: [],
    k_sparse: selectionCount.enabled ? {
      enabled: true,
      columns: selectionCount.variables,
      k: selectionCount.k,
      score: "abs",
      support_selection: "topk",
      final_priority: "grid"
    } : null,
    acquisition: {
      name: acquisitionName,
      beta: input.beta,
      acqf_kwargs: { web_family: input.acquisitionFamily }
    },
    optimizer: {
      name: searchMethod,
      q: input.q,
      num_restarts: input.numRestarts,
      raw_samples: input.rawSamples,
      sequential: true
    },
    drop_missing: true
  };
}

/** Download a project ZIP with latest-model inclusion enabled by default. */
export async function downloadExperimentProject(
  input: RunRegressionInput,
  result: RegressionResult,
  datasetName: string,
  options: ExperimentProjectExportOptions = {}
): Promise<void> {
  const path = "/experiment-projects/export";
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: input.datasetId,
      request: buildProjectRunRequest(input),
      result,
      include_latest_model: options.includeLatestModel ?? true,
      include_past_models: options.includePastModels ?? false
    })
  });
  if (!response.ok) {
    throw errorFromResponse(response, await responsePayload(response));
  }
  const blob = await response.blob();
  const stem = datasetName.replace(/\.[^.]+$/, "") || "bochan_project";
  const filename = artifactFilename(response, `${stem}.bochan-project.zip`);
  const blobUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = blobUrl;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(blobUrl);
}
