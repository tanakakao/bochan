import type {
  AcquisitionFamily,
  DatasetResponse,
  Direction,
  HealthResponse,
  LogsResponse,
  ModelArtifactImportResponse,
  RegressionResult,
  ResultVisualization,
  SearchVariable,
  TargetSetting,
  VisualizationRequest
} from "./types";
import {
  loadNoiseAlpha,
  saveNoiseAlpha,
  supportsNoiseAlpha
} from "./noiseAlphaSettings";
import {
  loadFeatureConstraints,
  loadFeatureMissingSettings,
  loadInputPerturbationRiskSettings,
  loadSearchMethod,
  loadSelectionCountConstraint
} from "./webRunSettings";

const RAW_API_BASE = String(import.meta.env.VITE_API_BASE ?? "/api/v1").trim();
const API_BASE = (RAW_API_BASE || "/api/v1").replace(/\/+$/, "");
const WEB_BACKEND_COMMAND = "uvicorn bochan.serving.webapp.app:app --reload --port 8000";

export interface WebCapabilities {
  task_types: string[];
  model_types: string[];
  gamma_model_types?: string[];
  acquisitions: string[];
  acquisition_families?: Partial<Record<AcquisitionFamily, string[]>>;
  optimizers: string[];
  data_sources: string[];
  visualizations: string[];
  model_artifacts?: Record<string, unknown>;
  logging?: Record<string, unknown>;
}

function webRouteNotFoundMessage(method: string, url: string): string {
  return [
    `Web APIが見つかりません (${method} ${url})。`,
    "通常のbochan Core APIではなく、Webワークベンチ用FastAPIを起動してください。",
    `起動コマンド: ${WEB_BACKEND_COMMAND}`,
    `VITE_API_BASEを設定している場合は、Web APIのprefixを含むURL（例: http://127.0.0.1:8000/api/v1）にしてください。`
  ].join("\n");
}

async function responsePayload(response: Response): Promise<any> {
  const responseText = await response.text();
  if (!responseText) return null;
  try {
    return JSON.parse(responseText);
  } catch {
    return responseText;
  }
}

function responseError(
  response: Response,
  payload: any,
  options: { method: string; url: string; missingWebRoute?: boolean }
): Error {
  const { method, url, missingWebRoute = false } = options;
  const requestId = response.headers.get("X-Request-ID");
  const detail = missingWebRoute
    ? webRouteNotFoundMessage(method, url)
    : (payload?.detail ?? payload ?? `HTTP ${response.status}`);
  const message = typeof detail === "string" ? detail : JSON.stringify(detail);
  return new Error(requestId ? `${message} [request_id=${requestId}]` : message);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  const payload = await responsePayload(response);
  if (!response.ok) {
    const method = String(init?.method ?? "GET").toUpperCase();
    const missingWebRoute = response.status === 404 && (
      path === "/capabilities" || path === "/datasets"
    );
    throw responseError(response, payload, { method, url, missingWebRoute });
  }
  return payload as T;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const [health] = await Promise.all([
    request<HealthResponse>("/health"),
    request<WebCapabilities>("/capabilities")
  ]);
  return health;
}

export async function fetchCapabilities(): Promise<WebCapabilities> {
  return request<WebCapabilities>("/capabilities");
}

export async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("ファイルを読み込めませんでした。"));
    reader.readAsDataURL(file);
  });
}

export async function uploadDataset(file: File): Promise<DatasetResponse> {
  const sourceType = file.name.toLowerCase().endsWith(".xlsx") || file.name.toLowerCase().endsWith(".xls")
    ? "excel"
    : "csv";
  const contentBase64 = await fileToDataUrl(file);
  return request<DatasetResponse>("/datasets", {
    method: "POST",
    body: JSON.stringify({
      source_type: sourceType,
      name: file.name,
      content_base64: contentBase64,
      encoding: "utf-8-sig",
      sheet_name: 0
    })
  });
}

export async function uploadModelArtifact(file: File): Promise<ModelArtifactImportResponse> {
  const path = "/model-artifacts/import?trust_pickle=true";
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Model-Filename": file.name
    },
    body: file
  });
  const payload = await responsePayload(response);
  if (!response.ok) {
    throw responseError(response, payload, {
      method: "POST",
      url,
      missingWebRoute: response.status === 404
    });
  }
  const imported = payload as ModelArtifactImportResponse;
  const savedAlpha = Number(
    (imported.request?.model_kwargs as Record<string, unknown> | undefined)?._tabular_noise_alpha
  );
  if (Number.isFinite(savedAlpha) && savedAlpha > 0) saveNoiseAlpha(savedAlpha);
  return imported;
}

function artifactFilename(response: Response, fallback: string): string {
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) return decodeURIComponent(utf8Match[1]);
  const plainMatch = disposition.match(/filename="?([^";]+)"?/i);
  return plainMatch?.[1] || fallback;
}

export async function downloadModelArtifact(runId: string, datasetName: string): Promise<void> {
  const path = `/runs/${encodeURIComponent(runId)}/model-artifact`;
  const url = `${API_BASE}${path}`;
  const response = await fetch(url);
  if (!response.ok) {
    const payload = await responsePayload(response);
    throw responseError(response, payload, {
      method: "GET",
      url,
      missingWebRoute: false
    });
  }
  const blob = await response.blob();
  const fallbackStem = datasetName.replace(/\.[^.]+$/, "") || "bochan_model";
  const filename = artifactFilename(response, `${fallbackStem}.bochan.pt`);
  const blobUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = blobUrl;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(blobUrl);
}

export interface RunRegressionInput {
  datasetId: string;
  featureColumns: string[];
  targetColumn: string;
  targetColumns: string[];
  targetSettings: TargetSetting[];
  targetDirections: Record<string, Direction>;
  direction: Direction;
  modelType: string;
  projectionDimensions: number;
  fitMaxiter: number;
  normalize: boolean;
  inputPerturbation: boolean;
  nW: number;
  perturbationStd: number;
  acquisitionFamily: AcquisitionFamily;
  acquisition: string;
  beta: number;
  q: number;
  numRestarts: number;
  rawSamples: number;
  searchSpace: SearchVariable[];
  reuseModelRunId?: string;
}

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

function resolvedNoiseAlpha(input: RunRegressionInput): number | null {
  const hasRegressionTarget = input.targetSettings.some(
    (setting) => setting.task_type === "regression"
  );
  return supportsNoiseAlpha(input.modelType) && hasRegressionTarget
    ? loadNoiseAlpha()
    : null;
}

/** Fingerprint containing only settings that affect model fitting and encoding. */
export function buildModelReuseSignature(input: RunRegressionInput): string {
  const featureMissing = loadFeatureMissingSettings();
  const noiseAlpha = resolvedNoiseAlpha(input);
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
    alpha: noiseAlpha,
    normalize: input.normalize,
    inputPerturbation: input.inputPerturbation,
    nW: input.nW,
    perturbationStd: input.perturbationStd,
    searchSpace: modelSearchSpace,
    featureMissing
  }));
}

function validateTargetSetting(setting: TargetSetting): string | null {
  if (!setting.target) return "目的変数名が空です。";
  if (setting.goal === "target" && !setting.optimize) {
    return `${setting.target}: 目標値を設定した目的変数は最適化対象にしてください。`;
  }
  if (setting.direction !== "maximize" && setting.direction !== "minimize") {
    return `${setting.target}: 最大化または最小化を選択してください。`;
  }
  if (setting.task_type === "regression") {
    if (setting.goal !== "none" && !Number.isFinite(Number(setting.value))) {
      return `${setting.target}: 回帰のしきい値または目標値を数値で指定してください。`;
    }
    return null;
  }
  if (setting.task_type === "classification") {
    const selectedClasses = setting.target_classes ?? [];
    if (setting.target_class === null || setting.target_class === undefined) {
      if (selectedClasses.length === 0) return `${setting.target}: ターゲットクラスを指定してください。`;
    }
    if (
      (setting.goal === "above" || setting.goal === "below") &&
      (!Number.isFinite(Number(setting.value)) || Number(setting.value) < 0 || Number(setting.value) > 1)
    ) {
      return `${setting.target}: 分類の以上・以下は0〜1の確率しきい値を指定してください。`;
    }
    return null;
  }
  const classOrder = setting.class_order ?? [];
  if (classOrder.length < 2) return `${setting.target}: 順序回帰のクラス順序を指定してください。`;
  if ((setting.goal === "above" || setting.goal === "below") && String(setting.value ?? "").trim() === "") {
    return `${setting.target}: 順序回帰の境界クラスを指定してください。`;
  }
  if (setting.goal === "target" && (setting.target_values ?? []).length === 0) {
    return `${setting.target}: 目標クラスを1つ以上指定してください。`;
  }
  return null;
}

/** Runs optimization, active learning, or level-set estimation through the Web API. */
export async function runRegression(input: RunRegressionInput): Promise<RegressionResult> {
  if (input.targetColumns.length < 1) throw new Error("目的変数を1列以上選択してください。");
  if (!input.targetColumns.includes(input.targetColumn)) {
    throw new Error("代表目的変数が選択済み目的変数に含まれていません。");
  }
  if (input.targetSettings.length !== input.targetColumns.length) {
    throw new Error("すべての目的変数にタスク種別を設定してください。");
  }
  const settingTargets = input.targetSettings.map((setting) => setting.target);
  if (input.targetColumns.some((target) => !settingTargets.includes(target))) {
    throw new Error("目的変数とTargetSettingの対応が不整合です。");
  }
  const optimized = input.targetSettings.filter((setting) => setting.optimize);
  if (optimized.length === 0) throw new Error("最適化対象の目的変数を1つ以上選択してください。");
  if (!optimized.some((setting) => setting.target === input.targetColumn)) {
    throw new Error("代表目的変数は最適化対象から選択してください。");
  }
  const settingError = input.targetSettings.map(validateTargetSetting).find(Boolean);
  if (settingError) throw new Error(settingError);
  if (!Number.isInteger(input.nW) || input.nW < 1) throw new Error("入力摂動のnは1以上の整数にしてください。");
  if (!Number.isFinite(input.perturbationStd) || input.perturbationStd <= 0) {
    throw new Error("入力摂動のばらつきは0より大きくしてください。");
  }

  const noiseAlpha = resolvedNoiseAlpha(input);
  if (noiseAlpha !== null && (!Number.isFinite(noiseAlpha) || noiseAlpha <= 0)) {
    throw new Error("観測ノイズ下限αは0より大きい有限値にしてください。");
  }

  const perturbationRisk = loadInputPerturbationRiskSettings();
  if (!Number.isFinite(perturbationRisk.alpha) || perturbationRisk.alpha <= 0 || perturbationRisk.alpha > 1) {
    throw new Error("VaR/CVaRのαは0より大きく1以下にしてください。");
  }
  const riskSupported = input.inputPerturbation && input.acquisitionFamily === "bayesian_optimization";
  const effectiveRiskType = riskSupported ? perturbationRisk.riskType : "none";

  const featureMissing = loadFeatureMissingSettings();
  if (!Number.isInteger(featureMissing.imputeMaxIter) || featureMissing.imputeMaxIter < 1) {
    throw new Error("欠損値補完の最大反復回数は1以上の整数にしてください。");
  }
  const featureConstraints = loadFeatureConstraints();
  const featureSet = new Set(input.featureColumns);
  const invalidConstraint = featureConstraints.find((constraint) => (
    constraint.variables.length === 0 ||
    constraint.variables.some((name) => !featureSet.has(name)) ||
    constraint.variables.some((name) => !Number.isFinite(constraint.coefficients[name])) ||
    !Number.isFinite(constraint.value)
  ));
  if (invalidConstraint) throw new Error("説明変数の制約に無効な列、係数、または値があります。");

  const selectionCount = loadSelectionCountConstraint();
  if (selectionCount.enabled) {
    if (!selectionCount.variables.length || selectionCount.variables.some((name) => !featureSet.has(name))) {
      throw new Error("有効変数数制約の説明変数を1つ以上選択してください。");
    }
    if (!Number.isInteger(selectionCount.k) || selectionCount.k < 1 || selectionCount.k > selectionCount.variables.length) {
      throw new Error("有効変数数制約の採用数を選択変数数以下の正の整数にしてください。");
    }
  }

  const searchMethod = loadSearchMethod();
  if (searchMethod === "nsgaii" && optimized.length < 2) {
    throw new Error("NSGA-IIは多目的の場合にのみ使用できます。");
  }
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
    // The Web endpoint keeps its public schema backward compatible. This
    // Web-only adapter setting is removed before model construction and is
    // executed through the same tabular converter used by /tabular/models.
    web_feature_missing: {
      strategy: featureMissing.strategy,
      continuous_impute_strategy: featureMissing.continuousStrategy,
      categorical_impute_strategy: featureMissing.categoricalStrategy,
      impute_random_state: featureMissing.imputeRandomState,
      impute_max_iter: featureMissing.imputeMaxIter,
      multiple_impute_sample_posterior: featureMissing.multipleImputeSamplePosterior
    }
  };
  if (noiseAlpha !== null) {
    modelKwargs._tabular_noise_alpha = noiseAlpha;
  }
  if (input.reuseModelRunId) {
    modelKwargs.web_reuse_model_run_id = input.reuseModelRunId;
  }
  if (input.modelType === "pca" || input.modelType === "rembo") {
    modelKwargs.n_components = input.projectionDimensions;
  }

  return request<RegressionResult>("/regression/run", {
    method: "POST",
    body: JSON.stringify({
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
        acqf_kwargs: {
          web_family: input.acquisitionFamily,
          web_risk_type: effectiveRiskType,
          web_risk_alpha: perturbationRisk.alpha
        }
      },
      optimizer: {
        name: searchMethod,
        q: input.q,
        num_restarts: input.numRestarts,
        raw_samples: input.rawSamples,
        sequential:
          input.searchSpace.some((variable) => variable.type === "categorical") ||
          searchMethod === "cmaes"
      },
      // Target missing values still use the automatic target policy. Feature
      // missing values are controlled independently through web_feature_missing.
      drop_missing: true
    })
  });
}

export async function fetchResultVisualization(
  runId: string,
  value: VisualizationRequest
): Promise<ResultVisualization> {
  return request<ResultVisualization>(`/runs/${encodeURIComponent(runId)}/visualizations`, {
    method: "POST",
    body: JSON.stringify(value)
  });
}

export async function fetchLogs(options?: {
  limit?: number;
  level?: string;
  event?: string;
  requestId?: string;
}): Promise<LogsResponse> {
  const params = new URLSearchParams();
  params.set("limit", String(options?.limit ?? 200));
  if (options?.level) params.set("level", options.level);
  if (options?.event) params.set("event", options.event);
  if (options?.requestId) params.set("request_id", options.requestId);
  return request<LogsResponse>(`/logs?${params.toString()}`);
}