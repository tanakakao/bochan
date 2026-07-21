import type {
  AcquisitionFamily,
  DatasetResponse,
  Direction,
  HealthResponse,
  LogsResponse,
  RegressionResult,
  SearchVariable,
  TargetSetting
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

export interface WebCapabilities {
  task_types: string[];
  model_types: string[];
  acquisitions: string[];
  acquisition_families?: Partial<Record<AcquisitionFamily, string[]>>;
  optimizers: string[];
  data_sources: string[];
  visualizations: string[];
  logging?: Record<string, unknown>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail ?? `HTTP ${response.status}`;
    const requestId = response.headers.get("X-Request-ID");
    const message = typeof detail === "string" ? detail : JSON.stringify(detail);
    throw new Error(requestId ? `${message} [request_id=${requestId}]` : message);
  }
  return payload as T;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
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

interface RunRegressionInput {
  datasetId: string;
  featureColumns: string[];
  targetColumn: string;
  targetColumns: string[];
  targetSettings: TargetSetting[];
  targetDirections: Record<string, Direction>;
  direction: Direction;
  modelType: string;
  fitMaxiter: number;
  acquisitionFamily: AcquisitionFamily;
  acquisition: string;
  beta: number;
  q: number;
  numRestarts: number;
  rawSamples: number;
  searchSpace: SearchVariable[];
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
      if (selectedClasses.length === 0) {
        return `${setting.target}: ターゲットクラスを指定してください。`;
      }
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
  if (input.targetColumns.length < 1) {
    throw new Error("目的変数を1列以上選択してください。");
  }
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
  if (optimized.length === 0) {
    throw new Error("最適化対象の目的変数を1つ以上選択してください。");
  }
  if (!optimized.some((setting) => setting.target === input.targetColumn)) {
    throw new Error("代表目的変数は最適化対象から選択してください。");
  }
  const settingError = input.targetSettings.map(validateTargetSetting).find(Boolean);
  if (settingError) throw new Error(settingError);

  const backendModelType = input.modelType === "robust" ? "rrp" : input.modelType;

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
      model_kwargs: {
        web_target_settings: input.targetSettings
      },
      fit_maxiter: input.fitMaxiter,
      normalize: true,
      outcome_transform: true,
      search_space: input.searchSpace,
      constraints: [],
      outcome_constraints: [],
      k_sparse: null,
      acquisition: {
        family: input.acquisitionFamily,
        name: input.acquisition,
        beta: input.beta,
        acqf_kwargs: {}
      },
      optimizer: {
        name: "optimize_acqf",
        q: input.q,
        num_restarts: input.numRestarts,
        raw_samples: input.rawSamples,
        sequential: true
      },
      drop_missing: true
    })
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
