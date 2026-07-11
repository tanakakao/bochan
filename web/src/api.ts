import type {
  DatasetResponse,
  HealthResponse,
  LogsResponse,
  RegressionResult,
  SearchVariable
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

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
  direction: "maximize" | "minimize";
  modelType: string;
  fitMaxiter: number;
  acquisition: string;
  beta: number;
  q: number;
  numRestarts: number;
  rawSamples: number;
  searchSpace: SearchVariable[];
}

export async function runRegression(input: RunRegressionInput): Promise<RegressionResult> {
  return request<RegressionResult>("/regression/run", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: input.datasetId,
      feature_columns: input.featureColumns,
      target_column: input.targetColumn,
      direction: input.direction,
      model_type: input.modelType,
      fit_maxiter: input.fitMaxiter,
      normalize: true,
      outcome_transform: true,
      search_space: input.searchSpace,
      constraints: [],
      acquisition: {
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
