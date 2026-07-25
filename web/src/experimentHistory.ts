import type { ResultVisualization, TargetSetting } from "./types";

const RAW_API_BASE = String(import.meta.env.VITE_API_BASE ?? "/api/v1").trim();
const API_BASE = (RAW_API_BASE || "/api/v1").replace(/\/+$/, "");

export interface ExperimentTargetSummary {
  task_type: string;
  metric_kind: string;
  direction: string;
  goal: string;
  references: number[];
  count: number;
  mean: number | null;
  min: number | null;
  max: number | null;
  best: number | null;
}

export interface ExperimentCycle {
  cycle_id: string;
  cycle_number: number;
  created_at: string;
  parent_dataset_id: string;
  dataset_id: string;
  dataset_name: string;
  source_run_id?: string | null;
  append_mode: "initial" | "manual" | "import";
  n_rows_before: number;
  n_rows_after: number;
  appended_rows: number;
  rows: Record<string, unknown>[];
  feature_columns: string[];
  target_columns: string[];
  target_settings: TargetSetting[];
  model: Record<string, unknown>;
  acquisition: Record<string, unknown>;
  optimizer: Record<string, unknown>;
  best_observed_before: Record<string, number | null>;
  candidate_count: number;
  notes?: string | null;
  target_summary: Record<string, ExperimentTargetSummary>;
}

export interface ExperimentHistoryVisualization extends ResultVisualization {
  target: string;
}

export interface ExperimentHistoryResponse {
  dataset_id: string;
  count: number;
  targets: string[];
  cycles: ExperimentCycle[];
  visualizations: ExperimentHistoryVisualization[];
}

export interface RecordExperimentCycleInput {
  parent_dataset_id: string;
  dataset_id: string;
  dataset_name: string;
  source_run_id?: string;
  append_mode: "manual" | "import";
  n_rows_before: number;
  n_rows_after: number;
  rows: Record<string, unknown>[];
  feature_columns: string[];
  target_columns: string[];
  target_settings: TargetSetting[];
  model: Record<string, unknown>;
  acquisition: Record<string, unknown>;
  optimizer: Record<string, unknown>;
  best_observed_before: Record<string, number | null>;
  candidate_count: number;
  notes?: string;
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
    const requestId = response.headers.get("X-Request-ID");
    const detail = payload?.detail ?? payload ?? `HTTP ${response.status}`;
    const message = typeof detail === "string" ? detail : JSON.stringify(detail);
    throw new Error(requestId ? `${message} [request_id=${requestId}]` : message);
  }
  return payload as T;
}

/** Persist one completed experiment cycle through the Web API. */
export async function recordExperimentCycle(
  input: RecordExperimentCycleInput
): Promise<ExperimentCycle> {
  const response = await request<{ cycle: ExperimentCycle }>("/experiment-cycles", {
    method: "POST",
    body: JSON.stringify(input)
  });
  return response.cycle;
}

/** Load the dataset lineage, cycle details, and generated objective-progress figures. */
export async function fetchExperimentHistory(datasetId: string): Promise<ExperimentHistoryResponse> {
  return request<ExperimentHistoryResponse>(
    `/experiment-cycles?dataset_id=${encodeURIComponent(datasetId)}`
  );
}
