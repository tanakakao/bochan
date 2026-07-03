export type ColumnKind = "numeric" | "categorical" | "datetime" | "string";

export interface ColumnProfile {
  name: string;
  dtype: string;
  kind: ColumnKind;
  missing_count: number;
  missing_rate: number;
  unique_count: number;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  std?: number | null;
  values?: string[];
}

export interface DatasetProfile {
  n_rows: number;
  n_columns: number;
  columns: ColumnProfile[];
}

export interface DatasetResponse {
  dataset_id: string;
  name: string;
  source_type: "csv" | "excel";
  profile: DatasetProfile;
  preview: Record<string, unknown>[];
}

export interface SearchVariable {
  name: string;
  type: "numeric" | "categorical";
  lower?: number;
  upper?: number;
  step?: number;
  fixed: boolean;
  fixed_value?: string | number;
  categories?: string[];
}

export interface CandidateRow {
  rank: number;
  values: Record<string, string | number>;
  acq_value: number | null;
  predicted_target_mean: number;
  predicted_target_std: number;
  constraints_ok: boolean;
}

export interface RegressionResult {
  dataset_id: string;
  dataset_name: string;
  task_type: string;
  model_type: string;
  n_train: number;
  n_features: number;
  feature_columns: string[];
  target_column: string;
  direction: "maximize" | "minimize";
  best_observed: number;
  candidates: CandidateRow[];
  metadata: Record<string, unknown>;
}
