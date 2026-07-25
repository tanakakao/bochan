export type ColumnKind = "numeric" | "categorical" | "datetime" | "string";
export type TargetClassValue = string | number;

export interface HealthResponse {
  status: string;
  application: string;
}

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
  values?: TargetClassValue[];
}

export interface DatasetProfile {
  n_rows: number;
  n_columns: number;
  columns: ColumnProfile[];
}

export interface DatasetResponse {
  dataset_id: string;
  name: string;
  source_type: "csv" | "excel" | "model_artifact";
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
  categories?: TargetClassValue[];
}

export type TaskType = "regression" | "classification" | "ordinal";
export type TargetGoal = "none" | "above" | "below" | "target";
export type Direction = "maximize" | "minimize";
export type AcquisitionFamily = "bayesian_optimization" | "active_learning" | "level_set_estimation";
export type ConstraintKind = "equality" | "inequality";
export type ConstraintOperator = "=" | "<=" | ">=";

/** Task definition, optimization role, direction, and optional constraint for one target column. */
export interface TargetSetting {
  target: string;
  task_type: TaskType;
  /** Whether this output participates in the acquisition objective. */
  optimize: boolean;
  /** Optimization direction. Ignored when goal is `target`. */
  direction: Direction;
  /** `none` adds no hard constraint. */
  goal: TargetGoal;
  /** Regression value, probability threshold, or ordinal boundary class. */
  value?: string | number | null;
  /** Selected positive/desired class for binary classification. */
  target_class?: TargetClassValue | null;
  /** Desired class set for multiclass classification. */
  target_classes?: TargetClassValue[];
  /** Explicit low-to-high class order for ordinal regression. */
  class_order?: TargetClassValue[];
  /** One or more desired ordinal classes when goal is `target`. */
  target_values?: TargetClassValue[];
}

export interface TargetMetadata extends Partial<TargetSetting> {
  target: string;
  task_type: TaskType;
  goal: TargetGoal;
  internal_task?: string;
  configured_value?: unknown;
  classes?: TargetClassValue[] | null;
  class_index?: number | null;
  class_indices?: number[];
  num_classes?: number | null;
}

export interface LinearConstraint {
  id: string;
  kind: ConstraintKind;
  terms: Record<string, number>;
  operator: ConstraintOperator;
  rhs: number;
}

/** Legacy response shape kept for backward compatibility. */
export interface OutcomeConstraint {
  id?: string;
  target: string;
  operator: "<=" | ">=";
  value: number;
}

export interface KSparseConfig {
  enabled: boolean;
  k: number;
  variables: string[];
}

export interface CandidatePrediction {
  mean: number;
  std: number;
  prediction_space?: string;
}

export interface CandidateRow {
  rank: number;
  values: Record<string, string | number>;
  acq_value: number | null;
  predictions: Record<string, CandidatePrediction>;
  /** First-target compatibility field. */
  predicted_target_mean: number;
  /** First-target compatibility field. */
  predicted_target_std: number;
  constraints_ok: boolean;
}

export interface PlotlyFigurePayload {
  data: unknown[];
  layout: Record<string, unknown>;
  frames?: unknown[];
}

export interface ResultVisualization {
  id: string;
  title: string;
  description: string;
  figure: PlotlyFigurePayload;
}

export type VisualizationKind = "yyplot" | "target_relation" | "pareto" | "1d" | "2d" | "ternary";

export interface VisualizationRequest {
  kind: VisualizationKind;
  target?: string;
  target_x?: string;
  target_y?: string;
  features?: string[];
  fixed_values?: Record<string, string | number>;
  show_type?: "pred" | "acqf";
  n?: number;
  sum_value?: number;
}

export interface VisualizationOptions {
  feature_columns: string[];
  numeric_features: string[];
  target_columns: string[];
  regression_targets: string[];
  target_tasks?: Record<string, string>;
  ternary_groups?: Array<{ features: string[]; sum_value: number }>;
  feature_controls?: Record<string, {
    kind: "numeric" | "categorical";
    min?: number;
    max?: number;
    default: string | number;
    values?: Array<string | number>;
  }>;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
  event?: string;
  request_id?: string;
  duration_ms?: number;
  [key: string]: unknown;
}

export interface LogsResponse {
  entries: LogEntry[];
  count: number;
  log_file: string;
}

export interface RegressionResult {
  dataset_id: string;
  dataset_name: string;
  task_type: string;
  model_type: string;
  n_train: number;
  n_features: number;
  feature_columns: string[];
  target_columns: string[];
  /** First-target compatibility field. */
  target_column: string;
  target_settings?: TargetSetting[];
  target_metadata?: Record<string, TargetMetadata>;
  directions: Record<string, Direction>;
  /** First-target compatibility field. */
  direction: Direction;
  best_observed: number | Record<string, number>;
  outcome_constraints?: OutcomeConstraint[];
  candidates: CandidateRow[];
  visualizations: ResultVisualization[];
  visualization_warnings: string[];
  visualization_run_id?: string;
  visualization_options?: VisualizationOptions;
  metadata: Record<string, unknown>;
}

export interface ModelArtifactImportResponse {
  dataset: DatasetResponse;
  result: RegressionResult;
  request: Record<string, unknown>;
  artifact: {
    filename?: string | null;
    artifact_version: number;
    bochan_version?: string | null;
    original_run_id?: string | null;
    restored_run_id: string;
    pickle_warning: string;
  };
}
