import "./types";
import type { FeatureImportanceSummaryRecord } from "./types";

export interface CompositionImportanceRecord extends FeatureImportanceSummaryRecord {
  evaluation_source?: "training" | "cross_validation";
  label?: string;
  n_repeats?: number | null;
}

export interface CompositionFeatureImportancePayload {
  column: string;
  elements: string[];
  coordinate_features: string[];
  mode: "proportional" | "balance";
  mode_label: string;
  balance_element?: string | null;
  evaluation_source: "training" | "cross_validation";
  requested_source?: "training" | "cross_validation" | null;
  n_repeats: number;
  overall: CompositionImportanceRecord[];
  elements: CompositionImportanceRecord[];
  warnings: string[];
}

declare module "./types" {
  interface RegressionResult {
    composition_feature_importance?: CompositionFeatureImportancePayload;
  }
}

export {};
