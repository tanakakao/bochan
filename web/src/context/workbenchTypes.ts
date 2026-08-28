import type {
  AcquisitionFamily,
  ColumnProfile,
  DatasetResponse,
  Direction,
  FeatureImportanceSettings,
  RegressionResult,
  SearchVariable,
  TargetSetting
} from "../types";
import type { CompositionSettings } from "../compositionExtension";

export type WorkbenchStep = "data" | "prepare" | "settings" | "optimize" | "results" | "logs";
export type Theme = "light" | "dark";
export type ModelExecutionMode = "reuse" | "retrain";
export type HealthState = {
  status: "loading" | "ready" | "error";
  text: string;
};
export type CrossValidationSettings = {
  enabled: boolean;
  method: "kfold" | "loo";
  nSplits: number;
};

export const STEPS: Array<[WorkbenchStep, string, string]> = [
  ["data", "Data", "データ読込"],
  ["prepare", "Select", "変数・型選択"],
  ["settings", "Model", "モデル設定"],
  ["optimize", "Suggest", "候補提案"],
  ["results", "Results", "候補と可視化"],
  ["logs", "Logs", "実行履歴"]
];

export interface WorkbenchContextValue {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  step: WorkbenchStep;
  setStep: (step: WorkbenchStep) => void;
  canOpenStep: (step: WorkbenchStep) => boolean;
  health: HealthState;
  busy: string | null;
  error: string | null;
  setError: (error: string | null) => void;
  dataset: DatasetResponse | null;
  datasetRevision: number;
  columns: ColumnProfile[];
  selectableColumns: ColumnProfile[];
  targetCandidates: ColumnProfile[];
  featureColumns: string[];
  targetColumn: string;
  targetColumns: string[];
  targetSettings: Record<string, TargetSetting>;
  selectedTargetSettings: TargetSetting[];
  optimizedTargetSettings: TargetSetting[];
  targetDirections: Record<string, Direction>;
  direction: Direction;
  variables: Record<string, SearchVariable>;
  selectedVariables: SearchVariable[];
  normalize: boolean;
  setNormalize: (value: boolean) => void;
  inputPerturbation: boolean;
  setInputPerturbation: (value: boolean) => void;
  nW: number;
  setNW: (value: number) => void;
  perturbationStd: number;
  setPerturbationStd: (value: number) => void;
  projectionDimensions: number;
  setProjectionDimensions: (value: number) => void;
  modelType: string;
  setModelType: (modelType: string) => void;
  compositionSettings: CompositionSettings;
  crabnetCheckpoint: string;
  setCrabnetCheckpoint: (checkpoint: string) => void;
  crabnetEncoderTraining: "partial" | "full";
  setCrabnetEncoderTraining: (mode: "partial" | "full") => void;
  acquisitionFamily: AcquisitionFamily;
  setAcquisitionFamily: (family: AcquisitionFamily) => void;
  acquisition: string;
  setAcquisition: (acquisition: string) => void;
  beta: number;
  setBeta: (beta: number) => void;
  fitMaxiter: number;
  setFitMaxiter: (fitMaxiter: number) => void;
  crossValidation: CrossValidationSettings;
  setCrossValidation: (value: CrossValidationSettings) => void;
  featureImportance: FeatureImportanceSettings;
  setFeatureImportance: (value: FeatureImportanceSettings) => void;
  q: number;
  setQ: (q: number) => void;
  sequential: boolean;
  setSequential: (sequential: boolean) => void;
  minimumCandidateDistanceRatio: number;
  setMinimumCandidateDistanceRatio: (ratio: number) => void;
  numRestarts: number;
  setNumRestarts: (numRestarts: number) => void;
  rawSamples: number;
  setRawSamples: (rawSamples: number) => void;
  result: RegressionResult | null;
  resultRevision: number;
  canConfigure: boolean;
  settingsValid: boolean;
  candidateSettingsValid: boolean;
  modelReuseAvailable: boolean;
  handleFile: (file: File | null) => Promise<void>;
  handleModelArtifact: (file: File | null) => Promise<void>;
  toggleFeature: (name: string) => void;
  toggleTarget: (name: string) => void;
  patchTargetSetting: (target: string, patch: Partial<TargetSetting>) => void;
  patchVariable: (name: string, patch: Partial<SearchVariable>) => void;
  execute: (mode?: ModelExecutionMode) => Promise<void>;
  numberOrUndefined: (value: string) => number | undefined;
}
