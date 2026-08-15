import type { AcquisitionFamily } from "./types";
import {
  loadFeatureConstraints,
  loadFeatureMissingSettings,
  loadSearchMethod,
  loadSelectionCountConstraint,
  saveFeatureConstraints,
  saveFeatureMissingSettings,
  saveSearchMethod,
  saveSelectionCountConstraint,
  type FeatureConstraint,
  type FeatureMissingSettings,
  type SearchMethod,
  type SelectionCountConstraint
} from "./webRunSettings";

/** Persisted settings that an analysis configuration may apply for execution. */
export interface StoredRunSettingsSnapshot {
  featureConstraints: FeatureConstraint[];
  featureMissing: FeatureMissingSettings;
  searchMethod: SearchMethod;
  selectionCount: SelectionCountConstraint;
}

/** Canonical model settings shared by every web analysis entry point. */
export interface AnalysisModelConfig {
  normalize: boolean;
  inputPerturbation: boolean;
  nW: number;
  perturbationStd: number;
  projectionDimensions: number;
  modelType: string;
  fitMaxiter: number;
}

/** Canonical acquisition settings shared by every web analysis entry point. */
export interface AnalysisAcquisitionConfig {
  family: AcquisitionFamily;
  name: string;
  beta: number;
}

/** Canonical candidate-search settings shared by every web analysis entry point. */
export interface AnalysisSearchConfig {
  q: number;
  numRestarts: number;
  rawSamples: number;
  /** Optional for guided flows that intentionally preserve the current setting. */
  sequential?: boolean;
  /** Optional for guided flows that intentionally preserve the current setting. */
  minimumCandidateDistanceRatio?: number;
}

/** Canonical analysis settings used by Simple, Advanced, and Conversation modes. */
export interface AnalysisConfig {
  model: AnalysisModelConfig;
  acquisition: AnalysisAcquisitionConfig;
  search: AnalysisSearchConfig;
  persisted: StoredRunSettingsSnapshot;
}

/** Setter surface needed to apply an AnalysisConfig to WorkbenchContext. */
export interface AnalysisConfigSetters {
  setNormalize: (value: boolean) => void;
  setInputPerturbation: (value: boolean) => void;
  setNW: (value: number) => void;
  setPerturbationStd: (value: number) => void;
  setProjectionDimensions: (value: number) => void;
  setModelType: (value: string) => void;
  setAcquisitionFamily: (value: AcquisitionFamily) => void;
  setAcquisition: (value: string) => void;
  setBeta: (value: number) => void;
  setFitMaxiter: (value: number) => void;
  setQ: (value: number) => void;
  setNumRestarts: (value: number) => void;
  setRawSamples: (value: number) => void;
  setSequential?: (value: boolean) => void;
  setMinimumCandidateDistanceRatio?: (value: number) => void;
}

const GUIDED_FEATURE_MISSING: FeatureMissingSettings = {
  strategy: "drop",
  continuousStrategy: "mean",
  categoricalStrategy: "mode",
  imputeMaxIter: 10,
  imputeRandomState: null,
  multipleImputeSamplePosterior: false
};

const GUIDED_SELECTION_COUNT: SelectionCountConstraint = {
  enabled: false,
  variables: [],
  k: 1
};

/** Capture persisted advanced settings from the current workbench state. */
export function captureStoredRunSettings(): StoredRunSettingsSnapshot {
  return {
    featureConstraints: loadFeatureConstraints(),
    featureMissing: loadFeatureMissingSettings(),
    searchMethod: loadSearchMethod(),
    selectionCount: loadSelectionCountConstraint()
  };
}

/** Build the recommended defaults used by Simple and Conversation modes. */
export function createGuidedAnalysisConfig({
  featureCount,
  targetCount,
  q
}: {
  featureCount: number;
  targetCount: number;
  q: number;
}): AnalysisConfig {
  return {
    model: {
      normalize: true,
      inputPerturbation: false,
      nW: 16,
      perturbationStd: 0.1,
      projectionDimensions: Math.min(2, Math.max(featureCount, 1)),
      modelType: "base",
      fitMaxiter: 128
    },
    acquisition: {
      family: "bayesian_optimization",
      name: targetCount > 1 ? "EHVI" : "EI",
      beta: 2
    },
    search: {
      q,
      numRestarts: 10,
      rawSamples: 256
    },
    persisted: {
      featureConstraints: [],
      featureMissing: { ...GUIDED_FEATURE_MISSING },
      searchMethod: "normal",
      selectionCount: { ...GUIDED_SELECTION_COUNT, variables: [] }
    }
  };
}

/**
 * Build the canonical config represented by the Advanced Model/Suggest controls.
 *
 * Persisted controls such as feature constraints, missing-value handling, and
 * search method are captured at execution time so child control state cannot
 * drift from the request that is about to run.
 */
export function createAdvancedAnalysisConfig({
  model,
  acquisition,
  search
}: {
  model: AnalysisModelConfig;
  acquisition: AnalysisAcquisitionConfig;
  search: Required<AnalysisSearchConfig>;
}): AnalysisConfig {
  return {
    model: { ...model },
    acquisition: { ...acquisition },
    search: { ...search },
    persisted: captureStoredRunSettings()
  };
}

/** Apply the persisted-settings portion of an AnalysisConfig. */
export function applyStoredRunSettings(settings: StoredRunSettingsSnapshot): void {
  saveFeatureConstraints(settings.featureConstraints);
  saveFeatureMissingSettings(settings.featureMissing);
  saveSearchMethod(settings.searchMethod);
  saveSelectionCountConstraint(settings.selectionCount);
}

/** Restore persisted advanced settings after guided execution completes or validation fails. */
export function restoreStoredRunSettings(snapshot: StoredRunSettingsSnapshot | null): void {
  if (!snapshot) return;
  applyStoredRunSettings(snapshot);
}

/** Apply model, acquisition, and search settings through the WorkbenchContext setters. */
export function applyAnalysisConfig(config: AnalysisConfig, setters: AnalysisConfigSetters): void {
  setters.setNormalize(config.model.normalize);
  setters.setInputPerturbation(config.model.inputPerturbation);
  setters.setNW(config.model.nW);
  setters.setPerturbationStd(config.model.perturbationStd);
  setters.setProjectionDimensions(config.model.projectionDimensions);
  setters.setModelType(config.model.modelType);
  setters.setAcquisitionFamily(config.acquisition.family);
  setters.setAcquisition(config.acquisition.name);
  setters.setBeta(config.acquisition.beta);
  setters.setFitMaxiter(config.model.fitMaxiter);
  setters.setQ(config.search.q);
  setters.setNumRestarts(config.search.numRestarts);
  setters.setRawSamples(config.search.rawSamples);
  if (config.search.sequential !== undefined) {
    setters.setSequential?.(config.search.sequential);
  }
  if (config.search.minimumCandidateDistanceRatio !== undefined) {
    setters.setMinimumCandidateDistanceRatio?.(config.search.minimumCandidateDistanceRatio);
  }
}
