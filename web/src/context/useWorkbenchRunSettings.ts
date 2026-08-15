import { useEffect, useRef, useState } from "react";
import type { AcquisitionFamily, FeatureImportanceSettings } from "../types";
import { loadCrossValidationSettings, saveCrossValidationSettings } from "../webRunSettings";
import { DEFAULT_FEATURE_IMPORTANCE } from "./workbenchDefaults";

const DEFAULT_CANDIDATE_DISTANCE_RATIO = 1e-3;
const DEFAULT_LSE_CANDIDATE_DISTANCE_RATIO = 1e-2;

interface RestoredRunSettings {
  normalize: boolean;
  inputPerturbation: boolean;
  nW: number;
  perturbationStd: number;
  projectionDimensions: number;
  modelType: string;
  acquisitionFamily: AcquisitionFamily;
  acquisition: string;
  beta: number;
  fitMaxiter: number;
  q: number;
  sequential: boolean;
  minimumCandidateDistanceRatio: number;
  numRestarts: number;
  rawSamples: number;
}

/** Owns model, acquisition, search, and diagnostic settings for a run. */
export function useWorkbenchRunSettings() {
  const [normalize, setNormalize] = useState(true);
  const [inputPerturbation, setInputPerturbation] = useState(false);
  const [nW, setNW] = useState(4);
  const [perturbationStd, setPerturbationStd] = useState(0.1);
  const [projectionDimensions, setProjectionDimensions] = useState(2);
  const [modelType, setModelType] = useState("base");
  const [acquisitionFamily, setAcquisitionFamilyState] = useState<AcquisitionFamily>("bayesian_optimization");
  const [acquisition, setAcquisition] = useState("EI");
  const [beta, setBeta] = useState(2);
  const [fitMaxiter, setFitMaxiter] = useState(128);
  const [crossValidation, setCrossValidation] = useState(loadCrossValidationSettings);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportanceSettings>(
    DEFAULT_FEATURE_IMPORTANCE
  );
  const [q, setQ] = useState(3);
  const [sequential, setSequential] = useState(true);
  const [minimumCandidateDistanceRatio, setMinimumCandidateDistanceRatioState] = useState(
    DEFAULT_CANDIDATE_DISTANCE_RATIO
  );
  const minimumCandidateDistanceTouched = useRef(false);
  const [numRestarts, setNumRestarts] = useState(10);
  const [rawSamples, setRawSamples] = useState(256);

  useEffect(() => saveCrossValidationSettings(crossValidation), [crossValidation]);

  function setAcquisitionFamily(nextFamily: AcquisitionFamily) {
    setAcquisitionFamilyState(nextFamily);
    if (!minimumCandidateDistanceTouched.current) {
      setMinimumCandidateDistanceRatioState(
        nextFamily === "level_set_estimation"
          ? DEFAULT_LSE_CANDIDATE_DISTANCE_RATIO
          : DEFAULT_CANDIDATE_DISTANCE_RATIO
      );
    }
  }

  function setMinimumCandidateDistanceRatio(value: number) {
    minimumCandidateDistanceTouched.current = true;
    setMinimumCandidateDistanceRatioState(value);
  }

  function resetDatasetSensitiveSettings(nextProjectionDimensions: number) {
    setNormalize(true);
    setInputPerturbation(false);
    setNW(4);
    setPerturbationStd(0.1);
    setProjectionDimensions(nextProjectionDimensions);
    setModelType("base");
    setAcquisitionFamilyState("bayesian_optimization");
    setAcquisition("EI");
    setSequential(true);
    minimumCandidateDistanceTouched.current = false;
    setMinimumCandidateDistanceRatioState(DEFAULT_CANDIDATE_DISTANCE_RATIO);
  }

  function restoreRunSettings(restored: RestoredRunSettings) {
    setNormalize(restored.normalize);
    setInputPerturbation(restored.inputPerturbation);
    setNW(restored.nW);
    setPerturbationStd(restored.perturbationStd);
    setProjectionDimensions(restored.projectionDimensions);
    setModelType(restored.modelType);
    setAcquisitionFamilyState(restored.acquisitionFamily);
    setAcquisition(restored.acquisition);
    setBeta(restored.beta);
    setFitMaxiter(restored.fitMaxiter);
    setQ(restored.q);
    setSequential(restored.sequential);
    minimumCandidateDistanceTouched.current = true;
    setMinimumCandidateDistanceRatioState(restored.minimumCandidateDistanceRatio);
    setNumRestarts(restored.numRestarts);
    setRawSamples(restored.rawSamples);
  }

  return {
    normalize,
    setNormalize,
    inputPerturbation,
    setInputPerturbation,
    nW,
    setNW,
    perturbationStd,
    setPerturbationStd,
    projectionDimensions,
    setProjectionDimensions,
    modelType,
    setModelType,
    acquisitionFamily,
    setAcquisitionFamily,
    acquisition,
    setAcquisition,
    beta,
    setBeta,
    fitMaxiter,
    setFitMaxiter,
    crossValidation,
    setCrossValidation,
    featureImportance,
    setFeatureImportance,
    q,
    setQ,
    sequential,
    setSequential,
    minimumCandidateDistanceRatio,
    setMinimumCandidateDistanceRatio,
    numRestarts,
    setNumRestarts,
    rawSamples,
    setRawSamples,
    resetDatasetSensitiveSettings,
    restoreRunSettings
  };
}
