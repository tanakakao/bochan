import { createContext, useContext, useMemo, type ReactNode } from "react";
import {
  buildModelReuseSignature,
  runRegression,
  uploadDataset,
  uploadModelArtifact,
  type RunRegressionInput
} from "../api";
import { restoreWorkbenchFromArtifact } from "../modelArtifactRestore";
import { createInitialSelectionState, numberOrUndefined } from "./workbenchDefaults";
import { useWorkbenchResultState } from "./useWorkbenchResultState";
import { useWorkbenchRuntimeState } from "./useWorkbenchRuntimeState";
import { useWorkbenchRunSettings } from "./useWorkbenchRunSettings";
import { useWorkbenchSelectionState } from "./useWorkbenchSelectionState";
import {
  STEPS,
  type ModelExecutionMode,
  type WorkbenchContextValue,
  type WorkbenchStep
} from "./workbenchTypes";
import { deriveWorkbenchState } from "./workbenchValidation";

export { STEPS } from "./workbenchTypes";
export type {
  CrossValidationSettings,
  HealthState,
  ModelExecutionMode,
  Theme,
  WorkbenchContextValue,
  WorkbenchStep
} from "./workbenchTypes";

const WorkbenchContext = createContext<WorkbenchContextValue | null>(null);

/** Composes the workbench domain hooks and coordinates cross-domain API actions. */
export function WorkbenchProvider({ children }: { children: ReactNode }) {
  const runtime = useWorkbenchRuntimeState();
  const selection = useWorkbenchSelectionState();
  const settings = useWorkbenchRunSettings();
  const results = useWorkbenchResultState();

  const derived = useMemo(() => deriveWorkbenchState({
    dataset: selection.dataset,
    featureColumns: selection.featureColumns,
    targetColumns: selection.targetColumns,
    targetSettings: selection.targetSettings,
    variables: selection.variables,
    inputPerturbation: settings.inputPerturbation,
    nW: settings.nW,
    perturbationStd: settings.perturbationStd,
    projectionDimensions: settings.projectionDimensions,
    modelType: settings.modelType,
    fitMaxiter: settings.fitMaxiter
  }), [
    selection.dataset,
    selection.featureColumns,
    selection.targetColumns,
    selection.targetSettings,
    selection.variables,
    settings.inputPerturbation,
    settings.nW,
    settings.perturbationStd,
    settings.projectionDimensions,
    settings.modelType,
    settings.fitMaxiter
  ]);

  const currentRunInput: RunRegressionInput | null = selection.dataset ? {
    datasetId: selection.dataset.dataset_id,
    featureColumns: selection.featureColumns,
    targetColumn: derived.targetColumn,
    targetColumns: selection.targetColumns,
    targetSettings: derived.selectedTargetSettings,
    targetDirections: derived.targetDirections,
    direction: derived.direction,
    modelType: settings.modelType,
    projectionDimensions: settings.projectionDimensions,
    fitMaxiter: settings.fitMaxiter,
    normalize: settings.normalize,
    inputPerturbation: settings.inputPerturbation,
    nW: settings.nW,
    perturbationStd: settings.perturbationStd,
    acquisitionFamily: settings.acquisitionFamily,
    acquisition: settings.acquisition,
    beta: settings.beta,
    q: settings.q,
    sequential: settings.sequential,
    minimumCandidateDistanceRatio: settings.minimumCandidateDistanceRatio,
    numRestarts: settings.numRestarts,
    rawSamples: settings.rawSamples,
    searchSpace: derived.selectedVariables,
    crossValidation: settings.crossValidation,
    featureImportance: settings.featureImportance
  } : null;
  const currentModelSignature = currentRunInput
    ? buildModelReuseSignature(currentRunInput)
    : null;
  const modelReuseAvailable = Boolean(
    derived.candidateSettingsValid &&
    !results.result?.metadata?.stale_after_data_append &&
    results.result?.visualization_run_id &&
    results.lastModelSignature &&
    currentModelSignature === results.lastModelSignature
  );

  function canOpenStep(nextStep: WorkbenchStep): boolean {
    if (nextStep === "data" || nextStep === "logs") return true;
    if (nextStep === "prepare") return Boolean(selection.dataset);
    if (nextStep === "settings") return derived.canConfigure;
    if (nextStep === "optimize") return derived.settingsValid;
    if (nextStep === "results") return Boolean(results.result);
    return false;
  }

  function setStep(nextStep: WorkbenchStep) {
    if (canOpenStep(nextStep)) runtime.setStepState(nextStep);
  }

  async function handleFile(file: File | null) {
    if (!file) return;
    runtime.setBusy("データを読み込んでいます");
    runtime.setError(null);
    results.clearResult();
    try {
      const loaded = await uploadDataset(file);
      const initial = createInitialSelectionState(loaded);
      selection.replaceSelection(initial);
      settings.resetDatasetSensitiveSettings(initial.projectionDimensions);
      runtime.setStepState("prepare");
    } catch (caught) {
      runtime.setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      runtime.setBusy(null);
    }
  }

  async function handleModelArtifact(file: File | null) {
    if (!file) return;
    runtime.setBusy("保存モデルを読み込んでいます");
    runtime.setError(null);
    try {
      const imported = await uploadModelArtifact(file);
      const restored = restoreWorkbenchFromArtifact(imported);
      selection.replaceSelection({
        dataset: imported.dataset,
        featureColumns: restored.featureColumns,
        targetColumns: restored.targetColumns,
        targetSettings: restored.targetSettings,
        variables: restored.variables
      });
      settings.restoreRunSettings(restored);
      results.setResult(imported.result);
      results.setLastModelSignature(restored.modelSignature);
      runtime.setStepState("results");
    } catch (caught) {
      runtime.setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      runtime.setBusy(null);
    }
  }

  async function execute(mode: ModelExecutionMode = "retrain") {
    if (!currentRunInput || !derived.candidateSettingsValid) return;
    if (
      settings.featureImportance.enabled &&
      settings.featureImportance.source === "cross_validation" &&
      !settings.crossValidation.enabled
    ) {
      runtime.setError("Cross-validation feature importance requires cross_validation=true.");
      return;
    }
    const modelSignature = currentModelSignature ?? buildModelReuseSignature(currentRunInput);
    const reusableRunId = results.result?.visualization_run_id;
    const canReuse = Boolean(
      reusableRunId &&
      !results.result?.metadata?.stale_after_data_append &&
      results.lastModelSignature === modelSignature
    );
    if (mode === "reuse" && !canReuse) {
      runtime.setError(
        "学習済みモデルを使用できません。データ、タスク、モデル、前処理、欠損処理、または探索範囲が変更されています。"
      );
      return;
    }
    const reuseModel = mode === "reuse";
    const input: RunRegressionInput = {
      ...currentRunInput,
      reuseModelRunId: reuseModel ? reusableRunId : undefined
    };

    runtime.setBusy(reuseModel
      ? "学習済みモデルを使用して候補提案を実行しています"
      : "モデルを再学習して候補提案を実行しています");
    runtime.setError(null);
    try {
      const response = await runRegression(input);
      results.setResult(response);
      results.setLastModelSignature(modelSignature);
      runtime.setStepState("results");
    } catch (caught) {
      runtime.setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      runtime.setBusy(null);
    }
  }

  const value: WorkbenchContextValue = {
    theme: runtime.theme,
    setTheme: runtime.setTheme,
    step: runtime.step,
    setStep,
    canOpenStep,
    health: runtime.health,
    busy: runtime.busy,
    error: runtime.error,
    setError: runtime.setError,
    dataset: selection.dataset,
    columns: derived.columns,
    selectableColumns: derived.selectableColumns,
    targetCandidates: derived.targetCandidates,
    featureColumns: selection.featureColumns,
    targetColumn: derived.targetColumn,
    targetColumns: selection.targetColumns,
    targetSettings: selection.targetSettings,
    selectedTargetSettings: derived.selectedTargetSettings,
    optimizedTargetSettings: derived.optimizedTargetSettings,
    targetDirections: derived.targetDirections,
    direction: derived.direction,
    variables: selection.variables,
    selectedVariables: derived.selectedVariables,
    normalize: settings.normalize,
    setNormalize: settings.setNormalize,
    inputPerturbation: settings.inputPerturbation,
    setInputPerturbation: settings.setInputPerturbation,
    nW: settings.nW,
    setNW: settings.setNW,
    perturbationStd: settings.perturbationStd,
    setPerturbationStd: settings.setPerturbationStd,
    projectionDimensions: settings.projectionDimensions,
    setProjectionDimensions: settings.setProjectionDimensions,
    modelType: settings.modelType,
    setModelType: settings.setModelType,
    acquisitionFamily: settings.acquisitionFamily,
    setAcquisitionFamily: settings.setAcquisitionFamily,
    acquisition: settings.acquisition,
    setAcquisition: settings.setAcquisition,
    beta: settings.beta,
    setBeta: settings.setBeta,
    fitMaxiter: settings.fitMaxiter,
    setFitMaxiter: settings.setFitMaxiter,
    crossValidation: settings.crossValidation,
    setCrossValidation: settings.setCrossValidation,
    featureImportance: settings.featureImportance,
    setFeatureImportance: settings.setFeatureImportance,
    q: settings.q,
    setQ: settings.setQ,
    sequential: settings.sequential,
    setSequential: settings.setSequential,
    minimumCandidateDistanceRatio: settings.minimumCandidateDistanceRatio,
    setMinimumCandidateDistanceRatio: settings.setMinimumCandidateDistanceRatio,
    numRestarts: settings.numRestarts,
    setNumRestarts: settings.setNumRestarts,
    rawSamples: settings.rawSamples,
    setRawSamples: settings.setRawSamples,
    result: results.result,
    canConfigure: derived.canConfigure,
    settingsValid: derived.settingsValid,
    candidateSettingsValid: derived.candidateSettingsValid,
    modelReuseAvailable,
    handleFile,
    handleModelArtifact,
    toggleFeature: selection.toggleFeature,
    toggleTarget: selection.toggleTarget,
    patchTargetSetting: selection.patchTargetSetting,
    patchVariable: selection.patchVariable,
    execute,
    numberOrUndefined
  };

  return <WorkbenchContext.Provider value={value}>{children}</WorkbenchContext.Provider>;
}

export function useWorkbench(): WorkbenchContextValue {
  const value = useContext(WorkbenchContext);
  if (!value) throw new Error("useWorkbench must be used inside WorkbenchProvider.");
  return value;
}
