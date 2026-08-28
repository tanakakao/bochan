import { useEffect, useState } from "react";
import {
  STEPS,
  type WorkbenchStep,
  useWorkbench
} from "../../context/WorkbenchContext";
import { useWorkbenchMode } from "../../workbenchMode";
import {
  clearAuxiliaryHash,
  currentAuxiliaryPage,
  type AuxiliaryPage
} from "./workbenchPages";
import {
  getWorkflowCompletion,
  workflowStatusText
} from "./workflowCompletion";

const CONTEXT_COLLAPSE_KEY = "bochan-context-rail-collapsed";

/** Owns shell-only routing, semantic progress, tutorial, and rail-collapse state. */
export function useWorkbenchShell() {
  const mode = useWorkbenchMode();
  const [auxiliaryPage, setAuxiliaryPage] = useState<AuxiliaryPage | null>(currentAuxiliaryPage);
  const [tutorialRequest, setTutorialRequest] = useState(0);
  const [contextRailCollapsed, setContextRailCollapsed] = useState(
    () => window.localStorage.getItem(CONTEXT_COLLAPSE_KEY) === "1"
  );
  const {
    step,
    setStep,
    canOpenStep,
    dataset,
    result,
    resultCurrent,
    canConfigure,
    settingsValid,
    candidateSettingsValid
  } = useWorkbench();

  const workflowSteps = STEPS.filter(([id]) => id !== "logs");
  const visibleSteps = mode === "simple"
    ? workflowSteps.filter(([id]) => id === "data" || id === "prepare" || id === "results")
    : workflowSteps;
  const index = visibleSteps.findIndex(([id]) => id === step);
  const workflowCompletion = getWorkflowCompletion({
    hasDataset: Boolean(dataset),
    canConfigure,
    settingsValid,
    candidateSettingsValid,
    result,
    resultCurrent
  });
  const completedStepCount = visibleSteps.filter(([id]) => workflowCompletion[id].complete).length;
  const experimentAvailable = Boolean(dataset && result);
  const activeAuxiliaryPage: AuxiliaryPage | null = auxiliaryPage === "conversation"
    ? "conversation"
    : auxiliaryPage === "experiment" && experimentAvailable
      ? "experiment"
      : null;
  const progressStepIndex = activeAuxiliaryPage === "experiment"
    ? visibleSteps.length - 1
    : Math.max(index, 0);
  const progressLabel = activeAuxiliaryPage === "conversation"
    ? "対話モード"
    : activeAuxiliaryPage === "experiment"
      ? "実験結果追加"
      : visibleSteps[progressStepIndex]?.[1] ?? "データ";
  const progressMeta = activeAuxiliaryPage === "conversation"
    ? "GUIDED FLOW"
    : activeAuxiliaryPage === "experiment"
      ? "EXPERIMENT"
      : `${completedStepCount} / ${visibleSteps.length} COMPLETE`;
  const progressPercent = visibleSteps.length
    ? Math.min(100, Math.max(0, (completedStepCount / visibleSteps.length) * 100))
    : 0;

  useEffect(() => {
    const handleHashChange = () => setAuxiliaryPage(currentAuxiliaryPage());
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(CONTEXT_COLLAPSE_KEY, contextRailCollapsed ? "1" : "0");
  }, [contextRailCollapsed]);

  useEffect(() => {
    if (auxiliaryPage === "experiment" && !experimentAvailable) {
      setStep("data");
      clearAuxiliaryHash();
    }
  }, [auxiliaryPage, experimentAvailable, setStep]);

  useEffect(() => {
    if (step === "logs") {
      setStep(result ? "results" : dataset ? "prepare" : "data");
      return;
    }
    if (mode === "simple" && (step === "settings" || step === "optimize")) {
      setStep(dataset ? "prepare" : "data");
    }
  }, [dataset, mode, result, setStep, step]);

  function isComplete(id: WorkbenchStep): boolean {
    return workflowCompletion[id]?.complete ?? false;
  }

  function getStatusText(id: WorkbenchStep): string {
    return workflowStatusText(workflowCompletion[id]);
  }

  function openStep(id: WorkbenchStep) {
    if (auxiliaryPage) clearAuxiliaryHash();
    setStep(id);
  }

  function openConversation() {
    window.location.hash = "conversation";
  }

  function openExperiment() {
    if (!experimentAvailable) return;
    window.location.hash = "experiment";
  }

  return {
    mode,
    step,
    visibleSteps,
    activeAuxiliaryPage,
    experimentAvailable,
    progressLabel,
    progressMeta,
    progressPercent,
    tutorialRequest,
    requestTutorial: () => setTutorialRequest((current) => current + 1),
    contextRailCollapsed,
    toggleContextRail: () => setContextRailCollapsed((current) => !current),
    canOpenStep,
    isComplete,
    getStatusText,
    openStep,
    openConversation,
    openExperiment
  };
}
