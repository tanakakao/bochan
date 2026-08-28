import type { RegressionResult } from "../../types";
import type { WorkbenchStep } from "../../context/WorkbenchContext";

export type WorkflowStepStatus = {
  complete: boolean;
  available: boolean;
  stale: boolean;
  label: string;
};

export type WorkflowCompletionInput = {
  hasDataset: boolean;
  canConfigure: boolean;
  settingsValid: boolean;
  candidateSettingsValid: boolean;
  result: RegressionResult | null;
};

function status(
  complete: boolean,
  available: boolean,
  label: string,
  stale = false
): WorkflowStepStatus {
  return {
    complete,
    available: available || complete,
    stale,
    label
  };
}

/**
 * Derives workflow state from actual workspace artifacts and validation state.
 * Navigation order is deliberately not part of the completion decision.
 */
export function getWorkflowCompletion({
  hasDataset,
  canConfigure,
  settingsValid,
  candidateSettingsValid,
  result
}: WorkflowCompletionInput): Record<WorkbenchStep, WorkflowStepStatus> {
  const hasResult = Boolean(result);
  const resultStale = Boolean(result?.metadata?.stale_after_data_append);
  const freshResult = hasResult && !resultStale;

  return {
    data: status(hasDataset, true, hasDataset ? "読込済み" : "未読込"),
    prepare: status(canConfigure, hasDataset, canConfigure ? "選択済み" : "未完了"),
    settings: status(settingsValid, canConfigure, settingsValid ? "設定済み" : "未完了"),
    optimize: status(
      freshResult,
      candidateSettingsValid,
      resultStale ? "更新必要" : freshResult ? "提案済み" : "未実行",
      resultStale
    ),
    results: status(
      freshResult,
      hasResult,
      resultStale ? "更新必要" : freshResult ? "結果あり" : "未実行",
      resultStale
    ),
    logs: status(false, true, "任意")
  };
}

export function workflowStatusText(stepStatus: WorkflowStepStatus): string {
  if (stepStatus.stale) return "更新必要";
  if (stepStatus.complete) return stepStatus.label;
  if (stepStatus.available) return "実行可能";
  return "未完了";
}
