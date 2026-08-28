import type { WorkbenchStep } from "../../context/workbenchTypes";
import type { RegressionResult } from "../../types";

export interface WorkflowStepStatus {
  complete: boolean;
  available: boolean;
  optional?: boolean;
  label: string;
}

export interface WorkflowCompletionInput {
  hasDataset: boolean;
  canConfigure: boolean;
  settingsValid: boolean;
  candidateSettingsValid: boolean;
  result: RegressionResult | null;
}

function status(
  complete: boolean,
  available: boolean,
  label: string,
  optional = false
): WorkflowStepStatus {
  return {
    complete,
    available: available || complete,
    optional,
    label
  };
}

/**
 * Derive workflow completion from actual workbench state rather than navigation order.
 *
 * A step only becomes complete once its domain outcome exists or validates. Merely
 * navigating past a page never produces a completion check.
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
  const hasCurrentResult = hasResult && !resultStale;

  return {
    data: status(hasDataset, true, hasDataset ? "読込済み" : "未読込"),
    prepare: status(canConfigure, hasDataset, canConfigure ? "選択済み" : "設定待ち"),
    settings: status(settingsValid, canConfigure, settingsValid ? "設定済み" : "設定待ち"),
    optimize: status(
      hasCurrentResult,
      candidateSettingsValid,
      hasCurrentResult ? "提案済み" : resultStale ? "再実行が必要" : "実行可能"
    ),
    results: status(
      hasCurrentResult,
      hasResult,
      hasCurrentResult ? "結果あり" : resultStale ? "旧結果" : "未実行"
    ),
    logs: status(false, true, "任意", true)
  };
}

export function workflowStatusText(stepStatus: WorkflowStepStatus): string {
  if (stepStatus.complete) return stepStatus.label;
  if (stepStatus.optional) return "任意";
  if (stepStatus.available) return stepStatus.label || "実行可能";
  return stepStatus.label || "未完了";
}
