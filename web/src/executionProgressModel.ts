import type { LogEntry } from "./types";

export type ExecutionStage = 0 | 1 | 2 | 3 | 4;

export interface ExecutionProgressState {
  stage: ExecutionStage;
  completedStage: number;
  label: string;
  failed: boolean;
  requestId?: string;
  timingsMs?: Record<string, number>;
}

export const EXECUTION_STAGES = [
  {
    label: "データ準備",
    detail: "入力データ・型・探索条件を準備"
  },
  {
    label: "モデル学習",
    detail: "CVと最終モデルを実際に学習"
  },
  {
    label: "候補探索",
    detail: "獲得関数を最適化して候補を生成"
  },
  {
    label: "予測・Results生成",
    detail: "候補後処理・予測・Results用図を生成"
  },
  {
    label: "完了",
    detail: "実行結果をResultsへ反映"
  }
] as const;

const GENERIC_FAILURE_EVENTS = new Set([
  "regression_run_failed",
  "http_request_failed"
]);

const FAILURE_EVENTS = new Set([
  ...GENERIC_FAILURE_EVENTS,
  "model_fit_failed",
  "model_output_fit_failed",
  "candidate_generation_failed"
]);

const WORKFLOW_EVENTS = new Set([
  "regression_run_requested",
  "workflow_started",
  "workflow_data_prepared",
  "model_fit_started",
  "model_fit_completed",
  "model_fit_failed",
  "model_output_fit_started",
  "model_output_fit_completed",
  "model_output_fit_failed",
  "model_reuse_completed",
  "candidate_generation_started",
  "candidate_generation_completed",
  "candidate_generation_failed",
  "workflow_completed",
  "regression_run_completed",
  ...FAILURE_EVENTS
]);

function timingsFromEntry(entry: LogEntry | undefined): Record<string, number> | undefined {
  const value = entry?.timings_ms;
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const timings = Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .map(([key, raw]) => [key, Number(raw)] as const)
      .filter(([, raw]) => Number.isFinite(raw) && raw >= 0)
  );
  return Object.keys(timings).length ? timings : undefined;
}

function positiveInteger(value: unknown): number | undefined {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function fitLabel(entry: LogEntry, completed: boolean): string {
  const phase = String(entry.fit_phase ?? "");
  const foldCurrent = positiveInteger(entry.fold_current);
  const foldTotal = positiveInteger(entry.fold_total);
  const outputIndex = positiveInteger(entry.output_index);
  const outputTotal = positiveInteger(entry.output_total);
  const outputName = String(entry.output_name ?? "").trim();
  const fitMode = String(entry.fit_mode ?? "");

  const outputPart = outputIndex && outputTotal
    ? `目的変数 ${outputIndex} / ${outputTotal}${outputName ? `: ${outputName}` : ""}`
    : "";
  if (phase === "cross_validation" && foldCurrent && foldTotal) {
    const fold = `CV fold ${foldCurrent} / ${foldTotal}`;
    if (outputPart) return `${fold}・${outputPart}${completed ? " 完了" : " を学習中"}`;
    return `${fold}${completed ? " が完了しました" : " を学習しています"}`;
  }
  if (phase === "cross_validation_or_final") {
    if (outputPart) {
      return `CV/最終fit・${outputPart}${completed ? " 完了" : " を学習中"}`;
    }
    return completed
      ? "CV/最終モデルの学習処理が1回完了しました"
      : "CV/最終モデルを学習しています";
  }
  if (outputPart) {
    return `${outputPart}${completed ? " の学習完了" : " を学習しています"}`;
  }
  if (fitMode === "joint" && outputTotal && outputTotal > 1) {
    return completed
      ? `${outputTotal}目的の共同モデル学習が完了しました`
      : `${outputTotal}目的を共同学習しています`;
  }
  return completed ? "最終モデル学習が完了しました" : "最終モデルを学習しています";
}

function failureLabel(stage: ExecutionStage): string {
  if (stage === 0) return "データ準備に失敗しました";
  if (stage === 1) return "モデル学習に失敗しました";
  if (stage === 2) return "候補探索に失敗しました";
  if (stage === 3) return "予測・Results生成に失敗しました";
  return "解析処理に失敗しました";
}

function stateForEntry(
  entry: LogEntry,
  requestId: string | undefined,
  timingsMs: Record<string, number> | undefined
): ExecutionProgressState {
  const event = entry.event ?? "";
  const failed = FAILURE_EVENTS.has(event);
  if (event === "regression_run_requested" || event === "workflow_started") {
    return { stage: 0, completedStage: -1, label: "データと探索条件を準備しています", failed, requestId };
  }
  if (event === "workflow_data_prepared") {
    return { stage: 1, completedStage: 0, label: "データ準備が完了しました", failed, requestId };
  }
  if (event === "model_fit_started" || event === "model_output_fit_started") {
    return { stage: 1, completedStage: 0, label: fitLabel(entry, false), failed, requestId };
  }
  if (event === "model_output_fit_completed") {
    return { stage: 1, completedStage: 0, label: fitLabel(entry, true), failed, requestId };
  }
  if (event === "model_fit_completed") {
    const phase = String(entry.fit_phase ?? "");
    if (phase === "cross_validation" || phase === "cross_validation_or_final") {
      return { stage: 1, completedStage: 0, label: fitLabel(entry, true), failed, requestId };
    }
    return { stage: 2, completedStage: 1, label: fitLabel(entry, true), failed, requestId };
  }
  if (event === "model_reuse_completed") {
    return { stage: 2, completedStage: 1, label: "学習済みモデルを再利用しました", failed, requestId };
  }
  if (event === "candidate_generation_started") {
    return {
      stage: 2,
      completedStage: 1,
      label: "候補を探索しています",
      failed,
      requestId
    };
  }
  if (event === "candidate_generation_completed") {
    return { stage: 3, completedStage: 2, label: "候補探索が完了し、予測・Results生成を処理しています", failed, requestId };
  }
  if (event === "workflow_completed") {
    return {
      stage: 3,
      completedStage: 2,
      label: "主要解析が完了し、Resultsを最終構成しています",
      failed,
      requestId,
      timingsMs
    };
  }
  if (event === "regression_run_completed") {
    return {
      stage: 4,
      completedStage: 4,
      label: "候補提案が完了しました",
      failed,
      requestId,
      timingsMs
    };
  }

  if (failed) {
    const stage: ExecutionStage = event.startsWith("candidate_") ? 2 : event.startsWith("model_") ? 1 : 3;
    return {
      stage,
      completedStage: Math.max(-1, stage - 1),
      label: failureLabel(stage),
      failed: true,
      requestId,
      timingsMs
    };
  }

  return { stage: 0, completedStage: -1, label: "FastAPIの受付を待っています", failed: false, requestId };
}

export function isExecutionWorkflowEvent(event: string | undefined): boolean {
  return Boolean(event && WORKFLOW_EVENTS.has(event));
}

/** Derive progress only from backend events that actually occurred. */
export function progressFromEntries(entries: LogEntry[]): ExecutionProgressState | null {
  const workflowEntries = entries.filter((entry) => isExecutionWorkflowEvent(entry.event));
  if (!workflowEntries.length) return null;

  const last = workflowEntries.at(-1)!;
  const completedEntry = [...workflowEntries]
    .reverse()
    .find((entry) => entry.event === "workflow_completed");
  const requestId = last.request_id ?? workflowEntries.find((entry) => entry.request_id)?.request_id;
  const timingsMs = timingsFromEntry(completedEntry);

  if (GENERIC_FAILURE_EVENTS.has(last.event ?? "")) {
    const previous = [...workflowEntries.slice(0, -1)]
      .reverse()
      .find((entry) => !GENERIC_FAILURE_EVENTS.has(entry.event ?? ""));
    if (previous) {
      const previousState = stateForEntry(previous, requestId, timingsMs);
      return {
        ...previousState,
        failed: true,
        label: failureLabel(previousState.stage)
      };
    }
  }

  return stateForEntry(last, requestId, timingsMs);
}

export function formatElapsed(elapsedMs: number): string {
  const safeMs = Math.max(0, elapsedMs);
  const totalSeconds = safeMs / 1000;
  if (totalSeconds < 60) return `${totalSeconds.toFixed(totalSeconds < 10 ? 1 : 0)}秒`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}分${String(seconds).padStart(2, "0")}秒`;
}

export function timingSummary(timingsMs: Record<string, number> | undefined): string[] {
  if (!timingsMs) return [];
  const labels: Array<[string, string]> = [
    ["prepare", "準備"],
    ["fit", "学習"],
    ["feature_importance", "重要度"],
    ["candidate", "候補探索"],
    ["prediction", "予測"],
    ["visualization", "可視化"],
    ["total", "合計"]
  ];
  return labels.flatMap(([key, label]) => {
    const value = timingsMs[key];
    return Number.isFinite(value) ? [`${label} ${formatElapsed(value)}`] : [];
  });
}
