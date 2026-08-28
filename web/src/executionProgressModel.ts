import type { LogEntry } from "./types";

export type ExecutionStage = 0 | 1 | 2;

export interface ExecutionProgressState {
  stage: ExecutionStage;
  label: string;
  failed: boolean;
  requestId?: string;
  timingsMs?: Record<string, number>;
}

export const EXECUTION_STAGES = [
  {
    label: "リクエスト受付",
    detail: "解析条件をFastAPIへ送信し、実行開始を確認"
  },
  {
    label: "バックエンド解析",
    detail: "モデル学習・候補探索・予測・可視化を実行"
  },
  {
    label: "完了",
    detail: "実行結果をResultsへ反映"
  }
] as const;

const FAILURE_EVENTS = new Set([
  "regression_run_failed",
  "http_request_failed"
]);

const WORKFLOW_EVENTS = new Set([
  "regression_run_requested",
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

export function isExecutionWorkflowEvent(event: string | undefined): boolean {
  return Boolean(event && WORKFLOW_EVENTS.has(event));
}

/** Derive progress only from backend events that actually occurred. */
export function progressFromEntries(entries: LogEntry[]): ExecutionProgressState | null {
  const workflowEntries = entries.filter((entry) => isExecutionWorkflowEvent(entry.event));
  if (!workflowEntries.length) return null;

  const last = workflowEntries.at(-1);
  const failedEntry = [...workflowEntries]
    .reverse()
    .find((entry) => FAILURE_EVENTS.has(entry.event ?? ""));
  const completedEntry = [...workflowEntries]
    .reverse()
    .find((entry) => entry.event === "workflow_completed");
  const requestId = last?.request_id ?? workflowEntries.find((entry) => entry.request_id)?.request_id;

  if (failedEntry) {
    return {
      stage: completedEntry ? 2 : 1,
      label: "解析処理に失敗しました",
      failed: true,
      requestId,
      timingsMs: timingsFromEntry(completedEntry)
    };
  }

  if (workflowEntries.some((entry) => entry.event === "regression_run_completed")) {
    return {
      stage: 2,
      label: "候補提案が完了しました",
      failed: false,
      requestId,
      timingsMs: timingsFromEntry(completedEntry)
    };
  }

  if (completedEntry) {
    return {
      stage: 2,
      label: "バックエンド解析が完了しました",
      failed: false,
      requestId,
      timingsMs: timingsFromEntry(completedEntry)
    };
  }

  return {
    stage: 1,
    label: "バックエンドで解析を実行しています",
    failed: false,
    requestId
  };
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
