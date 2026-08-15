import { useEffect, useMemo, useRef, useState } from "react";
import { fetchLogs } from "../api";
import type { LogEntry } from "../types";

type ProgressStage = {
  label: string;
  detail: string;
};

type WorkflowProgress = {
  stage: number;
  label: string;
  percent: number;
  failed: boolean;
  hasModelFit: boolean;
  estimated: boolean;
};

const WORKFLOW_STAGES: ProgressStage[] = [
  { label: "データ準備", detail: "入力データと探索条件を準備" },
  { label: "モデル学習", detail: "代理モデルを構築・学習" },
  { label: "候補探索", detail: "獲得関数を最適化" },
  { label: "予測・可視化", detail: "候補の予測値と図を生成" },
  { label: "完了", detail: "Resultsへ反映" }
];

const EVENT_STAGE: Record<string, number> = {
  workflow_started: 0,
  workflow_data_prepared: 0,
  model_fit_started: 1,
  model_fit_completed: 1,
  candidate_generation_started: 2,
  candidate_generation_completed: 2,
  candidate_prediction_completed: 3,
  visualization_created: 3,
  visualization_failed: 3,
  workflow_completed: 4,
  regression_run_completed: 4
};

const EVENT_LABEL: Record<string, string> = {
  workflow_started: "実行を開始しました",
  workflow_data_prepared: "データ準備が完了しました",
  model_fit_started: "モデルを学習しています",
  model_fit_completed: "モデル学習が完了しました",
  candidate_generation_started: "候補を探索しています",
  candidate_generation_completed: "候補探索が完了しました",
  candidate_prediction_completed: "候補の予測値を計算しました",
  visualization_created: "可視化を生成しています",
  visualization_failed: "可視化の一部を生成できませんでした",
  workflow_completed: "ワークフローが完了しました",
  regression_run_completed: "候補提案が完了しました",
  regression_run_failed: "候補提案に失敗しました",
  model_fit_failed: "モデル学習に失敗しました",
  candidate_generation_failed: "候補探索に失敗しました"
};

const FAILURE_EVENTS = new Set([
  "regression_run_failed",
  "model_fit_failed",
  "candidate_generation_failed"
]);

const INITIAL_PROGRESS: WorkflowProgress = {
  stage: 0,
  label: "実行準備中",
  percent: 5,
  failed: false,
  hasModelFit: false,
  estimated: false
};

function isWorkflowEvent(event: string | undefined): boolean {
  if (!event) return false;
  return event in EVENT_STAGE || FAILURE_EVENTS.has(event);
}

function progressFromEntries(entries: LogEntry[]): WorkflowProgress | null {
  const workflowEntries = entries.filter((entry) => isWorkflowEvent(entry.event));
  if (workflowEntries.length === 0) return null;

  let stage = 0;
  let currentEvent = "workflow_started";
  let failed = false;
  let hasModelFit = false;

  workflowEntries.forEach((entry) => {
    const event = entry.event ?? "";
    if (event.startsWith("model_fit_")) hasModelFit = true;
    if (FAILURE_EVENTS.has(event)) failed = true;
    const mapped = EVENT_STAGE[event];
    if (mapped !== undefined && mapped >= stage) {
      stage = mapped;
      currentEvent = event;
    }
  });

  const percentages = [15, 42, 70, 90, 100];
  const percent = failed ? Math.max(percentages[stage] - 5, 10) : percentages[stage];
  const lastEvent = workflowEntries.at(-1)?.event ?? "";
  const label = failed
    ? EVENT_LABEL[lastEvent] ?? "処理に失敗しました"
    : EVENT_LABEL[currentEvent] ?? WORKFLOW_STAGES[stage].label;

  return {
    stage,
    label,
    percent,
    failed,
    hasModelFit,
    estimated: false
  };
}

function estimatedProgress(elapsedMs: number): WorkflowProgress {
  const elapsedSeconds = elapsedMs / 1000;
  if (elapsedSeconds < 2) {
    return {
      stage: 0,
      percent: 10 + elapsedSeconds * 4,
      label: "データと探索条件を準備しています",
      failed: false,
      hasModelFit: true,
      estimated: true
    };
  }
  if (elapsedSeconds < 20) {
    const ratio = (elapsedSeconds - 2) / 18;
    return {
      stage: 1,
      percent: 18 + ratio * 37,
      label: "モデル学習を実行しています",
      failed: false,
      hasModelFit: true,
      estimated: true
    };
  }
  if (elapsedSeconds < 50) {
    const ratio = (elapsedSeconds - 20) / 30;
    return {
      stage: 2,
      percent: 55 + ratio * 25,
      label: "候補探索を実行しています",
      failed: false,
      hasModelFit: true,
      estimated: true
    };
  }
  const ratio = Math.min(1, (elapsedSeconds - 50) / 60);
  return {
    stage: 3,
    percent: 80 + ratio * 10,
    label: "候補の予測・可視化を処理しています",
    failed: false,
    hasModelFit: true,
    estimated: true
  };
}

export default function ExecutionProgress({ busy }: { busy: string }) {
  const workflowRun = useMemo(
    () => /候補|モデル.*学習|学習済みモデル/.test(busy),
    [busy]
  );
  const startedAtRef = useRef(Date.now());
  const requestIdRef = useRef<string | undefined>(undefined);
  const backendProgressSeenRef = useRef(false);
  const [progress, setProgress] = useState<WorkflowProgress>(INITIAL_PROGRESS);

  useEffect(() => {
    startedAtRef.current = Date.now();
    requestIdRef.current = undefined;
    backendProgressSeenRef.current = false;
    setProgress(INITIAL_PROGRESS);
    if (!workflowRun) return;

    let active = true;

    const poll = async () => {
      try {
        const response = await fetchLogs({
          limit: 250,
          requestId: requestIdRef.current
        });
        if (!active) return;

        const recent = response.entries.filter((entry) => {
          const timestamp = new Date(entry.timestamp).getTime();
          return Number.isFinite(timestamp) && timestamp >= startedAtRef.current - 3000;
        });

        if (!requestIdRef.current) {
          requestIdRef.current = [...recent]
            .reverse()
            .find((entry) => Boolean(entry.request_id) && isWorkflowEvent(entry.event))
            ?.request_id;
        }

        const entries = requestIdRef.current
          ? response.entries.filter((entry) => entry.request_id === requestIdRef.current)
          : recent;
        const next = progressFromEntries(entries);
        if (next) {
          backendProgressSeenRef.current = true;
          setProgress(next);
        }
      } catch {
        // Progress is supplemental. The fallback below keeps the UI moving when logs are unavailable.
      }
    };

    void poll();
    const pollTimer = window.setInterval(() => void poll(), 1200);
    const fallbackTimer = window.setInterval(() => {
      if (!active || backendProgressSeenRef.current) return;
      const estimated = estimatedProgress(Date.now() - startedAtRef.current);
      setProgress({ ...estimated, percent: Math.max(6, Math.min(90, estimated.percent)) });
    }, 650);

    return () => {
      active = false;
      window.clearInterval(pollTimer);
      window.clearInterval(fallbackTimer);
    };
  }, [busy, workflowRun]);

  if (!workflowRun) {
    return (
      <div className="execution-progress-panel generic" aria-live="polite">
        <div className="execution-progress-head">
          <span>PROCESS</span>
          <strong>{busy}</strong>
          <em>実行中</em>
        </div>
        <div className="execution-progress-track indeterminate"><span /></div>
      </div>
    );
  }

  return (
    <div
      className={`execution-progress-panel workflow ${progress.failed ? "failed" : ""} ${progress.estimated ? "estimated-fallback" : ""}`}
      aria-live="polite"
    >
      <div className="execution-progress-head">
        <span>RUN PROGRESS</span>
        <strong className="execution-progress-label">{progress.label}</strong>
        <em className="execution-progress-percent">
          {progress.failed
            ? "要確認"
            : progress.estimated
              ? `約${Math.round(progress.percent)}%`
              : `${Math.round(progress.percent)}%`}
        </em>
      </div>
      <div className="execution-progress-track">
        <span style={{ width: `${progress.percent}%` }} />
      </div>
      <ol className="execution-stage-list">
        {WORKFLOW_STAGES.map((stage, index) => {
          const skippedModel = index === 1 && progress.stage >= 2 && !progress.hasModelFit;
          const className = [
            index < progress.stage && !skippedModel ? "complete" : "",
            index === progress.stage && !progress.failed ? "active" : "",
            index === progress.stage && progress.failed ? "failed" : "",
            skippedModel ? "skipped" : ""
          ].filter(Boolean).join(" ");
          return (
            <li key={stage.label} data-stage={index} className={className}>
              <span className="execution-stage-marker">{index + 1}</span>
              <span>
                <strong>{stage.label}</strong>
                <small>{skippedModel ? "学習済みモデルを再利用" : stage.detail}</small>
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}