import { useEffect, useMemo, useRef, useState } from "react";
import { fetchLogs } from "../api";
import {
  EXECUTION_STAGES,
  formatElapsed,
  isExecutionWorkflowEvent,
  progressFromEntries,
  timingSummary,
  type ExecutionProgressState
} from "../executionProgressModel";
import "../styles/execution-progress.css";

const INITIAL_PROGRESS: ExecutionProgressState = {
  stage: 0,
  label: "FastAPIの受付を待っています",
  failed: false
};

export default function ExecutionProgress({ busy }: { busy: string }) {
  const workflowRun = useMemo(
    () => /候補|モデル.*学習|学習済みモデル/.test(busy),
    [busy]
  );
  const startedAtRef = useRef(Date.now());
  const requestIdRef = useRef<string | undefined>(undefined);
  const [now, setNow] = useState(Date.now());
  const [progress, setProgress] = useState<ExecutionProgressState>(INITIAL_PROGRESS);

  useEffect(() => {
    startedAtRef.current = Date.now();
    requestIdRef.current = undefined;
    setNow(Date.now());
    setProgress(INITIAL_PROGRESS);

    const elapsedTimer = window.setInterval(() => setNow(Date.now()), 250);
    if (!workflowRun) {
      return () => window.clearInterval(elapsedTimer);
    }

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
            .find((entry) => Boolean(entry.request_id) && isExecutionWorkflowEvent(entry.event))
            ?.request_id;
        }

        const entries = requestIdRef.current
          ? response.entries.filter((entry) => entry.request_id === requestIdRef.current)
          : recent;
        const next = progressFromEntries(entries);
        if (next) setProgress(next);
      } catch {
        // Progress is supplemental. Do not fabricate a percentage or stage when logs are unavailable.
      }
    };

    void poll();
    const pollTimer = window.setInterval(() => void poll(), 800);

    return () => {
      active = false;
      window.clearInterval(pollTimer);
      window.clearInterval(elapsedTimer);
    };
  }, [busy, workflowRun]);

  const elapsed = formatElapsed(now - startedAtRef.current);

  if (!workflowRun) {
    return (
      <div className="execution-progress-panel generic" aria-live="polite">
        <div className="execution-progress-head">
          <span>PROCESS</span>
          <strong>{busy}</strong>
          <em>経過 {elapsed}</em>
        </div>
        <div className="execution-progress-track indeterminate"><span /></div>
      </div>
    );
  }

  const measuredTimings = timingSummary(progress.timingsMs);

  return (
    <div
      className={`execution-progress-panel workflow ${progress.failed ? "failed" : ""}`}
      aria-live="polite"
    >
      <div className="execution-progress-head">
        <span>LIVE PROCESS</span>
        <strong className="execution-progress-label">{progress.label}</strong>
        <em className="execution-progress-percent">
          {progress.failed ? "要確認" : `経過 ${elapsed}`}
        </em>
      </div>

      <div
        className={`execution-progress-track ${progress.stage < 2 && !progress.failed ? "indeterminate" : ""}`}
        aria-label={`実イベント段階 ${progress.stage + 1} / ${EXECUTION_STAGES.length}`}
      >
        <span style={progress.stage >= 2 ? { width: "100%" } : undefined} />
      </div>

      <ol className="execution-stage-list">
        {EXECUTION_STAGES.map((stage, index) => {
          const className = [
            index < progress.stage ? "complete" : "",
            index === progress.stage && !progress.failed ? "active" : "",
            index === progress.stage && progress.failed ? "failed" : ""
          ].filter(Boolean).join(" ");
          return (
            <li key={stage.label} data-stage={index} className={className}>
              <span className="execution-stage-marker">{index + 1}</span>
              <span>
                <strong>{stage.label}</strong>
                <small>{stage.detail}</small>
              </span>
            </li>
          );
        })}
      </ol>

      <div className="execution-progress-foot">
        <span>進捗はbackendの実イベントのみ表示</span>
        {requestIdRef.current && <code>request {requestIdRef.current.slice(0, 8)}</code>}
      </div>

      {measuredTimings.length > 0 && (
        <div className="execution-timing-summary" aria-label="実測処理時間">
          {measuredTimings.map((timing) => <span key={timing}>{timing}</span>)}
        </div>
      )}
    </div>
  );
}
