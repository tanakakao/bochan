import { useCallback, useEffect, useState } from "react";
import { fetchLogs } from "./api";
import type { LogEntry } from "./types";

interface ExecutionLogsProps {
  requestId?: string;
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("ja-JP");
}

function entryDetails(entry: LogEntry): Record<string, unknown> {
  const { timestamp, level, logger, message, event, request_id, duration_ms, ...details } = entry;
  return details;
}

function latestWorkflowRequestId(entries: LogEntry[]): string | undefined {
  return [...entries]
    .reverse()
    .find((entry) =>
      Boolean(entry.request_id) &&
      ["workflow_completed", "regression_run_completed", "regression_run_failed"].includes(entry.event ?? "")
    )?.request_id;
}

export default function ExecutionLogs({ requestId }: ExecutionLogsProps) {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [resolvedRequestId, setResolvedRequestId] = useState(requestId);
  const [logFile, setLogFile] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchLogs({ limit: 300, requestId });
      const detectedRequestId = requestId ?? latestWorkflowRequestId(response.entries);
      setResolvedRequestId(detectedRequestId);
      setEntries(
        detectedRequestId
          ? response.entries.filter((entry) => entry.request_id === detectedRequestId)
          : response.entries
      );
      setLogFile(response.log_file);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLoading(false);
    }
  }, [requestId]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <section className="panel log-card">
      <div className="log-heading">
        <div>
          <span className="panel-kicker">STRUCTURED LOGS</span>
          <h3>実行ログ</h3>
          <p>
            {resolvedRequestId
              ? `表示中のリクエストID: ${resolvedRequestId}`
              : "直近のWeb APIログ"}
          </p>
          {logFile && <code>{logFile}</code>}
        </div>
        <button className="secondary" type="button" onClick={() => void load()} disabled={loading}>
          {loading ? "更新中..." : "ログを更新"}
        </button>
      </div>

      {error && <div className="alert error">{error}</div>}
      {!loading && entries.length === 0 && <div className="empty-log">対象のログはありません。</div>}

      {entries.length > 0 && (
        <div className="log-list">
          {entries.map((entry, index) => {
            const details = entryDetails(entry);
            const hasDetails = Object.keys(details).length > 0;
            return (
              <details className={`log-entry level-${entry.level.toLowerCase()}`} key={`${entry.timestamp}-${index}`}>
                <summary>
                  <span className="log-time">{formatTimestamp(entry.timestamp)}</span>
                  <span className="log-level">{entry.level}</span>
                  <span className="log-event">{entry.event ?? "log"}</span>
                  <span className="log-message">{entry.message}</span>
                  {typeof entry.duration_ms === "number" && (
                    <span className="log-duration">{entry.duration_ms.toFixed(1)} ms</span>
                  )}
                </summary>
                {hasDetails && <pre>{JSON.stringify(details, null, 2)}</pre>}
              </details>
            );
          })}
        </div>
      )}
    </section>
  );
}
