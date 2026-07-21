import { useCallback, useEffect, useMemo, useState } from "react";
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
      ["model_details", "workflow_completed", "regression_run_completed", "regression_run_failed"].includes(entry.event ?? "")
    )?.request_id;
}

function findModelDetails(entries: LogEntry[]): Record<string, unknown> | null {
  for (const entry of [...entries].reverse()) {
    const value = entry.model_details;
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return value as Record<string, unknown>;
    }
  }
  return null;
}

function display(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (Array.isArray(value)) return value.length ? value.map(String).join(" / ") : "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
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

  const details = useMemo(() => findModelDetails(entries), [entries]);
  const outputSpecs = Array.isArray(details?.output_specs)
    ? details.output_specs as Array<Record<string, unknown>>
    : [];

  return (
    <>
      {details && (
        <section className="panel model-details-card">
          <div className="panel-title">
            <div>
              <span className="panel-kicker">ACTUAL EXECUTION</span>
              <h3>実際に構築されたモデル</h3>
              <p>リクエスト名ではなく、FastAPI内部で解決されたモデル・獲得関数・変換設定です。</p>
            </div>
            <span className={`status-chip ${details.hybrid_model ? "success" : ""}`}>
              {details.hybrid_model ? "Hybrid" : "Single model"}
            </span>
          </div>

          <div className="model-detail-grid">
            <div><span>Backend</span><strong>{display(details.optimizer_backend)}</strong></div>
            <div><span>Model class</span><strong>{display(details.model_class)}</strong></div>
            <div><span>Requested model</span><strong>{display(details.requested_model_type)}</strong></div>
            <div><span>Internal model</span><strong>{display(details.internal_model_type)}</strong></div>
            <div><span>Acquisition family</span><strong>{display(details.acquisition_family)}</strong></div>
            <div><span>Effective acquisition</span><strong>{display(details.effective_acquisition)}</strong></div>
            <div><span>Acquisition class</span><strong>{display(details.acquisition_class)}</strong></div>
            <div><span>Effective optimizer</span><strong>{display(details.effective_optimizer)}</strong></div>
            <div><span>Normalize</span><strong>{display(details.normalize)}</strong></div>
            <div><span>Input perturbation</span><strong>{display(details.input_perturbation)}</strong></div>
            <div><span>n_w / std</span><strong>{display(details.n_w)} / {display(details.perturbation_std)}</strong></div>
            <div><span>Model kwargs</span><strong>{display(details.model_kwargs)}</strong></div>
          </div>

          {outputSpecs.length > 0 && (
            <details className="model-output-details" open>
              <summary>目的変数ごとのサブモデル</summary>
              <div className="table-wrap compact">
                <table>
                  <thead><tr><th>目的変数</th><th>タスク</th><th>モデルクラス</th></tr></thead>
                  <tbody>
                    {outputSpecs.map((spec, index) => (
                      <tr key={`${String(spec.name)}-${index}`}>
                        <td>{display(spec.name)}</td>
                        <td>{display(spec.task_type)}</td>
                        <td><code>{display(spec.model_class)}</code></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </section>
      )}

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
              const entryDetail = entryDetails(entry);
              const hasDetails = Object.keys(entryDetail).length > 0;
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
                  {hasDetails && <pre>{JSON.stringify(entryDetail, null, 2)}</pre>}
                </details>
              );
            })}
          </div>
        )}
      </section>
    </>
  );
}
