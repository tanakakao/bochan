import type { ComponentType } from "react";
import {
  STEPS,
  WorkbenchProvider,
  type WorkbenchStep,
  useWorkbench
} from "./context/WorkbenchContext";
import DataPage from "./pages/DataPage";
import LogsPage from "./pages/LogsPage";
import OptimizePage from "./pages/OptimizePage";
import PreparePage from "./pages/PreparePage";
import ResultsPage from "./pages/ResultsPage";
import SettingsPage from "./pages/SettingsPage";
import type { TargetSetting } from "./types";

const PAGES: Record<WorkbenchStep, ComponentType> = {
  data: DataPage,
  prepare: PreparePage,
  settings: SettingsPage,
  optimize: OptimizePage,
  results: ResultsPage,
  logs: LogsPage
};

const ICONS = ["▦", "◇", "⌘", "↗", "◎", "≡"];

function formatBestObserved(value: number | Record<string, number>): string {
  if (typeof value === "number") return Number.isFinite(value) ? value.toPrecision(5) : "—";
  return Object.entries(value)
    .map(([target, observed]) => `${target}: ${Number.isFinite(observed) ? observed.toPrecision(5) : "—"}`)
    .join(" / ");
}

function goalLabel(value: string): string {
  if (value === "none") return "制約なし";
  if (value === "below") return "≤";
  if (value === "target") return "目標";
  return "≥";
}

function summarizeTargetSetting(setting: TargetSetting): string {
  const classText = setting.task_type === "classification"
    ? `class=${setting.target_class ?? (setting.target_classes ?? []).join("|") || "—"}`
    : setting.task_type === "ordinal" && setting.goal === "target"
      ? `target=${(setting.target_values ?? []).join("|") || "—"}`
      : "";
  const constraintText = setting.goal === "none"
    ? goalLabel(setting.goal)
    : `${goalLabel(setting.goal)} ${setting.goal === "target" ? (setting.target_values ?? []).join("|") : String(setting.value ?? "—")}`;
  return [setting.target, classText, constraintText].filter(Boolean).join(": ");
}

function WorkbenchLayout() {
  const {
    theme,
    setTheme,
    step,
    setStep,
    canOpenStep,
    health,
    busy,
    error,
    setError,
    dataset,
    featureColumns,
    targetColumns,
    selectedTargetSettings,
    modelType,
    acquisition,
    q,
    result
  } = useWorkbench();
  const index = STEPS.findIndex(([id]) => id === step);
  const Page = PAGES[step];
  const targetSummary = selectedTargetSettings.length
    ? selectedTargetSettings.map(summarizeTargetSetting).join(" / ")
    : "—";

  function isComplete(id: WorkbenchStep, stepIndex: number): boolean {
    return stepIndex < index && canOpenStep(id);
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span>b</span></div>
          <div className="brand-wordmark">
            <h1>bochan</h1>
            <p>Bayesian optimization workbench</p>
          </div>
        </div>

        <div className="workflow-strip" aria-label="ワークフロー">
          {STEPS.map(([id, label], stepIndex) => (
            <div className="workflow-item" key={id}>
              <button
                className={`workflow-step ${id === step ? "active" : ""} ${isComplete(id, stepIndex) ? "complete" : ""}`}
                onClick={() => setStep(id)}
                disabled={!canOpenStep(id)}
                aria-current={id === step ? "step" : undefined}
              >
                <span>{stepIndex + 1}</span>
                <strong>{label}</strong>
              </button>
              {stepIndex < STEPS.length - 1 && <i />}
            </div>
          ))}
        </div>

        <div className="header-actions">
          <div className="runtime-pill" title={health.text}>
            <span className={`dot ${health.status}`} />
            <span className="runtime-copy">
              <small>API status</small>
              <strong>{health.text}</strong>
            </span>
          </div>
          <button
            className="icon-button secondary theme-toggle"
            title={theme === "dark" ? "ライトテーマへ" : "ダークテーマへ"}
            aria-label={theme === "dark" ? "ライトテーマへ切り替える" : "ダークテーマへ切り替える"}
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            <span aria-hidden="true">{theme === "dark" ? "☀" : "☾"}</span>
          </button>
        </div>
      </header>

      <main className="app-shell">
        <aside className="left-rail">
          <div className="rail-section-label">Workflow</div>
          <nav className="tabs" aria-label="ページナビゲーション">
            {STEPS.map(([id, label, detail], stepIndex) => (
              <button
                key={id}
                className={`tab ${step === id ? "active" : ""} ${isComplete(id, stepIndex) ? "complete" : ""}`}
                onClick={() => setStep(id)}
                disabled={!canOpenStep(id)}
                aria-current={step === id ? "page" : undefined}
              >
                <span className="nav-icon">{ICONS[stepIndex]}</span>
                <span><strong>{label}</strong><small>{detail}</small></span>
                <em>{stepIndex + 1}</em>
              </button>
            ))}
          </nav>
          <div className="rail-spacer" />
          <div className="rail-note">
            <div className="shield-icon">β</div>
            <div>
              <span>Computation stack</span>
              <strong>BoTorch + FastAPI</strong>
              <p>データから次の実験候補までを一つのワークスペースで扱います。</p>
            </div>
          </div>
        </aside>

        <section className="content">
          <div className="content-inner">
            {error && (
              <button className="message error inline-message" onClick={() => setError(null)}>
                {error}
              </button>
            )}
            <Page />
          </div>
        </section>

        <aside className="right-rail">
          <div className={`side-card runtime-card ${health.status}`}>
            <div className="side-card-title">
              <span>Runtime</span>
              <strong>API接続</strong>
            </div>
            <div className="runtime-large">
              <span className={`dot ${health.status}`} />
              <div><strong>FastAPI</strong><small>{health.text}</small></div>
            </div>
          </div>

          <div className="side-card">
            <div className="side-card-title">
              <span>Data context</span>
              <strong>現在のデータ</strong>
            </div>
            <div className="context-list">
              <div><span>File</span><strong>{dataset?.name || "—"}</strong></div>
              <div><span>Rows</span><strong>{dataset?.profile.n_rows ?? "—"}</strong></div>
              <div><span>Targets</span><strong>{targetColumns.length ? targetColumns.join(", ") : "—"}</strong></div>
              <div><span>Features</span><strong>{featureColumns.length || "—"}</strong></div>
            </div>
          </div>

          <div className="side-card">
            <div className="side-card-title">
              <span>Optimization</span>
              <strong>探索条件</strong>
            </div>
            <div className="context-list">
              <div><span>Targets</span><strong>{targetSummary}</strong></div>
              <div><span>Model</span><strong>{modelType}</strong></div>
              <div><span>Acquisition</span><strong>{acquisition}</strong></div>
              <div><span>q</span><strong>{q}</strong></div>
            </div>
          </div>

          <div className="side-card tips-card">
            <div className="side-card-title">
              <span>Result</span>
              <strong>最新の候補</strong>
            </div>
            {result ? (
              <div className="context-list">
                <div><span>Candidates</span><strong>{result.candidates.length}</strong></div>
                <div><span>Best observed</span><strong>{formatBestObserved(result.best_observed)}</strong></div>
                <div><span>Status</span><strong className="success-text">Ready</strong></div>
              </div>
            ) : (
              <p>候補生成後に、予測結果と可視化の概要をここへ表示します。</p>
            )}
          </div>
        </aside>
      </main>

      <footer className="statusbar">
        <span><span className={`dot ${health.status}`} /> API {health.status}</span>
        <span>{dataset ? `${dataset.profile.n_rows} rows` : "No data"}</span>
        <span>{result ? `${result.candidates.length} candidates` : "No result"}</span>
        <span className="privacy-status">React · FastAPI · BoTorch</span>
      </footer>

      {busy && (
        <div className="overlay" role="status" aria-live="polite">
          <div className="busy-card">
            <div className="spinner" />
            <h3>{busy}</h3>
            <p>処理が完了すると自動的に次の画面へ移動します。</p>
            <div className="busy-progress"><span /></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <WorkbenchProvider>
      <WorkbenchLayout />
    </WorkbenchProvider>
  );
}
