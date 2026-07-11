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

const PAGES: Record<WorkbenchStep, ComponentType> = {
  data: DataPage,
  prepare: PreparePage,
  optimize: OptimizePage,
  results: ResultsPage,
  logs: LogsPage
};

const ICONS = ["▦", "◇", "↗", "◎", "▧"];

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
    targetColumn,
    direction,
    modelType,
    acquisition,
    q,
    result
  } = useWorkbench();
  const index = STEPS.findIndex(([id]) => id === step);
  const Page = PAGES[step];

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark"><span>b</span></div>
          <div className="brand-wordmark">
            <h1>ベイズサイテキカ</h1>
            <p>BOCHAN BAYESIAN OPTIMIZATION WORKBENCH</p>
          </div>
        </div>

        <div className="workflow-strip">
          {STEPS.map(([id, label], stepIndex) => (
            <div className="workflow-item" key={id}>
              <button
                className={`workflow-step ${id === step ? "active" : ""} ${stepIndex < index ? "complete" : ""}`}
                onClick={() => setStep(id)}
                disabled={!canOpenStep(id)}
              >
                <span>{stepIndex + 1}</span>
                <strong>{label}</strong>
              </button>
              {stepIndex < STEPS.length - 1 && <i />}
            </div>
          ))}
        </div>

        <div className="header-actions">
          <div className="runtime-pill">
            <span className={`dot ${health.status}`} />
            <span>{health.text}</span>
          </div>
          <button
            className="icon-button secondary"
            title="テーマ切替"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            ◐
          </button>
        </div>
      </header>

      <main className="app-shell">
        <aside className="left-rail">
          <div className="rail-section-label">WORKSPACE</div>
          <nav className="tabs">
            {STEPS.map(([id, label, detail], stepIndex) => (
              <button
                key={id}
                className={`tab ${step === id ? "active" : ""} ${stepIndex < index ? "complete" : ""}`}
                onClick={() => setStep(id)}
                disabled={!canOpenStep(id)}
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
              <strong>BoTorch + FastAPI</strong>
              <p>データ確認はReact、モデル学習と候補探索はbochan APIで実行します。</p>
            </div>
          </div>
        </aside>

        <section className="content">
          {error && (
            <button className="message error inline-message" onClick={() => setError(null)}>
              {error}
            </button>
          )}
          <Page />
        </section>

        <aside className="right-rail">
          <div className={`side-card runtime-card ${health.status}`}>
            <div className="side-card-title">
              <span>RUNTIME</span>
              <strong>API接続</strong>
            </div>
            <div className="runtime-large">
              <span className={`dot ${health.status}`} />
              <div><strong>FastAPI</strong><small>{health.text}</small></div>
            </div>
          </div>

          <div className="side-card">
            <div className="side-card-title">
              <span>DATA CONTEXT</span>
              <strong>現在のデータ</strong>
            </div>
            <div className="context-list">
              <div><span>File</span><strong>{dataset?.name || "—"}</strong></div>
              <div><span>Rows</span><strong>{dataset?.profile.n_rows ?? "—"}</strong></div>
              <div><span>Target</span><strong>{targetColumn || "—"}</strong></div>
              <div><span>Features</span><strong>{featureColumns.length || "—"}</strong></div>
            </div>
          </div>

          <div className="side-card">
            <div className="side-card-title">
              <span>OPTIMIZATION</span>
              <strong>探索条件</strong>
            </div>
            <div className="context-list">
              <div><span>Direction</span><strong>{direction}</strong></div>
              <div><span>Model</span><strong>{modelType}</strong></div>
              <div><span>Acquisition</span><strong>{acquisition}</strong></div>
              <div><span>q</span><strong>{q}</strong></div>
            </div>
          </div>

          <div className="side-card tips-card">
            <div className="side-card-title">
              <span>RESULT</span>
              <strong>最新の候補</strong>
            </div>
            {result ? (
              <div className="context-list">
                <div><span>Candidates</span><strong>{result.candidates.length}</strong></div>
                <div><span>Best observed</span><strong>{result.best_observed}</strong></div>
                <div><span>Status</span><strong className="success-text">Ready</strong></div>
              </div>
            ) : (
              <p>候補生成後に、予測結果と可視化へのショートコンテキストを表示します。</p>
            )}
          </div>
        </aside>
      </main>

      <footer className="statusbar">
        <span><span className={`dot ${health.status}`} /> API {health.status}</span>
        <span>{dataset ? `${dataset.profile.n_rows} rows` : "No data"}</span>
        <span>{result ? `${result.candidates.length} candidates` : "No result"}</span>
        <span className="privacy-status">React + FastAPI + BoTorch</span>
      </footer>

      {busy && (
        <div className="overlay">
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
