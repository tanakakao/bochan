import { useEffect, useState, type ComponentType } from "react";
import {
  STEPS,
  WorkbenchProvider,
  type WorkbenchStep,
  useWorkbench
} from "./context/WorkbenchContext";
import ConversationPage from "./pages/ConversationPage";
import DataPage from "./pages/DataPage";
import ExperimentPage from "./pages/ExperimentPage";
import LogsPage from "./pages/LogsPage";
import OptimizePage from "./pages/OptimizePage";
import PreparePage from "./pages/PreparePage";
import ResultsPage from "./pages/ResultsPage";
import SettingsPage from "./pages/SettingsPage";
import { targetClassValues } from "./targetSettingUtils";
import TutorialGuide from "./tutorial/TutorialGuide";
import type { AcquisitionFamily, TargetSetting } from "./types";
import { setWorkbenchMode, useWorkbenchMode } from "./workbenchMode";

const PAGES: Record<WorkbenchStep, ComponentType> = {
  data: DataPage,
  prepare: PreparePage,
  settings: SettingsPage,
  optimize: OptimizePage,
  results: ResultsPage,
  logs: LogsPage
};

const ICONS: Record<WorkbenchStep, string> = {
  data: "▦",
  prepare: "◇",
  settings: "⌘",
  optimize: "↗",
  results: "◎",
  logs: "≡"
};

const API_STATUS_LABELS = {
  loading: "確認中",
  ready: "接続済み",
  error: "エラー"
} as const;
const DEFAULT_PORTAL_URL = "http://127.0.0.1:5172";

function resolvePortalUrl(): string {
  const configured = import.meta.env.VITE_PORTAL_URL?.trim();
  if (!configured) return DEFAULT_PORTAL_URL;
  try {
    return new URL(configured, window.location.href).toString();
  } catch {
    return DEFAULT_PORTAL_URL;
  }
}

type AuxiliaryPage = "conversation" | "experiment";

function currentAuxiliaryPage(): AuxiliaryPage | null {
  if (window.location.hash === "#conversation") return "conversation";
  if (window.location.hash === "#experiment") return "experiment";
  return null;
}

function clearAuxiliaryHash(): void {
  window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

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

function familyLabel(value: AcquisitionFamily): string {
  if (value === "active_learning") return "アクティブラーニング";
  if (value === "level_set_estimation") return "レベルセット推定";
  return "ベイズ最適化";
}

function summarizeTargetSetting(setting: TargetSetting): string {
  const roleText = setting.optimize
    ? setting.goal === "target"
      ? "目標最適化"
      : setting.direction === "minimize"
        ? "最小化"
        : "最大化"
    : "制約専用";
  const classText = setting.task_type === "classification"
    ? `class=${targetClassValues(setting).map(String).join("|") || "—"}`
    : setting.task_type === "ordinal" && setting.goal === "target"
      ? `target=${(setting.target_values ?? []).map(String).join("|") || "—"}`
      : "";
  const constraintText = setting.goal === "none"
    ? goalLabel(setting.goal)
    : `${goalLabel(setting.goal)} ${setting.goal === "target" ? (setting.target_values ?? []).map(String).join("|") : String(setting.value ?? "—")}`;
  return [setting.target, roleText, classText, constraintText].filter(Boolean).join(": ");
}

function WorkbenchLayout() {
  const mode = useWorkbenchMode();
  const [auxiliaryPage, setAuxiliaryPage] = useState<AuxiliaryPage | null>(currentAuxiliaryPage);
  const [tutorialRequest, setTutorialRequest] = useState(0);
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
    acquisitionFamily,
    acquisition,
    q,
    result
  } = useWorkbench();
  const workflowSteps = STEPS.filter(([id]) => id !== "logs");
  const visibleSteps = mode === "simple"
    ? workflowSteps.filter(([id]) => id === "data" || id === "prepare" || id === "results")
    : workflowSteps;
  const index = visibleSteps.findIndex(([id]) => id === step);
  const experimentAvailable = Boolean(dataset && result);
  const activeAuxiliaryPage = auxiliaryPage === "conversation"
    ? "conversation"
    : auxiliaryPage === "experiment" && experimentAvailable
      ? "experiment"
      : null;
  const Page = activeAuxiliaryPage === "conversation"
    ? ConversationPage
    : activeAuxiliaryPage === "experiment"
      ? ExperimentPage
      : PAGES[step];
  const targetSummary = selectedTargetSettings.length
    ? selectedTargetSettings.map(summarizeTargetSetting).join(" / ")
    : "—";
  const resultStale = Boolean(result?.metadata?.stale_after_data_append);
  const apiStatusLabel = API_STATUS_LABELS[health.status];
  const progressStepIndex = activeAuxiliaryPage === "experiment"
    ? visibleSteps.length - 1
    : Math.max(index, 0);
  const progressLabel = activeAuxiliaryPage === "conversation"
    ? "対話モード"
    : activeAuxiliaryPage === "experiment"
      ? "実験結果追加"
      : visibleSteps[progressStepIndex]?.[1] ?? "データ";
  const progressMeta = activeAuxiliaryPage === "conversation"
    ? "GUIDED FLOW"
    : activeAuxiliaryPage === "experiment"
      ? "EXPERIMENT"
      : `STEP ${progressStepIndex + 1} / ${visibleSteps.length}`;
  const progressPercent = visibleSteps.length
    ? Math.min(100, Math.max(0, ((progressStepIndex + 1) / visibleSteps.length) * 100))
    : 0;

  useEffect(() => {
    const handleHashChange = () => setAuxiliaryPage(currentAuxiliaryPage());
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  useEffect(() => {
    if (auxiliaryPage === "experiment" && !experimentAvailable) {
      setStep("data");
      clearAuxiliaryHash();
    }
  }, [auxiliaryPage, experimentAvailable, setStep]);

  useEffect(() => {
    if (step === "logs") {
      setStep(result ? "results" : dataset ? "prepare" : "data");
      return;
    }
    if (mode === "simple" && (step === "settings" || step === "optimize")) {
      setStep(dataset ? "prepare" : "data");
    }
  }, [dataset, mode, result, setStep, step]);

  function isComplete(id: WorkbenchStep, stepIndex: number): boolean {
    if (activeAuxiliaryPage) {
      return stepIndex <= index && canOpenStep(id);
    }
    return stepIndex < index && canOpenStep(id);
  }

  function openStep(id: WorkbenchStep) {
    if (auxiliaryPage) clearAuxiliaryHash();
    setStep(id);
  }

  function openConversation() {
    window.location.hash = "conversation";
  }

  function openExperiment() {
    if (!experimentAvailable) return;
    window.location.hash = "experiment";
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span>b</span></div>
          <div className="brand-wordmark">
            <h1>ベイズ最適化</h1>
            <p>Materials Analysis Workbench · bochan</p>
          </div>
        </div>

        <div className="workflow-progress" aria-label="現在の進捗" data-tutorial="workflow">
          <div className="workflow-progress-copy">
            <span>{progressMeta}</span>
            <strong>{progressLabel}</strong>
          </div>
          <div className="workflow-progress-track" aria-hidden="true">
            <span style={{ width: `${progressPercent}%` }} />
          </div>
        </div>

        <div className="header-actions tutorial-enabled">
          <div className="runtime-pill" title={`API接続: ${health.text}`}>
            <span className={`dot ${health.status}`} />
            <span className="runtime-copy">
              <small>API接続</small>
              <strong>{apiStatusLabel}</strong>
            </span>
          </div>
          <button
            type="button"
            className="icon-button secondary tutorial-button"
            title="チュートリアルを表示"
            aria-label="チュートリアルを表示"
            onClick={() => setTutorialRequest((current) => current + 1)}
          >
            <span aria-hidden="true">?</span>
          </button>
          <button
            type="button"
            className="portal-button secondary"
            title="ツール一覧へ戻る"
            aria-label="ツール一覧へ戻る"
            onClick={() => window.location.assign(resolvePortalUrl())}
          >
            <span aria-hidden="true">▦</span>
            <span>ツール一覧</span>
          </button>
          <button
            type="button"
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
          <button
            type="button"
            className={`conversation-launcher ${activeAuxiliaryPage === "conversation" ? "active" : ""}`}
            onClick={openConversation}
            aria-current={activeAuxiliaryPage === "conversation" ? "page" : undefined}
          >
            <span className="conversation-launcher-icon" aria-hidden="true">✦</span>
            <span className="conversation-launcher-copy">
              <strong>対話モード</strong>
              <small>質問に答えて候補を提案</small>
            </span>
            <span className="conversation-launcher-arrow" aria-hidden="true">›</span>
          </button>

          <div className="rail-section-label">Mode</div>
          <div className="workbench-mode-switch" role="group" aria-label="実行モード" data-tutorial="mode">
            <button
              type="button"
              className={mode === "simple" ? "active" : ""}
              aria-pressed={mode === "simple"}
              onClick={() => setWorkbenchMode("simple")}
            >
              簡易
            </button>
            <button
              type="button"
              className={mode === "advanced" ? "active" : ""}
              aria-pressed={mode === "advanced"}
              onClick={() => setWorkbenchMode("advanced")}
            >
              詳細
            </button>
          </div>

          <div className="rail-section-label">Workflow</div>
          <nav className="tabs" aria-label="ページナビゲーション" data-tutorial="navigation">
            {visibleSteps.map(([id, label, detail], stepIndex) => (
              <button
                key={id}
                className={`tab ${!activeAuxiliaryPage && step === id ? "active" : ""} ${isComplete(id, stepIndex) ? "complete" : ""}`}
                onClick={() => openStep(id)}
                disabled={!canOpenStep(id)}
                aria-current={!activeAuxiliaryPage && step === id ? "page" : undefined}
              >
                <span className="nav-icon">{ICONS[id]}</span>
                <span><strong>{label}</strong><small>{detail}</small></span>
                <em>{stepIndex + 1}</em>
              </button>
            ))}
            <button
              className={`tab ${activeAuxiliaryPage === "experiment" ? "active" : ""}`}
              onClick={openExperiment}
              disabled={!experimentAvailable}
              aria-current={activeAuxiliaryPage === "experiment" ? "page" : undefined}
            >
              <span className="nav-icon">＋</span>
              <span><strong>Experiment</strong><small>実験結果追加</small></span>
              <em>{visibleSteps.length + 1}</em>
            </button>
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

        <section className="content" data-tutorial="workspace">
          <div className="content-inner">
            {error && (
              <div className="inline-alert error" role="alert">
                <div className="inline-alert-icon" aria-hidden="true">!</div>
                <div className="inline-alert-copy">
                  <span className="eyebrow">ERROR</span>
                  <strong>処理を完了できませんでした</strong>
                  <p>{error}</p>
                  <small>入力内容と接続状態を確認し、必要な設定を修正して再実行してください。</small>
                </div>
                <button
                  type="button"
                  className="alert-close icon-button secondary"
                  title="エラー表示を閉じる"
                  aria-label="エラー表示を閉じる"
                  onClick={() => setError(null)}
                >
                  <span aria-hidden="true">×</span>
                </button>
              </div>
            )}
            <Page />
          </div>
        </section>

        <aside className="right-rail" data-tutorial="context">
          <div className={`side-card runtime-card ${health.status}`}>
            <div className="side-card-title">
              <span>Runtime</span>
              <strong>API接続</strong>
            </div>
            <div className="runtime-large">
              <span className={`dot ${health.status}`} />
              <div><strong>FastAPI</strong><small>{apiStatusLabel} · {health.text}</small></div>
            </div>
          </div>

          <div className="side-card">
            <div className="side-card-title">
              <span>Data context</span>
              <strong>現在のデータ</strong>
            </div>
            <div className="context-list">
              <div><span>Mode</span><strong>{activeAuxiliaryPage === "conversation" ? "対話" : mode === "simple" ? "簡易" : "詳細"}</strong></div>
              <div><span>File</span><strong title={dataset?.name || undefined}>{dataset?.name || "—"}</strong></div>
              <div><span>Rows</span><strong>{dataset?.profile.n_rows ?? "—"}</strong></div>
              <div><span>Targets</span><strong title={targetColumns.join(", ") || undefined}>{targetColumns.length ? targetColumns.join(", ") : "—"}</strong></div>
              <div><span>Features</span><strong>{featureColumns.length || "—"}</strong></div>
            </div>
          </div>

          <div className="side-card">
            <div className="side-card-title">
              <span>Optimization</span>
              <strong>探索条件</strong>
            </div>
            <div className="context-list">
              <div><span>Targets</span><strong title={targetSummary !== "—" ? targetSummary : undefined}>{targetSummary}</strong></div>
              <div><span>Model</span><strong title={modelType}>{modelType}</strong></div>
              <div><span>Family</span><strong title={familyLabel(acquisitionFamily)}>{familyLabel(acquisitionFamily)}</strong></div>
              <div><span>Acquisition</span><strong title={acquisition}>{acquisition}</strong></div>
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
                <div><span>Best observed</span><strong title={formatBestObserved(result.best_observed)}>{formatBestObserved(result.best_observed)}</strong></div>
                <div>
                  <span>Status</span>
                  <strong className={resultStale ? "warning-text" : "success-text"}>
                    {resultStale ? "再学習待ち" : "Ready"}
                  </strong>
                </div>
              </div>
            ) : (
              <p>候補生成後に、予測結果と可視化の概要をここへ表示します。</p>
            )}
          </div>
        </aside>
      </main>

      <footer className="statusbar" data-tutorial="status">
        <span><span className={`dot ${health.status}`} /> API接続 {apiStatusLabel}</span>
        <span>{activeAuxiliaryPage === "conversation" ? "Conversation mode" : mode === "simple" ? "Simple mode" : "Advanced mode"}</span>
        <span>{dataset ? `${dataset.profile.n_rows} rows` : "No data"}</span>
        <span>{result ? `${result.candidates.length} candidates${resultStale ? " · stale" : ""}` : "No result"}</span>
        <span className="privacy-status">React · FastAPI · BoTorch</span>
      </footer>

      <TutorialGuide
        requestId={tutorialRequest}
        mode={mode}
        hasDataset={Boolean(dataset)}
        hasResult={Boolean(result)}
      />

      {busy && (
        <div className="overlay" role="status" aria-live="polite" aria-busy="true">
          <div className="busy-card">
            <div className="spinner" aria-hidden="true" />
            <span className="eyebrow">PROCESSING</span>
            <h3>{busy}</h3>
            <p>処理中は画面操作を一時停止しています。完了後に結果を表示します。</p>
            <span className="busy-state">処理中</span>
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
