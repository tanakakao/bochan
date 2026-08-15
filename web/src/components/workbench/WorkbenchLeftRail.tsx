import type { WorkbenchStep } from "../../context/WorkbenchContext";
import { setWorkbenchMode, type WorkbenchMode } from "../../workbenchMode";
import { WORKBENCH_ICONS, type AuxiliaryPage } from "./workbenchPages";

interface WorkbenchLeftRailProps {
  mode: WorkbenchMode;
  step: WorkbenchStep;
  visibleSteps: Array<[WorkbenchStep, string, string]>;
  activeAuxiliaryPage: AuxiliaryPage | null;
  experimentAvailable: boolean;
  canOpenStep: (step: WorkbenchStep) => boolean;
  isComplete: (step: WorkbenchStep, index: number) => boolean;
  onOpenStep: (step: WorkbenchStep) => void;
  onOpenConversation: () => void;
  onOpenExperiment: () => void;
}

export default function WorkbenchLeftRail({
  mode,
  step,
  visibleSteps,
  activeAuxiliaryPage,
  experimentAvailable,
  canOpenStep,
  isComplete,
  onOpenStep,
  onOpenConversation,
  onOpenExperiment
}: WorkbenchLeftRailProps) {
  return (
    <aside className="left-rail">
      <button
        type="button"
        className={`conversation-launcher ${activeAuxiliaryPage === "conversation" ? "active" : ""}`}
        onClick={onOpenConversation}
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
            onClick={() => onOpenStep(id)}
            disabled={!canOpenStep(id)}
            aria-current={!activeAuxiliaryPage && step === id ? "page" : undefined}
          >
            <span className="nav-icon">{WORKBENCH_ICONS[id]}</span>
            <span><strong>{label}</strong><small>{detail}</small></span>
            <em>{stepIndex + 1}</em>
          </button>
        ))}
        <button
          className={`tab ${activeAuxiliaryPage === "experiment" ? "active" : ""}`}
          onClick={onOpenExperiment}
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
  );
}
