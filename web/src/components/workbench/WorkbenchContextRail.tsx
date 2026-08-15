import { useWorkbench } from "../../context/WorkbenchContext";
import type { WorkbenchMode } from "../../workbenchMode";
import type { AuxiliaryPage } from "./workbenchPages";
import {
  API_STATUS_LABELS,
  familyLabel,
  formatBestObserved,
  summarizeTargetSetting
} from "./workbenchPresentation";

interface WorkbenchContextRailProps {
  collapsed: boolean;
  mode: WorkbenchMode;
  activeAuxiliaryPage: AuxiliaryPage | null;
  onToggle: () => void;
}

export default function WorkbenchContextRail({
  collapsed,
  mode,
  activeAuxiliaryPage,
  onToggle
}: WorkbenchContextRailProps) {
  const {
    health,
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
  const apiStatusLabel = API_STATUS_LABELS[health.status];
  const targetSummary = selectedTargetSettings.length
    ? selectedTargetSettings.map(summarizeTargetSetting).join(" / ")
    : "—";
  const resultStale = Boolean(result?.metadata?.stale_after_data_append);

  return (
    <aside
      className={`right-rail ${collapsed ? "context-collapsed" : ""}`}
      data-tutorial="context"
    >
      <button
        type="button"
        className="context-rail-toggle secondary"
        aria-expanded={!collapsed}
        aria-label={collapsed ? "右サイドバーを開く" : "右サイドバーを折り畳む"}
        title={collapsed ? "設定サマリーを開く" : "設定サマリーを折り畳む"}
        onClick={onToggle}
      >
        {collapsed ? "‹" : "›"}
      </button>

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
  );
}
