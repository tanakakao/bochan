import { useWorkbench } from "../../context/WorkbenchContext";
import { API_STATUS_LABELS } from "./workbenchPresentation";

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

interface WorkbenchHeaderProps {
  progressMeta: string;
  progressLabel: string;
  progressPercent: number;
  onTutorialRequest: () => void;
}

export default function WorkbenchHeader({
  progressMeta,
  progressLabel,
  progressPercent,
  onTutorialRequest
}: WorkbenchHeaderProps) {
  const { theme, setTheme, health } = useWorkbench();
  const apiStatusLabel = API_STATUS_LABELS[health.status];

  return (
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
          onClick={onTutorialRequest}
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
  );
}
