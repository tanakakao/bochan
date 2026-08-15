import ExecutionProgress from "../ExecutionProgress";

interface WorkbenchBusyOverlayProps {
  busy: string | null;
}

export default function WorkbenchBusyOverlay({ busy }: WorkbenchBusyOverlayProps) {
  if (!busy) return null;

  return (
    <div className="overlay" role="status" aria-live="polite" aria-busy="true">
      <div className="busy-card">
        <div className="spinner" aria-hidden="true" />
        <span className="eyebrow">PROCESSING</span>
        <h3>{busy}</h3>
        <p>処理中は画面操作を一時停止しています。完了後に結果を表示します。</p>
        <ExecutionProgress busy={busy} />
        <span className="busy-state">処理中</span>
      </div>
    </div>
  );
}
