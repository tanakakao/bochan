interface WorkbenchErrorAlertProps {
  error: string | null;
  onClose: () => void;
}

export default function WorkbenchErrorAlert({ error, onClose }: WorkbenchErrorAlertProps) {
  if (!error) return null;

  return (
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
        onClick={onClose}
      >
        <span aria-hidden="true">×</span>
      </button>
    </div>
  );
}
