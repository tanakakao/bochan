import React, { type ErrorInfo, type ReactNode } from "react";

interface Props {
  resetKey: string;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/** Prevents a render exception from leaving the workbench as an empty page. */
export default class WorkbenchPageErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Workbench rendering failed", error, info);
  }

  componentDidUpdate(previousProps: Props): void {
    if (previousProps.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <main className="content">
        <div className="content-inner">
          <article className="panel compact-panel validation-panel page-render-error" role="alert">
            <div className="panel-title">
              <div>
                <span className="panel-kicker">PAGE ERROR</span>
                <h3>画面を表示できませんでした</h3>
                <p>ページ描画中の例外を検出しました。表示を再読み込みして復旧してください。</p>
              </div>
              <span className="status-chip warning">Render failed</span>
            </div>
            <pre>{error.message || String(error)}</pre>
            <div className="button-row">
              <button type="button" onClick={() => window.location.reload()}>
                画面を再読み込み
              </button>
            </div>
          </article>
        </div>
      </main>
    );
  }
}
