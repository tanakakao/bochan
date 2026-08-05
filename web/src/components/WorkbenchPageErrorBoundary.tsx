import React, { type ErrorInfo, type ReactNode } from "react";

interface Props {
  resetKey: string;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/** Keeps the workbench shell visible when one workflow page fails during rendering. */
export default class WorkbenchPageErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Workbench page rendering failed", error, info);
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
      <article className="panel compact-panel validation-panel page-render-error" role="alert">
        <div className="panel-title">
          <div>
            <span className="panel-kicker">PAGE ERROR</span>
            <h3>画面を表示できませんでした</h3>
            <p>ページ描画中の例外を検出しました。別ページへ移動すると表示状態をリセットします。</p>
          </div>
          <span className="status-chip warning">Render failed</span>
        </div>
        <pre>{error.message || String(error)}</pre>
      </article>
    );
  }
}
