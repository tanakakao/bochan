import Plot from "react-plotly.js";
import type { Data } from "plotly.js";
import { useWorkbench } from "./context/WorkbenchContext";
import { RESULT_PLOT_CONFIG } from "./plotConfig";
import { themedPlotLayout } from "./plotLayout";
import type { ResultVisualization } from "./types";

interface ResultVisualizationsProps {
  visualizations: ResultVisualization[];
  warnings: string[];
}

export default function ResultVisualizations({ visualizations, warnings }: ResultVisualizationsProps) {
  const { theme } = useWorkbench();

  if (visualizations.length === 0 && warnings.length === 0) return null;

  return (
    <section className="visualization-section">
      <div className="result-subheading">
        <div>
          <span className="eyebrow">Visualization</span>
          <h3>結果の可視化</h3>
          <p>モデル予測、候補点、不確かさを同じテーマで確認します。</p>
        </div>
      </div>

      {warnings.length > 0 && (
        <div className="alert warning">
          {warnings.map((warning) => <div key={warning}>{warning}</div>)}
        </div>
      )}

      <div className="visualization-grid">
        {visualizations.map((visualization) => (
          <article className="panel visualization-card" key={visualization.id}>
            <div className="visualization-heading">
              <div>
                <span className="panel-kicker">{visualization.id}</span>
                <h3>{visualization.title}</h3>
                <p>{visualization.description}</p>
              </div>
            </div>
            <div className="plot-container">
              <Plot
                data={visualization.figure.data as Data[]}
                layout={themedPlotLayout(visualization.figure.layout, theme)}
                config={RESULT_PLOT_CONFIG}
                useResizeHandler
                style={{ width: "100%", height: "100%" }}
              />
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
