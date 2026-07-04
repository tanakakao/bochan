import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import type { ResultVisualization } from "./types";

interface ResultVisualizationsProps {
  visualizations: ResultVisualization[];
  warnings: string[];
}

export default function ResultVisualizations({ visualizations, warnings }: ResultVisualizationsProps) {
  if (visualizations.length === 0 && warnings.length === 0) return null;

  return (
    <section className="visualization-section">
      <div className="result-subheading">
        <div>
          <h3>結果の可視化</h3>
          <p>グラフはbochan.visualizationで生成しています。</p>
        </div>
      </div>

      {warnings.length > 0 && (
        <div className="alert warning">
          {warnings.map((warning) => <div key={warning}>{warning}</div>)}
        </div>
      )}

      <div className="visualization-grid">
        {visualizations.map((visualization) => (
          <article className="card visualization-card" key={visualization.id}>
            <div className="visualization-heading">
              <div>
                <h3>{visualization.title}</h3>
                <p>{visualization.description}</p>
              </div>
            </div>
            <div className="plot-container">
              <Plot
                data={visualization.figure.data as Data[]}
                layout={{
                  ...(visualization.figure.layout as Partial<Layout>),
                  autosize: true,
                  width: undefined
                }}
                config={{
                  responsive: true,
                  displaylogo: false,
                  modeBarButtonsToRemove: ["lasso2d", "select2d"]
                }}
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
