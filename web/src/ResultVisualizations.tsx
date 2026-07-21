import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import { useWorkbench } from "./context/WorkbenchContext";
import type { ResultVisualization } from "./types";

interface ResultVisualizationsProps {
  visualizations: ResultVisualization[];
  warnings: string[];
}

function themedLayout(
  layout: Record<string, unknown>,
  theme: "light" | "dark"
): Partial<Layout> {
  const source = layout as Partial<Layout>;
  const dark = theme === "dark";
  const text = dark ? "#dfe6f1" : "#344054";
  const muted = dark ? "#7f8ba0" : "#98a2b3";
  const grid = dark ? "rgba(255,255,255,0.08)" : "rgba(15,23,42,0.08)";

  return {
    ...source,
    autosize: true,
    width: undefined,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      ...source.font,
      color: text,
      family: 'Inter, "Segoe UI", "Yu Gothic UI", Meiryo, sans-serif'
    },
    xaxis: {
      ...source.xaxis,
      color: text,
      gridcolor: grid,
      linecolor: grid,
      zerolinecolor: grid,
      tickfont: { ...source.xaxis?.tickfont, color: muted }
    },
    yaxis: {
      ...source.yaxis,
      color: text,
      gridcolor: grid,
      linecolor: grid,
      zerolinecolor: grid,
      tickfont: { ...source.yaxis?.tickfont, color: muted }
    },
    legend: {
      ...source.legend,
      font: { ...source.legend?.font, color: text }
    },
    hoverlabel: {
      ...source.hoverlabel,
      bgcolor: dark ? "#161d29" : "#ffffff",
      bordercolor: dark ? "rgba(255,255,255,0.12)" : "rgba(15,23,42,0.12)",
      font: { ...source.hoverlabel?.font, color: text }
    }
  };
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
                layout={themedLayout(visualization.figure.layout, theme)}
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
