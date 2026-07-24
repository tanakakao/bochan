import type { Layout } from "plotly.js";

/** Apply the shared Web theme and keep Plotly legends below the plotting area. */
export function themedPlotLayout(
  layout: Record<string, unknown>,
  theme: "light" | "dark"
): Partial<Layout> {
  const source = layout as Partial<Layout>;
  const dark = theme === "dark";
  const text = dark ? "#dfe6f1" : "#344054";
  const muted = dark ? "#7f8ba0" : "#98a2b3";
  const grid = dark ? "rgba(255,255,255,0.08)" : "rgba(15,23,42,0.08)";
  const sourceBottomMargin = typeof source.margin?.b === "number" ? source.margin.b : 0;
  const sourceTopMargin = typeof source.margin?.t === "number" ? source.margin.t : 30;

  return {
    ...source,
    title: undefined,
    autosize: true,
    width: undefined,
    margin: {
      ...source.margin,
      t: Math.min(sourceTopMargin, 40),
      b: Math.max(sourceBottomMargin, 130)
    },
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
      orientation: "h",
      x: 0.5,
      xanchor: "center",
      y: -0.18,
      yanchor: "top",
      traceorder: "normal",
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
