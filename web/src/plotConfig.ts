import type { Config } from "plotly.js";

/** Shared Plotly interaction settings for Web result figures. */
export const RESULT_PLOT_CONFIG: Partial<Config> = {
  responsive: true,
  displaylogo: false,
  displayModeBar: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d"]
};
