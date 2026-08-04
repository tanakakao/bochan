import "./types";

declare module "./types" {
  interface VisualizationOptions {
    feature_labels?: Record<string, string>;
    composition?: {
      column: string;
      elements: string[];
      fraction_features: string[];
      features: Array<{
        name: string;
        label: string;
        element: string;
        min: number;
        max: number;
        default: number;
      }>;
      default_mode: "proportional" | "balance";
      modes: Array<"proportional" | "balance">;
    };
  }
}

export {};
