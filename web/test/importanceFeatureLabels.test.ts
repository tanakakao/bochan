import assert from "node:assert/strict";

import { withImportanceFeatureLabels } from "../src/importanceFeatureLabels.ts";

const elements = Array.from({ length: 13 }, (_, index) => `E${index + 1}`);
const result = {
  feature_columns: ["formula", "temperature"],
  target_column: "property",
  target_columns: ["property"],
  metadata: {
    tabular_feature_names: [
      "formula__ilr__1",
      "formula__ilr__4",
      "formula__ilr__12",
      "temperature"
    ]
  },
  visualization_options: {
    feature_columns: ["temperature"],
    numeric_features: ["temperature"],
    target_columns: ["property"],
    regression_targets: ["property"],
    ternary_groups: [],
    composition: {
      column: "formula",
      elements,
      fraction_features: elements.map((element) => `formula__fraction__${element}`),
      features: elements.map((element) => ({
        name: `formula__fraction__${element}`,
        label: `${element} 比率`,
        element,
        min: 0,
        max: 1,
        default: 1 / elements.length
      })),
      default_mode: "proportional",
      modes: ["proportional", "balance"]
    }
  },
  visualizations: []
};

const relabeled = withImportanceFeatureLabels(result as never);

assert.equal(relabeled.feature_columns[0], "組成 ILR 1: GM(E1) / E2");
assert.equal(
  relabeled.feature_columns[1],
  "組成 ILR 4: GM(E1・E2・E3・E4) / E5"
);
assert.equal(
  relabeled.feature_columns[2],
  `組成 ILR 12: GM(${elements.slice(0, 12).join("・")}) / E13`
);
assert.equal(relabeled.feature_columns[3], "temperature");
