import assert from "node:assert/strict";

import {
  visualizationDefaultsKey,
  withFirstCandidateVisualizationDefaults
} from "../src/resultVisualizationDefaults.ts";
import type { RegressionResult } from "../src/types.ts";

function makeResult(): RegressionResult {
  return {
    dataset_id: "dataset-1",
    dataset_name: "sample.csv",
    task_type: "regression",
    model_type: "single_task_gp",
    n_train: 10,
    n_features: 4,
    feature_columns: ["temperature", "catalyst", "formula", "formula__fraction__Fe"],
    target_columns: ["property"],
    target_column: "property",
    directions: { property: "maximize" },
    direction: "maximize",
    best_observed: 1,
    candidates: [
      {
        rank: 2,
        values: {
          temperature: 850,
          catalyst: "A",
          formula: "Fe0.5Co0.5",
          formula__fraction__Fe: 0.5
        },
        acq_value: null,
        predictions: {},
        predicted_target_mean: 0,
        predicted_target_std: 0,
        constraints_ok: true
      },
      {
        rank: 1,
        values: {
          temperature: 1200,
          catalyst: "B",
          formula: "Fe0.7Co0.3",
          formula__fraction__Fe: 0.7
        },
        acq_value: null,
        predictions: {},
        predicted_target_mean: 0,
        predicted_target_std: 0,
        constraints_ok: true
      }
    ],
    visualizations: [],
    visualization_warnings: [],
    visualization_run_id: "run-1",
    visualization_options: {
      feature_columns: ["temperature", "catalyst", "formula__fraction__Fe"],
      numeric_features: ["temperature", "formula__fraction__Fe"],
      target_columns: ["property"],
      regression_targets: ["property"],
      ternary_groups: [],
      feature_controls: {
        temperature: { kind: "numeric", min: 800, max: 1000, default: 900 },
        catalyst: { kind: "categorical", values: ["A", "B"], default: "A" },
        formula__fraction__Fe: { kind: "numeric", min: 0, max: 1, default: 0.5 },
        missing_from_candidate: { kind: "numeric", min: 0, max: 10, default: 5 }
      }
    },
    metadata: {}
  };
}

const source = makeResult();
const updated = withFirstCandidateVisualizationDefaults(source);
const controls = updated.visualization_options?.feature_controls;

assert.equal(controls?.temperature.default, 1200);
assert.equal(controls?.temperature.min, 800);
assert.equal(controls?.temperature.max, 1200);
assert.equal(controls?.catalyst.default, "B");
assert.equal(controls?.formula__fraction__Fe.default, 0.7);
assert.equal(controls?.missing_from_candidate.default, 5);
assert.equal(source.visualization_options?.feature_controls?.temperature.default, 900);

const invalidCategory = makeResult();
invalidCategory.candidates[1].values.catalyst = "C";
const categoryFallback = withFirstCandidateVisualizationDefaults(invalidCategory);
assert.equal(categoryFallback.visualization_options?.feature_controls?.catalyst.default, "A");

const empty = makeResult();
empty.candidates = [];
assert.strictEqual(withFirstCandidateVisualizationDefaults(empty), empty);

const key1 = visualizationDefaultsKey(source);
const changed = makeResult();
changed.candidates[1].values.temperature = 1190;
const key2 = visualizationDefaultsKey(changed);
assert.notEqual(key1, key2);

console.log("result visualization default checks passed");
