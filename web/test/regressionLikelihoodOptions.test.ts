import assert from "node:assert/strict";

import {
  regressionModelVariantFor,
  regressionModelVariantLabel,
  selectRegressionModelType
} from "../src/regressionLikelihoodOptions.ts";

assert.equal(regressionModelVariantFor("tabpfn"), "tabpfn");
assert.equal(regressionModelVariantLabel("tabpfn"), "TabPFN");
assert.equal(regressionModelVariantLabel("future_foundation_model"), "future_foundation_model");

const options = [
  { value: "base", family: "standard_gp" },
  { value: "tabpfn", family: "foundation" }
] as const;

assert.equal(
  selectRegressionModelType(options, "gaussian", "base", "foundation"),
  "tabpfn"
);
