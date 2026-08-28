import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { createWorkbenchPageResetKeys } from "../src/components/workbench/workbenchPagePersistence.ts";

const initial = createWorkbenchPageResetKeys({
  datasetRevision: 1,
  datasetId: "dataset-a",
  resultRevision: 2
});
const sameWorkspace = createWorkbenchPageResetKeys({
  datasetRevision: 1,
  datasetId: "dataset-a",
  resultRevision: 2
});
assert.deepEqual(sameWorkspace, initial);

const nextDataset = createWorkbenchPageResetKeys({
  datasetRevision: 2,
  datasetId: "dataset-b",
  resultRevision: 2
});
for (const page of ["data", "prepare", "settings", "optimize", "conversation"] as const) {
  assert.notEqual(nextDataset[page], initial[page], `${page} should reset for a new dataset`);
}
assert.equal(nextDataset.results, initial.results);
assert.equal(nextDataset.experiment, initial.experiment);
assert.equal(nextDataset.logs, initial.logs);

const nextResult = createWorkbenchPageResetKeys({
  datasetRevision: 2,
  datasetId: "dataset-b",
  resultRevision: 3
});
assert.notEqual(nextResult.results, nextDataset.results);
assert.notEqual(nextResult.experiment, nextDataset.experiment);
assert.equal(nextResult.prepare, nextDataset.prepare);
assert.equal(nextResult.settings, nextDataset.settings);
assert.equal(nextResult.optimize, nextDataset.optimize);

const appSource = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
assert.match(appSource, /<PersistentWorkbenchPages/);
assert.doesNotMatch(appSource, /const Page = shell\.Page/);
assert.doesNotMatch(appSource, /<Page\s*\/>/);

const wrapperSource = readFileSync(
  new URL("../src/components/workbench/PersistentWorkbenchPage.tsx", import.meta.url),
  "utf8"
);
assert.match(wrapperSource, /hidden=\{!active\}/);
assert.match(wrapperSource, /requestAnimationFrame/);
assert.match(wrapperSource, /<Page key=\{renderKey\}/);

const pagesSource = readFileSync(
  new URL("../src/components/workbench/PersistentWorkbenchPages.tsx", import.meta.url),
  "utf8"
);
assert.match(pagesSource, /datasetRevision/);
assert.match(pagesSource, /resultRevision/);
assert.match(pagesSource, /AUXILIARY_PAGE_IDS/);
