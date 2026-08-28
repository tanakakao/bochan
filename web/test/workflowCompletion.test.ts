import assert from "node:assert/strict";
import { getWorkflowCompletion, workflowStatusText } from "../src/components/workbench/workflowCompletion.ts";

const empty = getWorkflowCompletion({
  hasDataset: false,
  canConfigure: false,
  settingsValid: false,
  candidateSettingsValid: false,
  result: null
});
assert.equal(empty.data.complete, false);
assert.equal(empty.data.available, true);
assert.equal(empty.prepare.complete, false);
assert.equal(empty.results.complete, false);

const configured = getWorkflowCompletion({
  hasDataset: true,
  canConfigure: true,
  settingsValid: true,
  candidateSettingsValid: true,
  result: null
});
assert.equal(configured.data.complete, true);
assert.equal(configured.prepare.complete, true);
assert.equal(configured.settings.complete, true);
assert.equal(configured.optimize.complete, false);
assert.equal(configured.optimize.available, true);
assert.equal(configured.results.complete, false);

const completed = getWorkflowCompletion({
  hasDataset: true,
  canConfigure: true,
  settingsValid: true,
  candidateSettingsValid: true,
  result: { metadata: {} } as any
});
assert.equal(completed.optimize.complete, true);
assert.equal(completed.results.complete, true);
assert.equal(workflowStatusText(completed.results), "結果あり");

const stale = getWorkflowCompletion({
  hasDataset: true,
  canConfigure: true,
  settingsValid: true,
  candidateSettingsValid: true,
  result: { metadata: { stale_after_data_append: true } } as any
});
assert.equal(stale.optimize.complete, false);
assert.equal(stale.optimize.stale, true);
assert.equal(stale.results.complete, false);
assert.equal(stale.results.stale, true);
assert.equal(workflowStatusText(stale.results), "更新必要");

console.log("semantic workflow completion checks passed");
