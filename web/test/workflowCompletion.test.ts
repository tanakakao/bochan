import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { getWorkflowCompletion, workflowStatusText } from "../src/components/workbench/workflowCompletion.ts";

const empty = getWorkflowCompletion({
  hasDataset: false,
  canConfigure: false,
  settingsValid: false,
  candidateSettingsValid: false,
  result: null,
  resultCurrent: false
});
assert.equal(empty.data.complete, false);
assert.equal(empty.prepare.complete, false);
assert.equal(empty.settings.complete, false);
assert.equal(empty.optimize.complete, false);
assert.equal(empty.results.complete, false);
assert.equal(empty.logs.optional, true);

const selected = getWorkflowCompletion({
  hasDataset: true,
  canConfigure: true,
  settingsValid: false,
  candidateSettingsValid: false,
  result: null,
  resultCurrent: false
});
assert.equal(selected.data.complete, true);
assert.equal(selected.prepare.complete, true);
assert.equal(selected.settings.complete, false);
assert.equal(workflowStatusText(selected.prepare), "選択済み");

const configured = getWorkflowCompletion({
  hasDataset: true,
  canConfigure: true,
  settingsValid: true,
  candidateSettingsValid: true,
  result: null,
  resultCurrent: false
});
assert.equal(configured.settings.complete, true);
assert.equal(configured.optimize.complete, false);
assert.equal(configured.optimize.available, true);

const currentResult = getWorkflowCompletion({
  hasDataset: true,
  canConfigure: true,
  settingsValid: true,
  candidateSettingsValid: true,
  result: { metadata: {} } as any,
  resultCurrent: true
});
assert.equal(currentResult.optimize.complete, true);
assert.equal(currentResult.results.complete, true);

const changedSettingsResult = getWorkflowCompletion({
  hasDataset: true,
  canConfigure: true,
  settingsValid: true,
  candidateSettingsValid: true,
  result: { metadata: {} } as any,
  resultCurrent: false
});
assert.equal(changedSettingsResult.optimize.complete, false);
assert.equal(changedSettingsResult.results.complete, false);
assert.equal(changedSettingsResult.results.available, true);
assert.equal(workflowStatusText(changedSettingsResult.optimize), "更新必要");
assert.equal(workflowStatusText(changedSettingsResult.results), "旧結果・更新必要");

const staleResult = getWorkflowCompletion({
  hasDataset: true,
  canConfigure: true,
  settingsValid: true,
  candidateSettingsValid: true,
  result: { metadata: { stale_after_data_append: true } } as any,
  resultCurrent: true
});
assert.equal(staleResult.optimize.complete, false);
assert.equal(staleResult.results.complete, false);
assert.equal(staleResult.results.available, true);
assert.equal(workflowStatusText(staleResult.results), "旧結果・更新必要");

const here = path.dirname(fileURLToPath(import.meta.url));
const shell = fs.readFileSync(
  path.join(here, "..", "src", "components", "workbench", "useWorkbenchShell.ts"),
  "utf8"
);
const rail = fs.readFileSync(
  path.join(here, "..", "src", "components", "workbench", "WorkbenchLeftRail.tsx"),
  "utf8"
);
assert.doesNotMatch(shell, /stepIndex\s*<\s*index/);
assert.match(shell, /completedStepCount/);
assert.match(shell, /getWorkflowCompletion/);
assert.match(shell, /resultCurrent/);
assert.match(rail, /complete \? "✓"/);
assert.match(rail, /complete \? "完了"/);

console.log("workflow completion regression checks passed");
