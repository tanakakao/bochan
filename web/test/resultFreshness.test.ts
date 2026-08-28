import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

function source(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}

const signatures = source("../src/resultSignatures.ts");
assert.match(signatures, /buildSuggestionSignature/);
assert.match(signatures, /modelSignature:\s*buildModelReuseSignature/);
for (const setting of [
  "targetSettings",
  "acquisition",
  "q:",
  "numRestarts",
  "rawSamples",
  "sequential",
  "minimumCandidateDistanceRatio",
  "searchSpace",
  "featureConstraints",
  "selectionCountConstraint",
  "compositionElementConstraints",
  "featureImportance"
]) {
  assert.ok(signatures.includes(setting), `suggestion signature should include ${setting}`);
}
assert.match(signatures, /fixedValue/);
assert.match(signatures, /step:/);
assert.match(signatures, /perturbationRisk/);

const context = source("../src/context/WorkbenchContext.tsx");
assert.match(context, /const currentModelSignature/);
assert.match(context, /const currentSuggestionSignature/);
assert.match(context, /const resultCurrent/);
assert.match(context, /currentSuggestionSignature === results\.lastSuggestionSignature/);
assert.match(context, /results\.setLastSuggestionSignature\(suggestionSignature\)/);
assert.match(context, /useWorkbenchExternalSettingsRevision\(\)/);

const resultState = source("../src/context/useWorkbenchResultState.ts");
assert.match(resultState, /lastModelSignature/);
assert.match(resultState, /lastSuggestionSignature/);
assert.match(resultState, /setLastSuggestionSignature\(null\)/);

const persistedSettings = source("../src/webRunSettings.ts");
for (const saveFunction of [
  "saveFeatureConstraints",
  "saveSelectionCountConstraint",
  "saveFeatureMissingSettings",
  "saveInputPerturbationRiskSettings",
  "saveSearchMethod",
  "saveCrossValidationSettings"
]) {
  const start = persistedSettings.indexOf(`export function ${saveFunction}`);
  assert.ok(start >= 0, `${saveFunction} should exist`);
  const nextExport = persistedSettings.indexOf("export function ", start + 1);
  const body = persistedSettings.slice(start, nextExport >= 0 ? nextExport : undefined);
  assert.match(body, /notifyWorkbenchRunSettingsChanged\(\)/, `${saveFunction} should notify the workbench`);
}

const noise = source("../src/noiseAlphaSettings.ts");
assert.match(noise, /notifyWorkbenchRunSettingsChanged\(\)/);
assert.match(noise, /typeof window === "undefined"/);

const freshnessHook = source("../src/context/useWorkbenchExternalSettingsRevision.ts");
assert.match(freshnessHook, /WORKBENCH_RUN_SETTINGS_CHANGE_EVENT/);
assert.match(freshnessHook, /setRevision/);

console.log("result freshness regression checks passed");
