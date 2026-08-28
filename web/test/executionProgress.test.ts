import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  formatElapsed,
  progressFromEntries,
  timingSummary
} from "../src/executionProgressModel.ts";

const base = {
  timestamp: "2026-08-28T07:00:00Z",
  level: "INFO",
  logger: "bochan.web.api",
  message: "progress",
  request_id: "abc123"
};

const requested = progressFromEntries([
  { ...base, event: "regression_run_requested" }
]);
assert.equal(requested?.stage, 0);
assert.equal(requested?.completedStage, -1);
assert.equal(requested?.failed, false);
assert.equal(requested?.requestId, "abc123");
assert.match(requested?.label ?? "", /データ/);

const prepared = progressFromEntries([
  { ...base, event: "workflow_started" },
  { ...base, event: "workflow_data_prepared" }
]);
assert.equal(prepared?.stage, 1);
assert.equal(prepared?.completedStage, 0);

const cvFold = progressFromEntries([
  {
    ...base,
    event: "model_fit_started",
    fit_phase: "cross_validation",
    fold_current: 3,
    fold_total: 5,
    output_total: 2,
    fit_mode: "independent"
  }
]);
assert.equal(cvFold?.stage, 1);
assert.match(cvFold?.label ?? "", /CV fold 3 \/ 5/);

const cvTarget = progressFromEntries([
  {
    ...base,
    event: "model_output_fit_started",
    fit_phase: "cross_validation",
    fold_current: 3,
    fold_total: 5,
    output_index: 2,
    output_total: 4,
    output_name: "strength"
  }
]);
assert.equal(cvTarget?.stage, 1);
assert.match(cvTarget?.label ?? "", /CV fold 3 \/ 5/);
assert.match(cvTarget?.label ?? "", /目的変数 2 \/ 4: strength/);

const loo = progressFromEntries([
  {
    ...base,
    event: "model_fit_started",
    fit_phase: "cross_validation_or_final",
    fit_mode: "single",
    output_total: 1
  }
]);
assert.match(loo?.label ?? "", /CV\/最終モデル/);
assert.doesNotMatch(loo?.label ?? "", /fold \d+ \/ \d+/);

const joint = progressFromEntries([
  {
    ...base,
    event: "model_fit_started",
    fit_phase: "final",
    fit_mode: "joint",
    output_total: 3
  }
]);
assert.match(joint?.label ?? "", /3目的を共同学習/);

const fitCompleted = progressFromEntries([
  {
    ...base,
    event: "model_fit_completed",
    fit_phase: "final",
    fit_mode: "single",
    output_total: 1
  }
]);
assert.equal(fitCompleted?.stage, 2);
assert.equal(fitCompleted?.completedStage, 1);

const reused = progressFromEntries([
  { ...base, event: "model_reuse_completed" }
]);
assert.equal(reused?.stage, 2);
assert.equal(reused?.completedStage, 1);
assert.match(reused?.label ?? "", /再利用/);

const candidateStarted = progressFromEntries([
  { ...base, event: "candidate_generation_started", q: 12 }
]);
assert.equal(candidateStarted?.stage, 2);
assert.equal(candidateStarted?.completedStage, 1);
assert.doesNotMatch(candidateStarted?.label ?? "", /q=12/);

const candidateCompleted = progressFromEntries([
  { ...base, event: "candidate_generation_completed" }
]);
assert.equal(candidateCompleted?.stage, 3);
assert.equal(candidateCompleted?.completedStage, 2);

const completed = progressFromEntries([
  { ...base, event: "regression_run_requested" },
  {
    ...base,
    timestamp: "2026-08-28T07:00:12Z",
    event: "workflow_completed",
    timings_ms: {
      prepare: 125,
      fit: 8200,
      candidate: 2400,
      prediction: 150,
      visualization: 625,
      total: 11500
    }
  },
  {
    ...base,
    timestamp: "2026-08-28T07:00:12Z",
    event: "regression_run_completed"
  }
]);
assert.equal(completed?.stage, 4);
assert.equal(completed?.completedStage, 4);
assert.equal(completed?.failed, false);
assert.equal(completed?.timingsMs?.fit, 8200);
assert.deepEqual(timingSummary(completed?.timingsMs), [
  "準備 0.1秒",
  "学習 8.2秒",
  "候補探索 2.4秒",
  "予測 0.1秒",
  "可視化 0.6秒",
  "合計 12秒"
]);

const modelFailed = progressFromEntries([
  { ...base, event: "workflow_data_prepared" },
  { ...base, event: "model_fit_started", fit_phase: "final" },
  { ...base, event: "model_fit_failed", fit_phase: "final" },
  { ...base, event: "regression_run_failed" }
]);
assert.equal(modelFailed?.stage, 1);
assert.equal(modelFailed?.completedStage, 0);
assert.equal(modelFailed?.failed, true);
assert.match(modelFailed?.label ?? "", /モデル学習に失敗/);

const candidateFailed = progressFromEntries([
  { ...base, event: "model_fit_completed", fit_phase: "final" },
  { ...base, event: "candidate_generation_started" },
  { ...base, event: "candidate_generation_failed" },
  { ...base, event: "regression_run_failed" }
]);
assert.equal(candidateFailed?.stage, 2);
assert.equal(candidateFailed?.completedStage, 1);
assert.equal(candidateFailed?.failed, true);
assert.match(candidateFailed?.label ?? "", /候補探索に失敗/);

assert.equal(formatElapsed(0), "0.0秒");
assert.equal(formatElapsed(9_250), "9.3秒");
assert.equal(formatElapsed(42_000), "42秒");
assert.equal(formatElapsed(125_000), "2分05秒");

const componentSource = readFileSync(new URL("../src/components/ExecutionProgress.tsx", import.meta.url), "utf8");
const modelSource = readFileSync(new URL("../src/executionProgressModel.ts", import.meta.url), "utf8");
assert.doesNotMatch(componentSource, /estimatedProgress|約\$\{|推定/);
assert.doesNotMatch(modelSource, /estimatedProgress/);
assert.match(componentSource, /progress\.completedStage/);
