import assert from "node:assert/strict";
import {
  formatElapsed,
  progressFromEntries,
  timingSummary
} from "../src/executionProgressModel.ts";

const requested = progressFromEntries([
  {
    timestamp: "2026-08-28T07:00:00Z",
    level: "INFO",
    logger: "bochan.web.api",
    message: "Regression workflow requested",
    event: "regression_run_requested",
    request_id: "abc123"
  }
]);
assert.equal(requested?.stage, 1);
assert.equal(requested?.failed, false);
assert.equal(requested?.requestId, "abc123");
assert.match(requested?.label ?? "", /バックエンド/);

const completed = progressFromEntries([
  {
    timestamp: "2026-08-28T07:00:00Z",
    level: "INFO",
    logger: "bochan.web.api",
    message: "Regression workflow requested",
    event: "regression_run_requested",
    request_id: "abc123"
  },
  {
    timestamp: "2026-08-28T07:00:12Z",
    level: "INFO",
    logger: "bochan.web.workflow",
    message: "Tabular target workflow completed",
    event: "workflow_completed",
    request_id: "abc123",
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
    timestamp: "2026-08-28T07:00:12Z",
    level: "INFO",
    logger: "bochan.web.api",
    message: "Regression workflow completed",
    event: "regression_run_completed",
    request_id: "abc123"
  }
]);
assert.equal(completed?.stage, 2);
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

const failed = progressFromEntries([
  {
    timestamp: "2026-08-28T07:00:00Z",
    level: "INFO",
    logger: "bochan.web.api",
    message: "Regression workflow requested",
    event: "regression_run_requested",
    request_id: "failed123"
  },
  {
    timestamp: "2026-08-28T07:00:03Z",
    level: "ERROR",
    logger: "bochan.web.api",
    message: "Regression workflow failed",
    event: "regression_run_failed",
    request_id: "failed123"
  }
]);
assert.equal(failed?.failed, true);
assert.match(failed?.label ?? "", /失敗/);

assert.equal(formatElapsed(0), "0.0秒");
assert.equal(formatElapsed(9_250), "9.3秒");
assert.equal(formatElapsed(42_000), "42秒");
assert.equal(formatElapsed(125_000), "2分05秒");
