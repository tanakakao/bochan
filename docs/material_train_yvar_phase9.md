# Material train_Yvar Phase 9: observation-aware acquisition baselines

## Goal

Phase 9 extends the observation semantics established in the earlier material / `train_Yvar` phases into candidate generation.

Model fitting already keeps objective regression rows separate from failure-classification rows: observed outcomes are used for regression, failed experiments remain available to the failure model, and pending experiments are excluded from completed observations. Partial multi-output targets and known observation variance (`train_Yvar`) are kept row-aligned during model fitting.

The remaining acquisition-side gap was the automatically inferred `X_baseline` / `Y_baseline`. For a partially observed multi-output data set, the generic optimizer context can contain every regression row even when a selected scalar objective is missing on some of those rows.

## Phase 9 semantics

For an automatically generated scalar acquisition baseline:

- resolve the scalar objective output from `ObjectiveConfig.output`;
- support integer output indices and configured/model `output_names`;
- for scalarization over multiple outputs, require the outputs participating in that scalar objective to be jointly finite;
- keep only baseline rows where the selected scalar objective value is actually observed;
- preserve row alignment between `X_baseline` and `Y_baseline`;
- leave the fitted `train_X`, `train_Y`, and `train_Yvar` untouched.

This is deliberately an acquisition-context operation. Known observation variance continues to belong to model fitting and posterior construction; Phase 9 does not reinterpret, aggregate, or re-index `train_Yvar`.

## Explicit baseline precedence

User-provided baselines remain authoritative. Observation-aware filtering is applied only when `X_baseline` and `Y_baseline` are absent or correspond to the optimizer's automatic training-data defaults.

This avoids changing advanced workflows that intentionally provide a custom comparison set.

## Multi-objective behavior

Vector / multi-objective acquisitions keep the existing partial baseline.

In particular, NEHVI-style acquisitions may use the model posterior over `X_baseline`. Requiring every baseline row to have every objective observed would throw away useful partially observed experiments and would conflict with the existing partial-target model semantics.

Observed Pareto construction for EHVI remains governed by the existing NaN-safe multi-objective defaults, which require complete finite rows where an observed Pareto partition is needed.

## Failure and pending experiments

Phase 9 does not change failure-model semantics.

- failed experiments can still participate in the failure/success classifier according to the Phase 7 rules;
- pending experiments remain excluded from completed regression observations;
- success-probability / feasibility weighting of acquisition functions is unchanged.

Because regression training preparation already excludes failed and pending rows, the baseline filtering added here operates only on the regression training rows and then removes rows missing the selected scalar objective.

## Backward compatibility

Complete legacy data is unchanged: when every selected objective value is finite, the existing baseline object is returned without copying or slicing.

Datasets without a resolvable scalar output also retain their previous baseline behavior. This avoids imposing scalar semantics on vector acquisitions or custom objective objects whose output dependency cannot be inferred safely.

## Tests

Phase 9 adds coverage for:

- integer scalar-output selection with partial targets;
- named output selection;
- scalarization requiring jointly observed outputs;
- preservation of partial baselines for multi-objective acquisition;
- custom baseline precedence;
- identity-preserving behavior for complete legacy data;
- a clear error when the requested scalar objective has no observed baseline row.

## Follow-up candidates

A later phase can expose the effective acquisition baseline mask and observation counts through Web/API diagnostics. That would make it easier to explain why a candidate was generated without changing the acquisition mathematics introduced here.
