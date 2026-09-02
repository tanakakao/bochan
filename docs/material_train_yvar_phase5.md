# Material `train_Yvar` Phase 5: observation-aware cross-validation

Phase 5 extends the observation-state workflow introduced in Phase 4 with an
objective-model cross-validation protocol for partial, pending, failed, and
known-noise observations.

## Protocol

- Folds are formed only from successful rows containing at least one observed
  objective cell.
- Failed and pending experiments are excluded from objective-model CV and are
  not treated as target observations.
- One row split is shared by all outputs, while each output is scored only on
  cells observed for that output.
- Known per-observation variance (`Yvar` / `target_variance_cols`) is sliced with
  the same fold rows and target-cell masks used for model fitting.
- Every fold must retain at least one training observation for every objective
  output. A clear error is raised otherwise.
- A fold with no validation observation for one output skips that output/fold
  metric; the other observed outputs remain valid.
- OOF predictions retain original row indices so sparse output coverage can be
  inspected directly.

## Tabular API

`cross_validation=True` now works with observation conversion such as
`target_missing_strategy="keep"`, `experiment_status_col`, and
`target_variance_cols`. Ordinary fully observed tabular CV keeps the existing
row-wise path unchanged.

## Scope boundary

This phase evaluates the objective model only. Failure/success classifiers are
not cross-validated together with the objective model. Fold-level feature
importance is also intentionally rejected for observation-aware CV until an
output-specific partial-target importance protocol is defined.
