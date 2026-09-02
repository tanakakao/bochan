# Material `train_Yvar` Phase 7

Phase 7 extends observation-aware cross-validation to the experiment
success/failure classifier that can accompany physical-experiment Bayesian
optimization.

## Protocol

When no failure model is configured, the Phase-5 protocol is unchanged: folds
are built only from successful rows carrying at least one observed objective.

When `ExperimentFailureConfig` is supplied:

- the split universe is every completed experiment (`success` + `failed`);
- pending experiments are excluded;
- one row split is shared by the objective model and success classifier;
- objective training and scoring still use only successful observed target cells;
- known per-cell `Yvar` remains aligned through fold slicing;
- the classifier predicts `P(success)` and reports OOF probability metrics;
- `splitter="auto"` stratifies the shared folds by success/failure status;
- a training fold containing only one outcome is rejected explicitly;
- ROC-AUC is reported as `NaN` with a warning when a validation fold contains
  only one class, while accuracy/F1/log-loss remain available.

The classifier result is exposed separately as
`CrossValidationResult.failure_model`, so existing objective-output consumers of
`CrossValidationResult.outputs` remain backward compatible.

## Metrics

The failure-model result includes the normal binary classification metrics plus
probability-aware `roc_auc` and `log_loss`. The probability column is ordered as
`[P(failure), P(success)]`; `predictive_mean` is `P(success)`.
