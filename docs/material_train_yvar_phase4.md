# Material known-observation variance Phase 4

Phase 4 integrates per-observation variance with the experiment observation-state workflow introduced for missing, failed, and pending experiments.

## Contract

- `ObservationData.Yvar` is optional and has the same wide `[n, m]` shape as `Y`.
- Every observed objective cell requires a finite, strictly positive variance.
- Variance for unobserved, failed, or pending cells is canonicalized to `NaN`.
- Known-variance and unknown-variance histories cannot be mixed implicitly during append / pending resolution.
- `target_variance_cols` may be used together with `target_missing_strategy="keep"` and `experiment_status_col` for DataFrame workflows.
- Variance columns are metadata/targets, never model input features.

## Partial multi-output behavior

Independent multi-output models slice `Yvar` with the exact per-output observation mask. `WideMultiTaskGP` converts wide variance to the same long task-feature rows used for observed targets, enabling correlated multitask regression with partial targets and known noise.

## Optimizer lifecycle

The public optimizer preserves Phase 3's direct fully-observed known-noise path. When explicit observation state, partial targets, pending/failed rows, or a failure model is present, it uses `ObservationData` and carries variance through fit/refit/tell/pending resolution.

Cross-validation for observation-aware status workflows remains intentionally unsupported because it requires a status-aware validation protocol rather than ordinary row splitting.
