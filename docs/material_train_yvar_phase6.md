# Material `train_Yvar` Phase 6

Phase 6 completes the core known-observation-variance path for correlated
material multitask GP/DKL models with partially observed target matrices.

## Supported contract

For a correlated wide target table, `train_Y` and `train_Yvar` both use shape
`[n, m]`.

- An observed target cell requires a finite, strictly positive variance.
- An unobserved target cell is represented by `NaN` in both `train_Y` and
  `train_Yvar`.
- The `NaN` pattern of `train_Yvar` must exactly match the missing pattern of
  `train_Y`.
- Every output task must have at least one observed target.
- Infinite targets or variances are rejected.

This contract is shared by the correlated MACE, CHGNet, M3GNet, ALIGNN, and
CrabNet multitask GP/DKL families, including their mixed-input variants.

## Exact-GP missing observation semantics

Training uses GPyTorch's exact marginal-likelihood observation mask so missing
target cells are removed from the likelihood event. Prediction uses BOCHAN's
correlated multitask prediction strategy, which applies the same observed-event
subset to both the predictive mean and predictive covariance.

The fixed-noise likelihood stores a finite internal placeholder at missing
variance positions because a covariance diagonal must be finite. That value is
never treated as an observation: the matching target event is removed before
conditioning. The public `task_noise` view retains `NaN` at missing cells.

## Outcome transforms

`Standardize` remains supported. Its NaN-aware statistics are computed per
output from observed targets, and known variances are scaled by the same output
variance. Missing target/variance cells remain aligned after transformation.

## High-level observation workflow

The observation-aware model builder recognizes correlated model classes that
advertise the shared partial-multitask contract. It therefore keeps the wide
matrix intact instead of splitting the outputs into independent models.

Kronecker and the current multi-fidelity correlated models still require a
complete rectangular target table and are not changed by Phase 6.

## Scope

Phase 6 completes the core Phase 1-6 `train_Yvar` series. Failure-classifier
cross-validation and partial-observation feature importance are separate
model-evaluation/diagnostics work rather than missing known-noise plumbing.
