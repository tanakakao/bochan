# Material `train_Yvar` Phase 8

Phase 8 adds cross-validated feature importance to the observation-aware validation protocol introduced in Phases 5 and 7.

## Objective feature importance

When `CrossValidationConfig.feature_importance_config` is supplied, feature importance is evaluated with each fold-trained objective model on validation data only.

For each objective output independently:

- only successful experiments are eligible;
- only rows where that output is observed are included;
- failed, pending, and missing target cells are excluded;
- the fold split remains shared with the observation-aware CV protocol;
- known observation variance (`Yvar` / `target_variance_cols`) remains aligned in fold training data but is never treated as an input feature.

This output-local protocol is required because partially observed multi-output data can have a different validation row set for each objective.

The aggregated result is returned through the existing `CrossValidationResult.feature_importance` contract. Each `CVFoldResult.feature_importance` also contains the corresponding fold/output result.

## Failure/success feature importance

When an `ExperimentFailureConfig` is also supplied, the success classifier is inspected separately:

- validation rows are completed experiments (`success` + `failed`);
- pending experiments are excluded;
- the inspected target is `P(success)`;
- results are returned in `CrossValidationResult.failure_feature_importance` under the output name `experiment_success`.

The failure classifier and objective models continue to use the same completed-row fold partition from Phase 7, preventing validation leakage while retaining their different target masks.

## Reproducibility

Feature-importance configuration is cloned per fold. A deterministic fold-specific permutation seed is derived from `FeatureImportanceConfig.random_state`; the caller's configuration object is not mutated.

## Compatibility

- Omitting `feature_importance_config` preserves the Phase 7 behavior.
- Fully observed ordinary cross-validation continues to use the existing feature-importance contract.
- `CrossValidationResult.failure_feature_importance` is additive and defaults to `None`.
- Feature importance remains validation-fold importance, not pooled OOF permutation importance.

## Scope

Phase 8 covers predictive raw-space feature importance and the existing lightweight diagnostics exposed by `bochan.inspection`. It does not add a new explanation method or alter acquisition/candidate selection semantics.
