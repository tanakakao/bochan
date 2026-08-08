# Model package layout

bochan organizes surrogate models by task first, then implementation family.

- Regression external estimators: `bochan.models.regression.external`
- Binary classification: `bochan.models.classification.binary`
- Multiclass classification: `bochan.models.classification.multiclass`
- Ordinal models: `bochan.models.ordinal`

`bochan.models.classification.common` and `bochan.models.external` contain shared private infrastructure rather than concrete public task models.

Random Forest and NGBoost regression wrappers are both placed under `regression.external`; Random Forest is not a boosting algorithm, and both wrappers share the external Tensor-to-NumPy estimator boundary.
