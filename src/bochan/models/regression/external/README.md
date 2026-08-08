# External regression models

Regression wrappers around non-PyTorch external estimators live in this package.

- `ngboost.py`: NGBoost single and bootstrap-ensemble regressors
- `random_forest.py`: Random Forest regressor using fitted trees as finite ensemble members

These models share the Tensor/NumPy and categorical preprocessing boundary from `bochan.models.external.common`.
