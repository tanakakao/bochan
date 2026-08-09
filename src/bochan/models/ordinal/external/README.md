# External ordinal models

External-estimator ordinal models live under the ordinal task package.

- `random_forest.py`: cumulative Random Forest ordinal model
- `ngboost.py`: cumulative NGBoost single and bootstrap-ensemble ordinal models
- `base.py`: shared cumulative probability / PAVA / latent compatibility infrastructure

For K ordered classes, the external models train K-1 cumulative threshold classifiers and project threshold probabilities to a monotone sequence before reconstructing class probabilities.
