# Multiclass external models

Multiclass classification wrappers around external estimators live in this package.

- `ngboost.py`: NGBoost single and bootstrap-ensemble multiclass classifiers
- `random_forest.py`: Random Forest multiclass classifier using tree-level probability members

Shared implementation details belong in `bochan.models.classification.common`; concrete public multiclass model classes belong here.
