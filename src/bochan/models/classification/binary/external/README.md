# Binary external models

Binary classification wrappers around external estimators live in this package.

- `ngboost.py`: NGBoost single and bootstrap-ensemble binary classifiers
- `random_forest.py`: Random Forest binary classifier using tree-level probability members

Shared implementation details belong in `bochan.models.classification.common`; concrete public binary model classes belong here.
