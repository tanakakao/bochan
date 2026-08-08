# Classification common internals

This package contains private implementation shared by binary and multiclass model wrappers.

- `probability.py`: finite probability posterior and acquisition bridges
- `ngboost.py`: shared NGBoost fitting implementation
- `random_forest.py`: shared Random Forest fitting implementation
- `deep_ensemble.py`: shared differentiable Deep Ensemble implementation

Concrete public model classes belong under `classification.binary` or `classification.multiclass`, not in this package.
