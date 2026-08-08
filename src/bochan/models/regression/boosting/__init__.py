"""Tree-based probabilistic boosting surrogate models."""

from .ngboost import NGBoostEnsembleModel, NGBoostRegressorModel

__all__ = [
    "NGBoostEnsembleModel",
    "NGBoostRegressorModel",
]
