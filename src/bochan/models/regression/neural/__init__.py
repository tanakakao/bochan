"""Neural-network surrogate models for regression."""

from .deep_ensemble import DeepEnsembleMixedRegressorModel, DeepEnsembleRegressorModel

__all__ = [
    "DeepEnsembleMixedRegressorModel",
    "DeepEnsembleRegressorModel",
]
