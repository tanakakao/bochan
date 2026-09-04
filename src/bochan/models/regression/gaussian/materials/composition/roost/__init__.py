"""Canonical Roost Gaussian model imports."""

from bochan.models.regression.gaussian.deep.roost import RoostDKLModel, RoostGPModel
from bochan.models.regression.gaussian.deep.roost_extended import (
    RoostMixedDKLModel,
    RoostMixedGPModel,
    RoostMixedMultiTaskDKLModel,
    RoostMixedMultiTaskGPModel,
    RoostMultiTaskDKLModel,
    RoostMultiTaskGPModel,
)

__all__ = [
    "RoostDKLModel",
    "RoostGPModel",
    "RoostMixedDKLModel",
    "RoostMixedGPModel",
    "RoostMixedMultiTaskDKLModel",
    "RoostMixedMultiTaskGPModel",
    "RoostMultiTaskDKLModel",
    "RoostMultiTaskGPModel",
]
