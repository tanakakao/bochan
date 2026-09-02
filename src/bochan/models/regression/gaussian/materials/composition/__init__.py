"""Canonical composition-model namespace for material-aware Gaussian models."""

from .crabnet import (
    CrabNetDKLModel,
    CrabNetGPModel,
    CrabNetMixedDKLModel,
    CrabNetMixedGPModel,
    CrabNetMixedMultiTaskDKLModel,
    CrabNetMixedMultiTaskGPModel,
    CrabNetMultiTaskDKLModel,
    CrabNetMultiTaskGPModel,
)
from .roost import RoostDKLModel, RoostGPModel

__all__ = [
    "CrabNetDKLModel",
    "CrabNetGPModel",
    "CrabNetMixedDKLModel",
    "CrabNetMixedGPModel",
    "CrabNetMixedMultiTaskDKLModel",
    "CrabNetMixedMultiTaskGPModel",
    "CrabNetMultiTaskDKLModel",
    "CrabNetMultiTaskGPModel",
    "RoostDKLModel",
    "RoostGPModel",
]
