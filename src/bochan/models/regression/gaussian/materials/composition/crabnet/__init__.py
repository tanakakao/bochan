"""Canonical CrabNet Gaussian model imports."""

from bochan.models.regression.gaussian.deep.crabnet import CrabNetDKLModel, CrabNetGPModel
from bochan.models.regression.gaussian.deep.crabnet_mixed import CrabNetMixedGPModel
from bochan.models.regression.gaussian.deep.crabnet_mixed_dkl import CrabNetMixedDKLModel
from bochan.models.regression.gaussian.deep.crabnet_multitask import (
    CrabNetMixedMultiTaskDKLModel,
    CrabNetMixedMultiTaskGPModel,
    CrabNetMultiTaskDKLModel,
    CrabNetMultiTaskGPModel,
)

__all__ = [
    "CrabNetDKLModel",
    "CrabNetGPModel",
    "CrabNetMixedDKLModel",
    "CrabNetMixedGPModel",
    "CrabNetMixedMultiTaskDKLModel",
    "CrabNetMixedMultiTaskGPModel",
    "CrabNetMultiTaskDKLModel",
    "CrabNetMultiTaskGPModel",
]
