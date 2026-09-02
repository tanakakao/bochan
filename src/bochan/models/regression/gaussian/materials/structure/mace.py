"""Canonical MACE Gaussian-model import surface."""

from ...deep.mace import MACEDKLModel, MACEGPModel
from ...deep.mace_mixed import MACEMixedDKLModel, MACEMixedGPModel
from ...deep.mace_multitask import (
    MACEMixedMultiTaskDKLModel,
    MACEMixedMultiTaskGPModel,
    MACEMultiTaskDKLModel,
    MACEMultiTaskGPModel,
)

__all__ = [
    "MACEDKLModel",
    "MACEGPModel",
    "MACEMixedDKLModel",
    "MACEMixedGPModel",
    "MACEMixedMultiTaskDKLModel",
    "MACEMixedMultiTaskGPModel",
    "MACEMultiTaskDKLModel",
    "MACEMultiTaskGPModel",
]
