"""Canonical CHGNet Gaussian-model import surface."""

from ...deep.chgnet import (
    CHGNetDKLModel,
    CHGNetGPModel,
    CHGNetMixedDKLModel,
    CHGNetMixedGPModel,
)
from ...deep.chgnet_multitask import (
    CHGNetMixedMultiTaskDKLModel,
    CHGNetMixedMultiTaskGPModel,
    CHGNetMultiTaskDKLModel,
    CHGNetMultiTaskGPModel,
)

__all__ = [
    "CHGNetDKLModel",
    "CHGNetGPModel",
    "CHGNetMixedDKLModel",
    "CHGNetMixedGPModel",
    "CHGNetMixedMultiTaskDKLModel",
    "CHGNetMixedMultiTaskGPModel",
    "CHGNetMultiTaskDKLModel",
    "CHGNetMultiTaskGPModel",
]
