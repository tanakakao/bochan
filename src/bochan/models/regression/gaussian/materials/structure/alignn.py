"""Canonical ALIGNN Gaussian-model import surface.

The implementation remains under :mod:`bochan.models.regression.gaussian.deep`
during the staged migration so existing pickle/module paths and internal
relative imports remain stable. This module exposes the canonical structure
namespace while preserving exact class identity with the historical imports.
"""

from ...deep.alignn import ALIGNNDKLModel, ALIGNNGPModel
from ...deep.alignn_mixed import ALIGNNMixedDKLModel, ALIGNNMixedGPModel
from ...deep.alignn_multitask import (
    ALIGNNMixedMultiTaskDKLModel,
    ALIGNNMixedMultiTaskGPModel,
    ALIGNNMultiTaskDKLModel,
    ALIGNNMultiTaskGPModel,
)

__all__ = [
    "ALIGNNDKLModel",
    "ALIGNNGPModel",
    "ALIGNNMixedDKLModel",
    "ALIGNNMixedGPModel",
    "ALIGNNMixedMultiTaskDKLModel",
    "ALIGNNMixedMultiTaskGPModel",
    "ALIGNNMultiTaskDKLModel",
    "ALIGNNMultiTaskGPModel",
]
