from __future__ import annotations

from typing import Any

from .decomposition import (
    PCAMulticlassClassificationGPModel as _PCAMulticlassClassificationGPModel,
    PCAMulticlassClassificationMixedGPModel as _PCAMulticlassClassificationMixedGPModel,
    REMBOMulticlassClassificationGPModel as _REMBOMulticlassClassificationGPModel,
    REMBOMulticlassClassificationMixedGPModel as _REMBOMulticlassClassificationMixedGPModel,
)
from .saas import (
    SaasMulticlassClassificationGPModel,
    SaasMulticlassClassificationMixedGPModel,
)


class _ProjectedMulticlassMLLMixin:
    """Delegate projected-wrapper MLL construction to the internal model."""

    def make_mll(self, **kwargs: Any):
        return self.base_model.make_mll(**kwargs)


class PCAMulticlassClassificationGPModel(
    _ProjectedMulticlassMLLMixin,
    _PCAMulticlassClassificationGPModel,
):
    pass


class REMBOMulticlassClassificationGPModel(
    _ProjectedMulticlassMLLMixin,
    _REMBOMulticlassClassificationGPModel,
):
    pass


class PCAMulticlassClassificationMixedGPModel(
    _ProjectedMulticlassMLLMixin,
    _PCAMulticlassClassificationMixedGPModel,
):
    pass


class REMBOMulticlassClassificationMixedGPModel(
    _ProjectedMulticlassMLLMixin,
    _REMBOMulticlassClassificationMixedGPModel,
):
    pass


__all__ = [
    "SaasMulticlassClassificationGPModel",
    "SaasMulticlassClassificationMixedGPModel",
    "PCAMulticlassClassificationGPModel",
    "PCAMulticlassClassificationMixedGPModel",
    "REMBOMulticlassClassificationGPModel",
    "REMBOMulticlassClassificationMixedGPModel",
]
