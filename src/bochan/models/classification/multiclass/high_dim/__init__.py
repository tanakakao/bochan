from __future__ import annotations

from typing import Any

from torch import Tensor

from bochan.models.components.projected_utils import (
    flatten_projected_one_to_many_point_axes,
)
from .decomposition import (
    PCAMulticlassClassificationGPModel as _PCAMulticlassClassificationGPModel,
    PCAMulticlassClassificationMixedGPModel as _PCAMulticlassClassificationMixedGPModel,
    REMBOMulticlassClassificationGPModel as _REMBOMulticlassClassificationGPModel,
    REMBOMulticlassClassificationMixedGPModel as _REMBOMulticlassClassificationMixedGPModel,
)


class _ProjectedMulticlassModelMixin:
    def transform_inputs(self, X: Tensor) -> Tensor:
        transformed = super().transform_inputs(X)
        return flatten_projected_one_to_many_point_axes(X, transformed)

    def make_mll(self, beta: float = 1.0, **kwargs: Any):
        return self.base_model.make_mll(beta=float(beta), **kwargs)


class PCAMulticlassClassificationGPModel(
    _ProjectedMulticlassModelMixin,
    _PCAMulticlassClassificationGPModel,
):
    pass


class REMBOMulticlassClassificationGPModel(
    _ProjectedMulticlassModelMixin,
    _REMBOMulticlassClassificationGPModel,
):
    pass


class PCAMulticlassClassificationMixedGPModel(
    _ProjectedMulticlassModelMixin,
    _PCAMulticlassClassificationMixedGPModel,
):
    pass


class REMBOMulticlassClassificationMixedGPModel(
    _ProjectedMulticlassModelMixin,
    _REMBOMulticlassClassificationMixedGPModel,
):
    pass


__all__ = [
    "PCAMulticlassClassificationGPModel",
    "REMBOMulticlassClassificationGPModel",
    "PCAMulticlassClassificationMixedGPModel",
    "REMBOMulticlassClassificationMixedGPModel",
]
