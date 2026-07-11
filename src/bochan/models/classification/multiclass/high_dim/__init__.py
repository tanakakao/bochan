from __future__ import annotations

from typing import Any

from bochan.models.projected_input_perturbation import (
    configure_projected_model_classes,
)

from .decomposition import (
    PCAMulticlassClassificationGPModel as _PCAMulticlassClassificationGPModel,
)
from .decomposition import (
    PCAMulticlassClassificationMixedGPModel as _PCAMulticlassClassificationMixedGPModel,
)
from .decomposition import (
    REMBOMulticlassClassificationGPModel as _REMBOMulticlassClassificationGPModel,
)
from .decomposition import (
    REMBOMulticlassClassificationMixedGPModel as _REMBOMulticlassClassificationMixedGPModel,
)
from .saas import (
    SaasMulticlassClassificationGPModel,
    SaasMulticlassClassificationMixedGPModel,
)


class _ProjectedMulticlassMLLMixin:
    """Delegate projected-wrapper MLL construction to the internal model."""

    def make_mll(self, beta: float = 1.0, **kwargs: Any):
        return self.base_model.make_mll(beta=float(beta), **kwargs)


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


configure_projected_model_classes(
    [
        PCAMulticlassClassificationGPModel,
        REMBOMulticlassClassificationGPModel,
    ]
)


__all__ = [
    "SaasMulticlassClassificationGPModel",
    "SaasMulticlassClassificationMixedGPModel",
    "PCAMulticlassClassificationGPModel",
    "PCAMulticlassClassificationMixedGPModel",
    "REMBOMulticlassClassificationGPModel",
    "REMBOMulticlassClassificationMixedGPModel",
]
