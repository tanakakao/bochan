from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from botorch.models import SingleTaskGP
from torch import Tensor

from bochan.models.regression.gaussian.materials.common import (
    DirectMaterialPredictor,
    ResidualMaterialGPModel,
)


class ToyStructureBaseline(DirectMaterialPredictor):
    """Deterministic structure-index baseline used by production contract tests."""

    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.offset = float(offset)

    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, X: Tensor) -> Tensor:
        return X[..., :1] * 0.05 + self.offset


class ToyStructureResidualGP(ResidualMaterialGPModel):
    """Exact residual GP with the same constructor surface as material families."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        outcome_transform: Any = None,
        baseline_offset: float = 0.0,
        **kwargs: Any,
    ) -> None:
        del structures, kwargs
        predictor = ToyStructureBaseline(baseline_offset)
        residual_Y = train_Y - predictor(train_X)
        residual_model = SingleTaskGP(
            train_X,
            residual_Y,
            train_Yvar=train_Yvar,
            outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model)


class ToyStructureGP(SingleTaskGP):
    """Ordinary exact GP accepting structure-family constructor kwargs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        outcome_transform: Any = None,
        **kwargs: Any,
    ) -> None:
        del structures, kwargs
        super().__init__(
            train_X,
            train_Y,
            train_Yvar=train_Yvar,
            outcome_transform=outcome_transform,
        )


def resolve_toy_material_model(class_name: str) -> type:
    """Resolve residual vs ordinary material classes without loading heavy backends."""

    return ToyStructureResidualGP if "Residual" in class_name else ToyStructureGP


__all__ = [
    "ToyStructureBaseline",
    "ToyStructureGP",
    "ToyStructureResidualGP",
    "resolve_toy_material_model",
]
