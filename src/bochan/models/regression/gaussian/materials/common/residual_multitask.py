"""Adapters for applying scalar pretrained baselines to wide residual targets."""

from __future__ import annotations

import torch
from torch import Tensor

from .residual import DirectMaterialPredictor, predict_material_baseline


class SingleOutputBaselineAdapter(DirectMaterialPredictor):
    """Embed a scalar pretrained predictor into one column of a wide target.

    Non-selected output columns receive a deterministic zero baseline. This
    allows a correlated multitask residual GP to correct one pretrained property
    while learning the remaining properties directly from data.
    """

    def __init__(
        self,
        predictor: DirectMaterialPredictor,
        *,
        output_dim: int,
        output_index: int,
    ) -> None:
        super().__init__()
        if not isinstance(predictor, DirectMaterialPredictor):
            raise TypeError("predictor must implement DirectMaterialPredictor.")
        if predictor.output_dim != 1:
            raise ValueError("SingleOutputBaselineAdapter requires predictor.output_dim == 1.")
        if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim < 2:
            raise ValueError("output_dim must be an integer >= 2.")
        if isinstance(output_index, bool) or not isinstance(output_index, int):
            raise TypeError("output_index must be an integer.")
        resolved_index = output_index if output_index >= 0 else output_dim + output_index
        if resolved_index < 0 or resolved_index >= output_dim:
            raise ValueError("output_index is outside the configured output range.")
        self.predictor = predictor
        self._output_dim = output_dim
        self._output_index = resolved_index

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def output_index(self) -> int:
        return self._output_index

    def forward(self, X: Tensor) -> Tensor:
        scalar = predict_material_baseline(self.predictor, X)
        baseline = torch.zeros(
            (*X.shape[:-1], self.output_dim),
            device=X.device,
            dtype=X.dtype,
        )
        baseline[..., self.output_index] = scalar[..., 0]
        return baseline


__all__ = ["SingleOutputBaselineAdapter"]
