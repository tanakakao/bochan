"""Hybrid wrapper retaining the original partially observed wide training table."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from .multi_output import HybridMultiOutputModel
from .specs import OutputSpec


class PartiallyObservedHybridMultiOutputModel(HybridMultiOutputModel):
    """Hybrid model whose submodels may use different observed training rows.

    Each submodel is fitted only on rows where its output is observed.  The
    wrapper nevertheless retains the original wide ``train_X`` / ``train_Y`` so
    high-level acquisition defaults and diagnostics can reason about target-level
    missingness without attempting to concatenate unequal submodel datasets.
    """

    def __init__(
        self,
        specs: Sequence[OutputSpec],
        *,
        train_X_wide: Tensor,
        train_Y_wide: Tensor,
        observed_mask_wide: Tensor,
        **kwargs: Any,
    ) -> None:
        super().__init__(specs=specs, **kwargs)
        self.train_X_wide = train_X_wide
        self.train_Y_wide = train_Y_wide
        self.observed_mask_wide = observed_mask_wide

    @property
    def raw_train_X(self) -> Tensor:
        return self.train_X_wide

    @property
    def train_Y(self) -> Tensor:
        return self.train_Y_wide


__all__ = ["PartiallyObservedHybridMultiOutputModel"]
