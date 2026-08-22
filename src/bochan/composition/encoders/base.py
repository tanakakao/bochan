"""Base contract for material-representation encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


class MaterialEncoder(nn.Module, ABC):
    """Encode a material-specific input into a fixed-width representation.

    Concrete encoders own their input signature because composition tensors,
    graph structures, and other material representations require different
    inputs.  Every implementation must preserve its input leading dimensions
    in the returned tensor and expose the final feature width through
    :attr:`output_dim`.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the width of the material representation."""

        raise NotImplementedError

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Return a material representation with ``output_dim`` features."""

        raise NotImplementedError


__all__ = ["MaterialEncoder"]
