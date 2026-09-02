"""Fusion contracts for material representations and process features."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn


class MaterialProcessFusion(nn.Module, ABC):
    """Base contract for combining material and process representations."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the width of the fused representation."""

        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        material_features: Tensor,
        process_features: Tensor | None = None,
    ) -> Tensor:
        """Return one fused feature tensor while preserving leading dimensions."""

        raise NotImplementedError


class ConcatFusion(MaterialProcessFusion):
    """Concatenate material and process features along the final dimension."""

    def __init__(self, material_dim: int, process_dim: int = 0) -> None:
        super().__init__()
        self.material_dim = int(material_dim)
        self.process_dim = int(process_dim)
        if self.material_dim <= 0:
            raise ValueError("material_dim must be positive.")
        if self.process_dim < 0:
            raise ValueError("process_dim must be non-negative.")

    @property
    def output_dim(self) -> int:
        """Return ``material_dim + process_dim``."""

        return self.material_dim + self.process_dim

    @staticmethod
    def _validate_tensor(name: str, value: Tensor, expected_dim: int) -> None:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a Tensor.")
        if value.ndim == 0:
            raise ValueError(f"{name} must have a feature dimension.")
        if value.shape[-1] != expected_dim:
            raise ValueError(
                f"{name} width does not match its configured dimension: "
                f"{value.shape[-1]} != {expected_dim}."
            )

    def forward(
        self,
        material_features: Tensor,
        process_features: Tensor | None = None,
    ) -> Tensor:
        """Return the concatenated material/process representation."""

        self._validate_tensor("material_features", material_features, self.material_dim)
        if process_features is None:
            if self.process_dim:
                raise ValueError(
                    "process_features is required when process_dim is positive."
                )
            return material_features

        self._validate_tensor("process_features", process_features, self.process_dim)
        if material_features.shape[:-1] != process_features.shape[:-1]:
            raise ValueError(
                "material_features and process_features must have identical "
                "leading dimensions. "
                f"Got {tuple(material_features.shape[:-1])} and "
                f"{tuple(process_features.shape[:-1])}."
            )
        if material_features.device != process_features.device:
            raise ValueError(
                "material_features and process_features must be on the same device."
            )
        if material_features.dtype != process_features.dtype:
            raise ValueError(
                "material_features and process_features must have the same dtype."
            )
        return torch.cat([material_features, process_features], dim=-1)


def build_material_process_fusion(
    fusion: Literal["concat"] | MaterialProcessFusion = "concat",
    *,
    material_dim: int,
    process_dim: int = 0,
) -> MaterialProcessFusion:
    """Build the requested material/process fusion module."""

    if isinstance(fusion, MaterialProcessFusion):
        if fusion.output_dim <= 0:
            raise ValueError("fusion.output_dim must be positive.")
        return fusion
    if fusion == "concat":
        return ConcatFusion(material_dim=material_dim, process_dim=process_dim)
    raise ValueError("fusion must be 'concat' or a MaterialProcessFusion instance.")


__all__ = [
    "ConcatFusion",
    "MaterialProcessFusion",
    "build_material_process_fusion",
]
