"""Shared mixed-process layout utilities for material-aware Gaussian models."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.utils.transforms import normalize_indices
from torch import Tensor


@dataclass(frozen=True)
class MixedProcessLayout:
    """Describe material, numeric-process, and categorical input partitions."""

    input_dim: int
    material_dims: tuple[int, ...]
    categorical_dims: tuple[int, ...]
    continuous_dims: tuple[int, ...]
    numeric_process_dims: tuple[int, ...]

    @property
    def categorical_process_dim(self) -> int:
        return len(self.categorical_dims)

    @property
    def numeric_process_dim(self) -> int:
        return len(self.numeric_process_dims)


def resolve_mixed_process_layout(
    input_dim: int,
    cat_dims: Sequence[int],
    *,
    material_dims: Sequence[int] = (0,),
    require_categorical: bool = True,
) -> MixedProcessLayout:
    """Resolve one stable mixed-input partition using BoTorch index semantics."""

    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError("input_dim must be a positive integer.")
    normalized_cat_dims = tuple(normalize_indices(indices=list(cat_dims), d=input_dim))
    if require_categorical and not normalized_cat_dims:
        raise ValueError("At least one categorical process dimension is required.")
    if len(set(normalized_cat_dims)) != len(normalized_cat_dims):
        raise ValueError("cat_dims must not contain duplicates.")

    normalized_material_dims = tuple(normalize_indices(indices=list(material_dims), d=input_dim))
    if not normalized_material_dims:
        raise ValueError("material_dims must not be empty.")
    if len(set(normalized_material_dims)) != len(normalized_material_dims):
        raise ValueError("material_dims must not contain duplicates.")
    overlap = set(normalized_cat_dims) & set(normalized_material_dims)
    if overlap:
        raise ValueError("Material selector/coordinate dimensions cannot be categorical process dimensions.")

    categorical = set(normalized_cat_dims)
    material = set(normalized_material_dims)
    continuous_dims = tuple(index for index in range(input_dim) if index not in categorical)
    numeric_process_dims = tuple(index for index in continuous_dims if index not in material)
    return MixedProcessLayout(
        input_dim=input_dim,
        material_dims=normalized_material_dims,
        categorical_dims=normalized_cat_dims,
        continuous_dims=continuous_dims,
        numeric_process_dims=numeric_process_dims,
    )


def resolve_mixed_process_input_transform(
    train_X: Tensor,
    layout: MixedProcessLayout,
    input_transform: str | InputTransform | None,
) -> str | InputTransform | None:
    """Resolve DEFAULT to normalization of numeric process columns only."""

    if not torch.is_tensor(train_X):
        raise TypeError("train_X must be a Tensor.")
    if train_X.ndim != 2 or train_X.shape[-1] != layout.input_dim:
        raise ValueError("train_X must have shape [n, layout.input_dim].")
    if not isinstance(input_transform, str) or input_transform.upper() != "DEFAULT":
        return input_transform
    if not layout.numeric_process_dims:
        return None
    return Normalize(d=layout.input_dim, indices=list(layout.numeric_process_dims))


def select_continuous_process_branch(X: Tensor, layout: MixedProcessLayout) -> Tensor:
    """Return material dimensions plus numeric process dimensions in raw order."""

    if not torch.is_tensor(X):
        raise TypeError("X must be a Tensor.")
    if X.ndim == 0 or X.shape[-1] != layout.input_dim:
        raise ValueError("X width must equal layout.input_dim.")
    indices = torch.tensor(layout.continuous_dims, dtype=torch.long, device=X.device)
    return X.index_select(-1, indices)


__all__ = [
    "MixedProcessLayout",
    "resolve_mixed_process_input_transform",
    "resolve_mixed_process_layout",
    "select_continuous_process_branch",
]
