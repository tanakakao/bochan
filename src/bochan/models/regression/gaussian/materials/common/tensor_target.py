"""Tensor-valued target layouts for MLIP residual Gaussian processes."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import torch
from torch import Tensor

TensorTargetKind = Literal["force", "stress"]


@dataclass(frozen=True)
class TensorTargetLayout:
    """Describe how one tensor-valued physical target is flattened for BoTorch.

    BoTorch wide-output Gaussian models operate on ``[n, m]`` targets. MLIP
    observables such as forces and stress naturally have additional physical
    axes, so bochan keeps the physical tensor shape as metadata and flattens only
    the output axes. No component reordering or unit conversion is performed.
    """

    kind: TensorTargetKind
    tensor_shape: tuple[int, ...]
    component_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"force", "stress"}:
            raise ValueError("kind must be 'force' or 'stress'.")
        if not self.tensor_shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.tensor_shape
        ):
            raise ValueError("tensor_shape must contain positive integer dimensions.")
        if self.kind == "force":
            if len(self.tensor_shape) != 2 or self.tensor_shape[-1] != 3:
                raise ValueError("Force tensor_shape must be (num_atoms, 3).")
        elif self.tensor_shape != (3, 3):
            raise ValueError("Stress tensor_shape must be (3, 3).")

        if self.component_names is not None:
            names = tuple(str(name) for name in self.component_names)
            if len(names) != self.output_dim:
                raise ValueError(
                    "component_names length must match the flattened output dimension."
                )
            if any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("component_names must be non-empty and unique.")
            object.__setattr__(self, "component_names", names)

    @classmethod
    def force(cls, num_atoms: int) -> TensorTargetLayout:
        """Create a fixed-topology Cartesian force layout with ``3 * N`` outputs."""

        if isinstance(num_atoms, bool) or not isinstance(num_atoms, int) or num_atoms <= 0:
            raise ValueError("num_atoms must be a positive integer.")
        axes = ("x", "y", "z")
        names = tuple(
            f"force_atom_{atom}_{axis}"
            for atom in range(num_atoms)
            for axis in axes
        )
        return cls(kind="force", tensor_shape=(num_atoms, 3), component_names=names)

    @classmethod
    def stress(cls) -> TensorTargetLayout:
        """Create a full Cartesian 3x3 stress layout with nine outputs."""

        axes = ("x", "y", "z")
        names = tuple(f"stress_{row}{col}" for row in axes for col in axes)
        return cls(kind="stress", tensor_shape=(3, 3), component_names=names)

    @property
    def output_dim(self) -> int:
        """Return the number of flattened scalar GP outputs."""

        return int(prod(self.tensor_shape))

    def flatten(self, values: Tensor, *, n: int | None = None) -> Tensor:
        """Validate and flatten physical tensors to ``[..., output_dim]``."""

        if not torch.is_tensor(values):
            raise TypeError("values must be a Tensor.")
        if values.ndim >= len(self.tensor_shape) and tuple(values.shape[-len(self.tensor_shape) :]) == self.tensor_shape:
            flat = values.reshape(*values.shape[: -len(self.tensor_shape)], self.output_dim)
        elif values.ndim >= 1 and values.shape[-1] == self.output_dim:
            flat = values
        else:
            raise ValueError(
                f"{self.kind} targets must end with {self.tensor_shape} or ({self.output_dim},); "
                f"got {tuple(values.shape)}."
            )
        if n is not None and (flat.ndim != 2 or flat.shape[0] != n):
            raise ValueError(
                f"Flattened {self.kind} targets must have shape [{n}, {self.output_dim}]."
            )
        if not torch.isfinite(flat).all():
            raise FloatingPointError(f"{self.kind} targets contain non-finite values.")
        return flat

    def unflatten(self, values: Tensor) -> Tensor:
        """Restore the physical tensor axes from flattened GP outputs."""

        if not torch.is_tensor(values):
            raise TypeError("values must be a Tensor.")
        if values.ndim < 1 or values.shape[-1] != self.output_dim:
            raise ValueError(
                f"Flattened values must end with output_dim={self.output_dim}; "
                f"got {tuple(values.shape)}."
            )
        return values.reshape(*values.shape[:-1], *self.tensor_shape)

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible layout metadata."""

        return {
            "kind": self.kind,
            "tensor_shape": list(self.tensor_shape),
            "output_dim": self.output_dim,
            "component_names": None if self.component_names is None else list(self.component_names),
        }


__all__ = ["TensorTargetKind", "TensorTargetLayout"]
