"""Log-ratio transforms for compositional data."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from operator import index
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

_SIMPLEX_METHODS = {"none", "fractions", "clr", "alr", "ilr"}


def _normalize_method(method: str) -> str:
    normalized = method.lower()
    if normalized not in _SIMPLEX_METHODS:
        raise ValueError("method must be one of 'none', 'fractions', 'clr', 'alr', or 'ilr'.")
    return "none" if normalized == "fractions" else normalized


def _resolve_reference(reference_index: int | None, n_components: int) -> int:
    reference = n_components - 1 if reference_index is None else int(reference_index)
    if not 0 <= reference < n_components:
        raise ValueError(f"reference_index must be between 0 and {n_components - 1}.")
    return reference


def _validate_n_components(n_components: int) -> int:
    try:
        resolved = index(n_components)
    except TypeError as error:
        raise TypeError("n_components must be an integer.") from error
    if resolved < 2:
        raise ValueError("n_components must be at least 2.")
    return resolved


def _ilr_basis_values(n_components: int) -> list[list[float]]:
    n_components = _validate_n_components(n_components)
    basis = [[0.0] * (n_components - 1) for _ in range(n_components)]
    for column in range(n_components - 1):
        scale = sqrt((column + 1) * (column + 2))
        for row in range(column + 1):
            basis[row][column] = 1.0 / scale
        basis[column + 1][column] = -(column + 1) / scale
    return basis


def close_compositions(values: Any, *, pseudocount: float = 0.0) -> np.ndarray:
    """Project positive rows to the unit simplex by closure."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("Compositions must be a 2D array with at least two components.")
    if not np.isfinite(array).all():
        raise ValueError("Composition values must be finite.")
    if np.any(array < 0):
        raise ValueError("Composition values must be non-negative.")
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative.")
    if pseudocount:
        array = array + pseudocount
    row_sums = array.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each composition must contain at least one positive component.")
    return array / row_sums


def ilr_basis(n_components: int) -> np.ndarray:
    """Return the sequential binary partition (Helmert) ILR basis."""

    return np.asarray(_ilr_basis_values(n_components), dtype=float)


@dataclass(frozen=True)
class SimplexTransform:
    """Apply CLR, ALR, or ILR coordinates to closed compositions.

    Args:
        method: ``none``, ``clr``, ``alr``, or ``ilr``.
        pseudocount: Positive value added before log-ratio transforms.
        reference_index: Denominator component for ALR. Defaults to the last component.
    """

    method: str = "none"
    pseudocount: float = 1e-12
    reference_index: int | None = None

    def _method(self) -> str:
        return _normalize_method(self.method)

    def _reference(self, n_components: int) -> int:
        return _resolve_reference(self.reference_index, n_components)

    def transform(self, values: Any) -> np.ndarray:
        """Transform compositions from the simplex to model coordinates."""

        method = self._method()
        closed = close_compositions(values, pseudocount=self.pseudocount if method != "none" else 0.0)
        if method == "none":
            return closed

        log_values = np.log(closed)
        if method == "clr":
            return log_values - log_values.mean(axis=1, keepdims=True)
        if method == "alr":
            reference = self._reference(closed.shape[1])
            indices = [index for index in range(closed.shape[1]) if index != reference]
            return log_values[:, indices] - log_values[:, [reference]]
        return log_values @ ilr_basis(closed.shape[1])

    def inverse_transform(self, values: Any, *, n_components: int | None = None) -> np.ndarray:
        """Map model coordinates back to closed compositions."""

        method = self._method()
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("Transformed values must be a finite 2D array.")

        if method == "none":
            return close_compositions(array)
        if method == "clr":
            if n_components is not None and array.shape[1] != n_components:
                raise ValueError("CLR input width does not match n_components.")
            logits = array
        elif method == "alr":
            resolved_components = array.shape[1] + 1 if n_components is None else int(n_components)
            if array.shape[1] != resolved_components - 1:
                raise ValueError("ALR input width must be n_components - 1.")
            reference = self._reference(resolved_components)
            logits = np.zeros((array.shape[0], resolved_components), dtype=float)
            indices = [index for index in range(resolved_components) if index != reference]
            logits[:, indices] = array
        else:
            resolved_components = array.shape[1] + 1 if n_components is None else int(n_components)
            if array.shape[1] != resolved_components - 1:
                raise ValueError("ILR input width must be n_components - 1.")
            logits = array @ ilr_basis(resolved_components).T

        logits = logits - logits.max(axis=1, keepdims=True)
        exp_values = np.exp(logits)
        return exp_values / exp_values.sum(axis=1, keepdims=True)


class TorchSimplexTransform(nn.Module):
    """Map Torch model coordinates to fractions on the unit simplex.

    The final dimension contains composition coordinates. All leading dimensions
    are preserved, so the module accepts both ordinary batches and BoTorch
    ``batch_shape x q x d`` inputs. The inverse is implemented entirely with
    Torch operations and therefore remains in the autograd graph.

    Args:
        n_components: Number of output fractions. Must be at least two.
        method: Coordinate representation: ``fractions``, ``clr``, ``alr``, or
            ``ilr``. ``none`` is equivalent to ``fractions``.
        reference_index: Zero-based denominator component for ALR. Defaults to
            the final component.
    """

    _ilr_basis: Tensor

    def __init__(
        self,
        n_components: int,
        *,
        method: str = "ilr",
        reference_index: int | None = None,
    ) -> None:
        super().__init__()
        self.n_components = _validate_n_components(n_components)
        self.method = _normalize_method(method)
        self.reference_index = (
            None if reference_index is None else _resolve_reference(reference_index, self.n_components)
        )
        basis = torch.tensor(
            _ilr_basis_values(self.n_components),
            dtype=torch.double,
        )
        self.register_buffer("_ilr_basis", basis, persistent=True)

    @property
    def input_dim(self) -> int:
        """Return the required size of the coordinate dimension."""

        if self.method in {"alr", "ilr"}:
            return self.n_components - 1
        return self.n_components

    def forward(self, values: Tensor) -> Tensor:
        """Return closed fractions while preserving dtype, device, and gradients."""

        if not isinstance(values, Tensor):
            raise TypeError("values must be a torch.Tensor.")
        if not values.is_floating_point():
            raise TypeError("values must have a floating-point dtype.")
        if values.ndim < 1 or values.shape[-1] != self.input_dim:
            raise ValueError(f"Expected a final dimension of {self.input_dim}, got {tuple(values.shape)}.")
        if not torch.isfinite(values).all():
            raise ValueError("Transformed values must be finite.")

        if self.method == "none":
            if torch.any(values < 0):
                raise ValueError("Composition values must be non-negative.")
            totals = values.sum(dim=-1, keepdim=True)
            if torch.any(totals <= 0):
                raise ValueError("Each composition must contain at least one positive component.")
            return values / totals

        if self.method == "clr":
            logits = values
        elif self.method == "alr":
            reference = _resolve_reference(self.reference_index, self.n_components)
            zero = values.new_zeros((*values.shape[:-1], 1))
            logits = torch.cat(
                (values[..., :reference], zero, values[..., reference:]),
                dim=-1,
            )
        else:
            basis = self._ilr_basis.to(dtype=values.dtype, device=values.device)
            logits = values @ basis.transpose(-2, -1)
        return torch.softmax(logits, dim=-1)

    def extra_repr(self) -> str:
        """Return a concise representation for nested model summaries."""

        reference = ""
        if self.method == "alr":
            reference = f", reference_index={_resolve_reference(self.reference_index, self.n_components)}"
        return f"n_components={self.n_components}, method={self.method!r}{reference}"


__all__ = [
    "SimplexTransform",
    "TorchSimplexTransform",
    "close_compositions",
    "ilr_basis",
]
