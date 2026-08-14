"""Log-ratio transforms for compositional data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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

    if n_components < 2:
        raise ValueError("n_components must be at least 2.")
    basis = np.zeros((n_components, n_components - 1), dtype=float)
    for column in range(n_components - 1):
        scale = np.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / scale
        basis[column + 1, column] = -(column + 1) / scale
    return basis


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
        method = self.method.lower()
        if method not in {"none", "fractions", "clr", "alr", "ilr"}:
            raise ValueError("method must be one of 'none', 'fractions', 'clr', 'alr', or 'ilr'.")
        return "none" if method == "fractions" else method

    def _reference(self, n_components: int) -> int:
        reference = n_components - 1 if self.reference_index is None else int(self.reference_index)
        if not 0 <= reference < n_components:
            raise ValueError(f"reference_index must be between 0 and {n_components - 1}.")
        return reference

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
