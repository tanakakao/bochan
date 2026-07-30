"""Prediction-shape helpers for one-to-many input transforms."""

from __future__ import annotations

import math
from typing import Any, Literal

RiskType = Literal["var", "cvar"] | None


def normalize_prediction_rows(
    value: Any,
    *,
    n_rows: int,
    risk_type: RiskType = None,
    alpha: float = 0.5,
) -> Any:
    """Normalize posterior values to ``[n, m]``.

    BoTorch's ``InputPerturbation`` expands each original input into consecutive
    perturbation samples. When predictions contain a whole-number multiple of
    ``n_rows``, this helper infers that expansion factor and aggregates each
    consecutive block back to one value per original input.

    ``risk_type=None`` keeps the previous behavior and returns the mean over all
    perturbations. ``var`` returns the boundary of the lower ``alpha`` tail and
    ``cvar`` returns the mean of that lower tail. Web optimization passes values
    that are already aligned to the maximization direction, so the lower tail is
    the adverse side for both originally maximized and minimized targets.

    Args:
        value: Posterior mean or variance tensor-like value.
        n_rows: Number of original, unperturbed input rows.
        risk_type: ``None`` for mean aggregation, ``var``, or ``cvar``.
        alpha: Fraction of perturbation samples included in the adverse tail.

    Returns:
        Tensor normalized to shape ``[n_rows, n_outputs]``.

    Raises:
        ValueError: If an aggregation setting is invalid.
        RuntimeError: If the prediction cannot be mapped to ``n_rows``.
    """

    import torch

    n_rows = int(n_rows)
    if n_rows <= 0:
        raise ValueError("n_rows must be a positive integer.")
    if risk_type not in {None, "var", "cvar"}:
        raise ValueError("risk_type must be None, 'var', or 'cvar'.")
    if risk_type is not None and not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0, 1] for VaR/CVaR aggregation.")

    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)

    if tensor.ndim < 2:
        raise RuntimeError(
            "Could not normalize prediction to [n, m]. "
            f"shape={tuple(tensor.shape)}, n_rows={n_rows}"
        )

    n_outputs = int(tensor.shape[-1])
    flattened = tensor.reshape(-1, n_outputs)
    expanded_rows = int(flattened.shape[0])

    if expanded_rows == n_rows:
        return flattened

    if expanded_rows % n_rows != 0:
        raise RuntimeError(
            "Could not normalize prediction to [n, m]. "
            f"shape={tuple(tensor.shape)}, n_rows={n_rows}"
        )

    n_w = expanded_rows // n_rows
    if n_w <= 1:
        raise RuntimeError(
            "Could not normalize prediction to [n, m]. "
            f"shape={tuple(tensor.shape)}, n_rows={n_rows}"
        )

    values_w = flattened.reshape(n_rows, n_w, n_outputs)
    if risk_type is None:
        return values_w.mean(dim=1)

    k = max(1, int(math.ceil(n_w * float(alpha))))
    adverse_tail = torch.sort(values_w, dim=1).values[:, :k, :]
    if risk_type == "var":
        return adverse_tail[:, k - 1, :]
    return adverse_tail.mean(dim=1)


__all__ = ["RiskType", "normalize_prediction_rows"]
