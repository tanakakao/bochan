"""Prediction-shape helpers for one-to-many input transforms."""

from __future__ import annotations

from typing import Any


def normalize_prediction_rows(value: Any, *, n_rows: int) -> Any:
    """Normalize posterior values to ``[n, m]``.

    BoTorch's ``InputPerturbation`` expands each original input into consecutive
    perturbation samples. When predictions contain a whole-number multiple of
    ``n_rows``, this helper infers that expansion factor and averages each
    consecutive block back to one display prediction per original input.

    Args:
        value: Posterior mean or variance tensor-like value.
        n_rows: Number of original, unperturbed input rows.

    Returns:
        Tensor normalized to shape ``[n_rows, n_outputs]``.

    Raises:
        ValueError: If ``n_rows`` is not positive.
        RuntimeError: If the prediction cannot be mapped to ``n_rows``.
    """

    import torch

    n_rows = int(n_rows)
    if n_rows <= 0:
        raise ValueError("n_rows must be a positive integer.")

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

    return flattened.reshape(n_rows, n_w, n_outputs).mean(dim=1)


__all__ = ["normalize_prediction_rows"]
