"""Input-perturbation-aware multiclass and ordinal probability helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import multiclass as _multiclass
from . import ordinal as _ordinal
from .input_perturbation import (
    aggregate_input_perturbation_probabilities,
    input_perturbation_n_w,
)
from .utils import ensure_2d


def _aggregate_probability_values(
    obj: Any,
    X: Any,
    values: Any,
) -> tuple[np.ndarray, int]:
    """Return probability-like values aggregated to the original input count."""

    n_points = len(ensure_2d(X))
    aggregated = aggregate_input_perturbation_probabilities(
        values,
        n_points=n_points,
        n_w=input_perturbation_n_w(obj),
    )
    return aggregated, n_points


def multiclass_probabilities(
    obj: Any,
    X: Any,
    *,
    output_index: int = 0,
) -> np.ndarray:
    """Return multiclass probabilities with perturbation rows averaged."""

    values = _multiclass._probability_tensor(
        obj,
        X,
        output_index=output_index,
    )
    aggregated, n_points = _aggregate_probability_values(obj, X, values)
    return _multiclass._as_probability_matrix(
        aggregated,
        n_points=n_points,
    )


def ordinal_probabilities(
    obj: Any,
    X: Any,
    *,
    output_index: int = 0,
) -> np.ndarray:
    """Return ordinal probabilities with perturbation rows averaged."""

    values = _ordinal._ordinal_probability_tensor(
        obj,
        X,
        output_index=output_index,
    )
    aggregated, n_points = _aggregate_probability_values(obj, X, values)
    return _multiclass._as_probability_matrix(
        aggregated,
        n_points=n_points,
    )


__all__ = [
    "multiclass_probabilities",
    "ordinal_probabilities",
]
