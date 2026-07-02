"""Prediction aggregation for models using input perturbation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .utils import (
    ensure_2d,
    get_model,
    prediction_mean_std as _base_prediction_mean_std,
    to_numpy,
)


def _n_w_from_model_config(model_config: Any | None) -> int | None:
    """Return ``n_w`` when a model config enables input perturbation."""

    if model_config is None:
        return None
    transform_config = getattr(model_config, "input_transform_config", None)
    if transform_config is None:
        return None
    if not bool(getattr(transform_config, "perturbation", False)):
        return None
    n_w = getattr(transform_config, "n_w", None)
    if n_w is None:
        return None
    n_w = int(n_w)
    return n_w if n_w > 1 else None


def _uniform_value(values: Iterable[int | None]) -> int | None:
    """Return the common non-None value, or None when values disagree."""

    filtered = [int(value) for value in values if value is not None]
    if filtered and len(set(filtered)) == 1:
        return filtered[0]
    return None


def _n_w_from_input_transform(transform: Any | None) -> int | None:
    """Infer ``n_w`` from a built BoTorch input transform as a fallback."""

    if transform is None:
        return None

    perturbation_set = getattr(transform, "perturbation_set", None)
    if perturbation_set is not None and hasattr(perturbation_set, "shape"):
        n_w = int(perturbation_set.shape[-2])
        return n_w if n_w > 1 else None

    for name in ("perturb", "input_perturbation"):
        nested = getattr(transform, name, None)
        n_w = _n_w_from_input_transform(nested)
        if n_w is not None:
            return n_w
    return None


def input_perturbation_n_w(obj: Any) -> int | None:
    """Infer the perturbation sample count from an optimizer, bundle, or model."""

    bundle = getattr(obj, "bundle", None)
    for candidate in (obj, bundle):
        n_w = _n_w_from_model_config(getattr(candidate, "model_config", None))
        if n_w is not None:
            return n_w

    sub_bundles = []
    metadata = getattr(bundle, "metadata", None)
    if isinstance(metadata, dict):
        sub_bundles = list(metadata.get("sub_bundles", []) or [])
    n_w = _uniform_value(
        _n_w_from_model_config(getattr(sub_bundle, "model_config", None))
        for sub_bundle in sub_bundles
    )
    if n_w is not None:
        return n_w

    model = get_model(obj)
    return _n_w_from_input_transform(getattr(model, "input_transform", None))


def _num_input_points(X: Any) -> int:
    """Return the number of original points before input perturbation expansion."""

    arr = to_numpy(X)
    if arr.ndim <= 1:
        return 1
    return int(np.prod(arr.shape[:-1]))


def aggregate_input_perturbation_moments(
    mean: Any,
    std: Any,
    *,
    n_points: int,
    n_w: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate expanded perturbation predictions back to one row per point.

    The mean is averaged over the perturbation samples. The variance follows the
    law of total variance, combining posterior variance within each perturbed
    input with variation of the posterior means between perturbations.
    """

    mean_arr = ensure_2d(mean)
    std_arr = ensure_2d(std)
    if n_w is None or n_w <= 1 or n_points <= 0:
        return mean_arr, std_arr
    if mean_arr.shape != std_arr.shape:
        return mean_arr, std_arr
    if mean_arr.shape[0] != n_points * n_w:
        return mean_arr, std_arr

    mean_grouped = mean_arr.reshape(n_points, n_w, -1)
    var_grouped = np.square(std_arr).reshape(n_points, n_w, -1)

    aggregated_mean = mean_grouped.mean(axis=1)
    second_moment = (var_grouped + np.square(mean_grouped)).mean(axis=1)
    aggregated_var = np.clip(
        second_moment - np.square(aggregated_mean),
        0.0,
        None,
    )
    return aggregated_mean, np.sqrt(aggregated_var)


def prediction_mean_std(
    obj: Any,
    X: Any,
    *,
    uncertainty_kind: str = "epistemic",
    num_uncertainty_samples: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return visualization moments with input perturbations aggregated."""

    mean, std = _base_prediction_mean_std(
        obj,
        X,
        uncertainty_kind=uncertainty_kind,
        num_uncertainty_samples=num_uncertainty_samples,
    )
    return aggregate_input_perturbation_moments(
        mean,
        std,
        n_points=_num_input_points(X),
        n_w=input_perturbation_n_w(obj),
    )


__all__ = [
    "aggregate_input_perturbation_moments",
    "input_perturbation_n_w",
    "prediction_mean_std",
]
