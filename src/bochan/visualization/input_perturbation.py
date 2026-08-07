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
    to_tensor_like,
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


def _hybrid_display_prediction_mean_std(
    obj: Any,
    X: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return Hybrid model predictions on the human-facing display scale.

    ``HybridMultiOutputModel.posterior()`` defaults to ``output_mode='objective'``.
    That scale is correct for acquisition optimization, but it is not suitable for
    prediction plots: a target-value regression becomes ``-abs(y - target)`` and a
    minimized regression is sign-flipped.  Visualization must instead use the raw
    prediction scale.  Ordinal outputs are displayed as expected rank, matching
    the Web candidate table.
    """

    model = get_model(obj)
    specs = list(getattr(model, "specs", []) or [])
    posterior_fn = getattr(model, "posterior", None)
    if not specs or not callable(posterior_fn):
        return None

    X_t = to_tensor_like(X, obj)
    try:
        posterior = posterior_fn(X_t, output_mode="mean")
    except TypeError:
        return None

    mean = getattr(posterior, "mean", None)
    variance = getattr(posterior, "variance", None)
    if mean is None or variance is None:
        return None

    ordinal_indices = [
        index
        for index, spec in enumerate(specs)
        if str(getattr(spec, "task_type", "")) == "ordinal"
    ]
    class_probs_list = getattr(model, "class_probs_list", None)
    if ordinal_indices and callable(class_probs_list):
        import torch

        if torch.is_tensor(mean) and torch.is_tensor(variance):
            mean = mean.clone()
            variance = variance.clone()
            for index in ordinal_indices:
                probs = class_probs_list(X_t, output_indices=[index])[0]
                ranks = torch.arange(
                    probs.shape[-1],
                    device=probs.device,
                    dtype=probs.dtype,
                )
                ordinal_mean = (probs * ranks).sum(dim=-1)
                ordinal_variance = (
                    probs * (ranks - ordinal_mean.unsqueeze(-1)).pow(2)
                ).sum(dim=-1)
                if ordinal_mean.ndim == mean.ndim and ordinal_mean.shape[-1] == 1:
                    ordinal_mean = ordinal_mean.squeeze(-1)
                    ordinal_variance = ordinal_variance.squeeze(-1)
                mean[..., index] = ordinal_mean
                variance[..., index] = ordinal_variance

    mean_arr = ensure_2d(mean)
    std_arr = np.sqrt(np.clip(ensure_2d(variance), 0.0, None))
    return mean_arr, std_arr


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


def aggregate_input_perturbation_probabilities(
    values: Any,
    *,
    n_points: int,
    n_w: int | None,
) -> np.ndarray:
    """Average expanded class probabilities back to one row per input point.

    Input perturbation commonly flattens ``n_points x n_w`` into one candidate
    axis. Probability visualization must restore that grouping before checking
    that the number of probability rows matches the original input grid.

    Additional leading dimensions, such as posterior sample or model batch
    dimensions, are averaged together after preserving the point-major
    ``n_points x n_w`` grouping.
    """

    arr = np.asarray(to_numpy(values), dtype=float)
    if n_w is None or n_w <= 1 or n_points <= 0 or arr.ndim < 2:
        return arr

    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    while arr.ndim > 2 and arr.shape[-2] == 1:
        arr = np.squeeze(arr, axis=-2)

    expanded_points = n_points * n_w
    if (
        arr.ndim == 2
        and arr.shape[0] != expanded_points
        and arr.shape[1] == expanded_points
    ):
        arr = arr.T

    n_classes = int(arr.shape[-1])
    flat = arr.reshape(-1, n_classes)
    if flat.shape[0] % expanded_points != 0:
        return arr

    grouped = flat.reshape(-1, n_points, n_w, n_classes)
    return grouped.mean(axis=(0, 2))


def prediction_mean_std(
    obj: Any,
    X: Any,
    *,
    uncertainty_kind: str = "epistemic",
    num_uncertainty_samples: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return display-scale visualization moments with perturbations aggregated."""

    hybrid_display = _hybrid_display_prediction_mean_std(obj, X)
    if hybrid_display is None:
        mean, std = _base_prediction_mean_std(
            obj,
            X,
            uncertainty_kind=uncertainty_kind,
            num_uncertainty_samples=num_uncertainty_samples,
        )
    else:
        mean, std = hybrid_display
    return aggregate_input_perturbation_moments(
        mean,
        std,
        n_points=_num_input_points(X),
        n_w=input_perturbation_n_w(obj),
    )


__all__ = [
    "aggregate_input_perturbation_moments",
    "aggregate_input_perturbation_probabilities",
    "input_perturbation_n_w",
    "prediction_mean_std",
]
