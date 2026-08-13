"""Prediction formatting and decoded classification labels for tabular models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

_CLASSIFICATION_TASK_TYPES = {"binary", "multiclass", "ordinal"}
LABEL_RETURN_TYPES = {
    "label",
    "labels",
    "predicted_label",
    "predicted_labels",
    "classification_labels",
}
DATAFRAME_RETURN_TYPES = {
    "dataframe",
    "df",
    "mean_variance_dataframe",
    "mean_variance_df",
}


def prediction_array(value: Any):
    """Convert tensor-like prediction values to a two-or-more dimensional array."""

    if value is None:
        return None
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def prediction_frame(
    value: Any,
    *,
    kind: str,
    target_names: Sequence[Any],
    task_type: str | None,
):
    """Convert a prediction mean or variance array to a DataFrame."""

    import numpy as np
    import pandas as pd

    array = prediction_array(value)
    if array is None:
        return pd.DataFrame()
    names = [str(name) for name in target_names] or ["prediction"]
    n_rows = array.shape[0]
    tail = tuple(int(dim) for dim in array.shape[1:])
    flat = array.reshape(n_rows, -1)
    task = str(task_type or "").lower()

    if not tail:
        columns = [f"{names[0]}_{kind}"]
    elif len(tail) == 1 and tail[0] == len(names):
        columns = [f"{name}_{kind}" for name in names]
    elif len(tail) == 1 and task in {"multiclass", "ordinal"} and len(names) == 1:
        columns = [f"{names[0]}_class_{index}_{kind}" for index in range(tail[0])]
    elif len(tail) == 2 and task in {"multiclass", "ordinal"}:
        columns = []
        n_outputs, n_classes = tail
        for output_index in range(n_outputs):
            base = names[output_index] if output_index < len(names) else f"output_{output_index}"
            columns.extend(
                f"{base}_class_{class_index}_{kind}"
                for class_index in range(n_classes)
            )
    else:
        columns = [
            f"output_{'_'.join(str(item) for item in index)}_{kind}"
            for index in np.ndindex(tail)
        ]
    if len(columns) != flat.shape[1]:
        columns = [f"{kind}_{index}" for index in range(flat.shape[1])]
    return pd.DataFrame(flat, columns=columns)


def _call_with_fallbacks(calls: Sequence[Callable[[], Any]]) -> Any:
    last_error: TypeError | None = None
    for call in calls:
        try:
            return call()
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No model accessor call was provided.")


def _call_output_accessor(
    fn: Callable[..., Any],
    X: Tensor,
    *,
    output_index: int,
    kwargs: Mapping[str, Any],
) -> Any:
    call_kwargs = dict(kwargs)
    return _call_with_fallbacks(
        [
            lambda: fn(X=X, output_indices=[output_index], **call_kwargs),
            lambda: fn(X, output_indices=[output_index], **call_kwargs),
            lambda: fn(X=X, output_indices=[output_index]),
            lambda: fn(X, output_indices=[output_index]),
        ]
    )


def _call_single_accessor(
    fn: Callable[..., Any],
    X: Tensor,
    *,
    kwargs: Mapping[str, Any],
) -> Any:
    call_kwargs = dict(kwargs)
    return _call_with_fallbacks(
        [
            lambda: fn(X=X, **call_kwargs),
            lambda: fn(X, **call_kwargs),
            lambda: fn(X=X),
            lambda: fn(X),
        ]
    )


def _output_task_types(optimizer: Any) -> list[str]:
    if optimizer.dataset is None:
        return []
    target_names = list(optimizer.dataset.target_names)
    model = getattr(optimizer.bo, "model", None)
    if model is None:
        model = getattr(getattr(optimizer.bo, "bundle", None), "model", None)

    task_types = getattr(model, "task_types", None)
    if callable(task_types):
        task_types = task_types()
    if task_types is not None:
        values = [str(value).lower() for value in task_types]
        if len(values) == len(target_names):
            return values

    bundle = getattr(optimizer.bo, "bundle", None)
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    configured = metadata.get("output_task_types")
    if configured is not None:
        values = [str(value).lower() for value in configured]
        if len(values) == len(target_names):
            return values

    fallback = getattr(optimizer.model_config, "task_type", "regression")
    task_type = str(getattr(bundle, "task_type", fallback)).lower()
    if task_type != "hybrid":
        return [task_type] * len(target_names)

    multi_output = getattr(optimizer.model_config, "multi_output_config", None)
    output_configs = getattr(multi_output, "output_configs", None)
    if output_configs is None or len(output_configs) != len(target_names):
        return ["regression"] * len(target_names)
    result = []
    for item in output_configs:
        if isinstance(item, Mapping):
            result.append(str(item.get("task_type", "regression")).lower())
        elif isinstance(item, str):
            result.append(item.lower())
        else:
            result.append(str(getattr(item, "task_type", "regression")).lower())
    return result


def _select_output(value: Any, *, output_index: int, num_outputs: int) -> Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if num_outputs > 1 and tensor.ndim >= 3 and tensor.shape[-2] == num_outputs:
        tensor = tensor[..., output_index, :]
    return tensor


def _reduce_to_rows(probabilities: Tensor, *, n_rows: int) -> Tensor:
    probabilities = probabilities.detach()
    while probabilities.ndim > 2:
        if probabilities.shape[-2] == n_rows:
            probabilities = probabilities.reshape(
                -1,
                n_rows,
                probabilities.shape[-1],
            ).mean(dim=0)
        elif probabilities.shape[0] == n_rows:
            probabilities = probabilities.reshape(
                n_rows,
                -1,
                probabilities.shape[-1],
            ).mean(dim=1)
        else:
            probabilities = probabilities.mean(dim=0)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)
    if probabilities.ndim != 2:
        raise RuntimeError(
            f"Class probabilities cannot be reduced to rows x classes: {tuple(probabilities.shape)}."
        )
    if probabilities.shape[0] != n_rows:
        if probabilities.shape[0] % n_rows != 0:
            raise RuntimeError("Class probabilities cannot be aligned with prediction rows.")
        probabilities = probabilities.reshape(
            n_rows,
            -1,
            probabilities.shape[-1],
        ).mean(dim=1)
    return probabilities


def _probabilities_for_output(
    optimizer: Any,
    X: Tensor,
    *,
    output_index: int,
    task_type: str,
    posterior_kwargs: Mapping[str, Any],
    n_rows: int,
) -> Tensor | None:
    model = getattr(optimizer.bo, "model", None)
    if model is None:
        model = getattr(getattr(optimizer.bo, "bundle", None), "model", None)
    if model is None:
        return None
    num_outputs = len(optimizer.dataset.target_names)

    fn = getattr(model, "class_probs_list", None)
    if callable(fn):
        values = _call_output_accessor(
            fn,
            X,
            output_index=output_index,
            kwargs=posterior_kwargs,
        )
        if len(values) != 1:
            raise RuntimeError("class_probs_list must return one value per requested output.")
        return _reduce_to_rows(
            _select_output(values[0], output_index=output_index, num_outputs=num_outputs),
            n_rows=n_rows,
        )

    fn = getattr(model, "class_probs", None)
    if callable(fn):
        value = _call_single_accessor(fn, X, kwargs=posterior_kwargs)
        return _reduce_to_rows(
            _select_output(value, output_index=output_index, num_outputs=num_outputs),
            n_rows=n_rows,
        )

    posterior = None
    fn = getattr(model, "probability_posterior", None)
    if callable(fn):
        try:
            posterior = _call_output_accessor(
                fn,
                X,
                output_index=output_index,
                kwargs=posterior_kwargs,
            )
        except (TypeError, NotImplementedError):
            posterior = _call_single_accessor(fn, X, kwargs=posterior_kwargs)
    elif task_type in _CLASSIFICATION_TASK_TYPES:
        fn = getattr(model, "posterior", None)
        if callable(fn):
            try:
                posterior = _call_output_accessor(
                    fn,
                    X,
                    output_index=output_index,
                    kwargs=posterior_kwargs,
                )
            except (TypeError, NotImplementedError):
                posterior = _call_single_accessor(fn, X, kwargs=posterior_kwargs)

    mean = getattr(posterior, "mean", None)
    if mean is None:
        return None
    probabilities = _reduce_to_rows(
        _select_output(mean, output_index=output_index, num_outputs=num_outputs),
        n_rows=n_rows,
    )
    if task_type == "binary" and probabilities.shape[-1] == 1:
        p1 = probabilities[..., 0].clamp(0.0, 1.0)
        probabilities = torch.stack([1.0 - p1, p1], dim=-1)
    return probabilities


def _predicted_classes(
    optimizer: Any,
    X: Tensor,
    *,
    output_index: int,
    posterior_kwargs: Mapping[str, Any],
    n_rows: int,
) -> Tensor | None:
    model = getattr(optimizer.bo, "model", None)
    if model is None:
        model = getattr(getattr(optimizer.bo, "bundle", None), "model", None)
    fn = getattr(model, "predict_class", None)
    if not callable(fn):
        return None
    try:
        values = _call_output_accessor(
            fn,
            X,
            output_index=output_index,
            kwargs=posterior_kwargs,
        )
    except (TypeError, NotImplementedError):
        values = _call_single_accessor(fn, X, kwargs=posterior_kwargs)
    tensor = values if torch.is_tensor(values) else torch.as_tensor(values)
    if tensor.ndim >= 2 and tensor.shape[-1] == len(optimizer.dataset.target_names):
        tensor = tensor[..., output_index]
    tensor = tensor.detach().reshape(-1)
    if tensor.numel() != n_rows:
        if tensor.numel() % n_rows != 0:
            raise RuntimeError("Predicted classes cannot be aligned with prediction rows.")
        tensor = tensor.reshape(n_rows, -1).double().mean(dim=1).round()
    return tensor.long()


def classification_prediction_dataframe(
    optimizer: Any,
    X: Tensor,
    *,
    posterior_kwargs: Mapping[str, Any] | None = None,
    binary_threshold: float = 0.5,
):
    """Build decoded prediction-label columns for classification outputs."""

    import pandas as pd

    if optimizer.dataset is None:
        raise RuntimeError("No fitted tabular dataset found. Call fit() first.")
    threshold = float(binary_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("binary_threshold must be between 0 and 1.")

    target_names = list(optimizer.dataset.target_names)
    task_types = _output_task_types(optimizer)
    n_rows = int(X.shape[-2]) if X.ndim >= 2 else int(X.shape[0])
    posterior_kwargs = dict(posterior_kwargs or {})
    columns: dict[str, Any] = {}
    inverse_maps = dict(
        getattr(optimizer.dataset, "inverse_target_category_maps", None) or {}
    )

    for output_index, (target_name, task_type) in enumerate(
        zip(target_names, task_types, strict=True)
    ):
        if task_type not in _CLASSIFICATION_TASK_TYPES:
            continue
        probabilities = _probabilities_for_output(
            optimizer,
            X,
            output_index=output_index,
            task_type=task_type,
            posterior_kwargs=posterior_kwargs,
            n_rows=n_rows,
        )
        predicted_probability = None
        if probabilities is not None:
            probabilities = probabilities.clamp_min(0.0)
            if task_type == "binary":
                p1 = (
                    probabilities[..., 0]
                    if probabilities.shape[-1] == 1
                    else probabilities[..., 1]
                ).clamp(0.0, 1.0)
                predicted_class = (p1 >= threshold).long()
                predicted_probability = torch.where(
                    predicted_class == 1,
                    p1,
                    1.0 - p1,
                )
            else:
                probabilities = probabilities / probabilities.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
                predicted_probability, predicted_class = probabilities.max(dim=-1)
        else:
            predicted_class = _predicted_classes(
                optimizer,
                X,
                output_index=output_index,
                posterior_kwargs=posterior_kwargs,
                n_rows=n_rows,
            )
            if predicted_class is None:
                raise AttributeError(
                    f"Could not obtain predicted classes for output {target_name!r}."
                )

        class_indices = predicted_class.detach().cpu().long().tolist()
        inverse = inverse_maps.get(target_name) or inverse_maps.get(str(target_name))
        labels = [
            inverse.get(int(index), int(index)) if inverse is not None else int(index)
            for index in class_indices
        ]
        prefix = str(target_name)
        columns[f"{prefix}_predicted_class_index"] = class_indices
        columns[f"{prefix}_predicted_label"] = labels
        if predicted_probability is not None:
            columns[f"{prefix}_predicted_probability"] = (
                predicted_probability.detach().cpu().tolist()
            )
    return pd.DataFrame(columns)


__all__ = [
    "DATAFRAME_RETURN_TYPES",
    "LABEL_RETURN_TYPES",
    "classification_prediction_dataframe",
    "prediction_array",
    "prediction_frame",
]
