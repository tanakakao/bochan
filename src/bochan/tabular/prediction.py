"""Prediction-label helpers for the canonical tabular optimizer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bochan.models._posterior_utils import extract_mean_and_variance

_LABEL_RETURN_TYPES = {"labels", "label", "classes", "class_labels"}
_DATAFRAME_RETURN_TYPES = {
    "dataframe",
    "df",
    "mean",
    "posterior_mean",
    "posterior mean",
}


def _task_type(optimizer: Any) -> str:
    bundle = getattr(optimizer.bo, "bundle", None)
    if bundle is not None:
        return str(bundle.task_type)
    return str(getattr(optimizer.model_config, "task_type", "regression"))


def _model_type(config: Any) -> str:
    if isinstance(config, Mapping):
        return str(config.get("model_type", "base"))
    return str(getattr(config, "model_type", "base"))


def _normalize_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _classification_output_configs(optimizer: Any) -> list[Any]:
    bundle = getattr(optimizer.bo, "bundle", None)
    metadata = getattr(bundle, "metadata", {}) if bundle is not None else {}
    configured = metadata.get("output_configs")
    if configured is not None:
        return list(configured)

    multi_output = getattr(optimizer.model_config, "multi_output_config", None)
    if multi_output is None:
        return []
    if isinstance(multi_output, Mapping):
        output_configs = multi_output.get("output_configs")
    else:
        output_configs = getattr(multi_output, "output_configs", None)
    return list(output_configs or ())


def _prediction_tensor_and_index(optimizer: Any, data: Any) -> tuple[Any, Any | None]:
    import pandas as pd

    original_index = data.index if isinstance(data, pd.DataFrame) else None
    X, _, _, _ = optimizer._prepare_prediction(data)
    return X, original_index


def _category_maps(optimizer: Any) -> dict[Any, dict[Any, int]]:
    dataset = getattr(optimizer, "dataset", None)
    if dataset is None:
        return {}
    return dict(getattr(dataset, "target_category_maps", None) or {})


def _target_names(optimizer: Any, n_outputs: int) -> list[Any]:
    dataset = getattr(optimizer, "dataset", None)
    names = list(getattr(dataset, "target_names", ()) or ()) if dataset is not None else []
    if len(names) >= n_outputs:
        return names[:n_outputs]
    return [*names, *[f"output_{index}" for index in range(len(names), n_outputs)]]


def _inverse_category_map(
    category_maps: Mapping[Any, Mapping[Any, int]],
    output_name: Any,
) -> dict[int, Any] | None:
    category_map = category_maps.get(output_name)
    if category_map is None:
        category_map = category_maps.get(str(output_name))
    if category_map is None:
        return None
    return {int(index): category for category, index in category_map.items()}


def _decode_indices(
    indices: np.ndarray,
    inverse_map: Mapping[int, Any] | None,
) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if inverse_map is None:
        return values
    return np.asarray(
        [inverse_map.get(int(index), int(index)) for index in values],
        dtype=object,
    )


def _multiclass_probabilities(model: Any, X: Any, posterior_kwargs: dict[str, Any]) -> Any:
    for method_name in (
        "class_probs",
        "predict_proba",
        "predict_probabilities",
        "class_probabilities",
    ):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        try:
            return method(X, **posterior_kwargs)
        except TypeError:
            return method(X)

    posterior = model.posterior(X, **posterior_kwargs)
    mean, _ = extract_mean_and_variance(posterior)
    return mean


def _ordinal_scores(model: Any, X: Any, posterior_kwargs: dict[str, Any]) -> Any:
    for method_name in (
        "expected_utility",
        "ordinal_expected_utility",
        "predict_scores",
    ):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        try:
            return method(X, **posterior_kwargs)
        except TypeError:
            return method(X)
    posterior = model.posterior(X, **posterior_kwargs)
    mean, _ = extract_mean_and_variance(posterior)
    return mean


def _ordinal_probabilities(model: Any, X: Any, posterior_kwargs: dict[str, Any]) -> Any | None:
    for method_name in (
        "class_probs",
        "ordinal_probs",
        "predict_proba",
        "predict_probabilities",
        "class_probabilities",
    ):
        method = getattr(model, method_name, None)
        if method is None:
            continue
        try:
            return method(X, **posterior_kwargs)
        except TypeError:
            return method(X)
    return None


def _single_output_labels(
    *,
    task_type: str,
    model: Any,
    X: Any,
    posterior_kwargs: dict[str, Any],
    binary_threshold: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    if task_type == "binary":
        posterior = model.posterior(X, **posterior_kwargs)
        mean, _ = extract_mean_and_variance(posterior)
        probabilities = mean
        if probabilities.ndim > 1 and probabilities.shape[-1] == 1:
            probabilities = probabilities.squeeze(-1)
        indices = (probabilities >= float(binary_threshold)).to(dtype=torch.long)
        return indices.detach().cpu().numpy(), probabilities.detach().cpu().numpy()

    if task_type == "multiclass":
        probabilities = _multiclass_probabilities(model, X, posterior_kwargs)
        if isinstance(probabilities, Sequence) and not hasattr(probabilities, "shape"):
            probabilities = probabilities[0]
        tensor = torch.as_tensor(probabilities)
        if tensor.ndim == 1:
            indices = tensor.to(dtype=torch.long)
            return indices.detach().cpu().numpy(), None
        indices = tensor.argmax(dim=-1)
        return indices.detach().cpu().numpy(), tensor.detach().cpu().numpy()

    if task_type == "ordinal":
        probabilities = _ordinal_probabilities(model, X, posterior_kwargs)
        if probabilities is not None:
            if isinstance(probabilities, Sequence) and not hasattr(probabilities, "shape"):
                probabilities = probabilities[0]
            tensor = torch.as_tensor(probabilities)
            if tensor.ndim > 1 and tensor.shape[-1] > 1:
                indices = tensor.argmax(dim=-1)
                return indices.detach().cpu().numpy(), tensor.detach().cpu().numpy()

        score = torch.as_tensor(_ordinal_scores(model, X, posterior_kwargs))
        if score.ndim > 1 and score.shape[-1] == 1:
            score = score.squeeze(-1)
        indices = score.round().to(dtype=torch.long).clamp_min(0)
        return indices.detach().cpu().numpy(), None

    raise ValueError(f"Unsupported classification task_type={task_type!r}.")


def _wrapper_models(model: Any) -> list[Any]:
    models = getattr(model, "models", None)
    if models is not None:
        return list(models)
    return []


def _multi_output_classification_labels(
    optimizer: Any,
    X: Any,
    posterior_kwargs: dict[str, Any],
    binary_threshold: float,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    bundle = optimizer.bo.bundle
    model = bundle.model
    output_configs = _classification_output_configs(optimizer)
    wrapper_models = _wrapper_models(model)

    if output_configs and wrapper_models and len(output_configs) == len(wrapper_models):
        labels: list[np.ndarray] = []
        probabilities: list[np.ndarray | None] = []
        for output_config, submodel in zip(output_configs, wrapper_models, strict=True):
            if isinstance(output_config, Mapping):
                task_type = str(output_config.get("task_type", "regression"))
            else:
                task_type = str(getattr(output_config, "task_type", "regression"))
            if task_type not in {"binary", "multiclass", "ordinal"}:
                labels.append(np.asarray([], dtype=int))
                probabilities.append(None)
                continue
            output_labels, output_probabilities = _single_output_labels(
                task_type=task_type,
                model=submodel,
                X=X,
                posterior_kwargs=posterior_kwargs,
                binary_threshold=binary_threshold,
            )
            labels.append(output_labels)
            probabilities.append(output_probabilities)
        return labels, probabilities

    task_type = _task_type(optimizer)
    if task_type not in {"binary", "multiclass", "ordinal"}:
        return [], []

    if task_type == "multiclass":
        values = _multiclass_probabilities(model, X, posterior_kwargs)
        if isinstance(values, Sequence) and not hasattr(values, "shape"):
            arrays = [np.asarray(torch_value.detach().cpu()) for torch_value in values]
            return [array.argmax(axis=-1) for array in arrays], arrays
        tensor = torch.as_tensor(values)
        if tensor.ndim >= 3:
            labels = [tensor[..., index, :].argmax(dim=-1).detach().cpu().numpy() for index in range(tensor.shape[-2])]
            probs = [tensor[..., index, :].detach().cpu().numpy() for index in range(tensor.shape[-2])]
            return labels, probs

    posterior = model.posterior(X, **posterior_kwargs)
    mean, _ = extract_mean_and_variance(posterior)
    tensor = torch.as_tensor(mean)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    labels: list[np.ndarray] = []
    for index in range(tensor.shape[-1]):
        values = tensor[..., index]
        if task_type == "binary":
            labels.append((values >= binary_threshold).to(dtype=torch.long).detach().cpu().numpy())
        else:
            labels.append(values.round().to(dtype=torch.long).clamp_min(0).detach().cpu().numpy())
    return labels, [None] * len(labels)


def classification_prediction_dataframe(
    optimizer: Any,
    X: Any,
    *,
    posterior_kwargs: dict[str, Any] | None = None,
    binary_threshold: float = 0.5,
) -> Any:
    """Return decoded class labels for classification / ordinal outputs."""

    import pandas as pd

    posterior_kwargs = dict(posterior_kwargs or {})
    category_maps = _category_maps(optimizer)
    labels, _ = _multi_output_classification_labels(
        optimizer,
        X,
        posterior_kwargs,
        float(binary_threshold),
    )
    if not labels:
        return pd.DataFrame(index=range(int(X.shape[0])))

    target_names = _target_names(optimizer, len(labels))
    output_configs = _classification_output_configs(optimizer)
    task_type = _task_type(optimizer)
    columns: dict[str, Any] = {}
    for index, encoded in enumerate(labels):
        if encoded.size == 0:
            continue
        output_task = task_type
        if output_configs and index < len(output_configs):
            output_config = output_configs[index]
            output_task = (
                str(output_config.get("task_type", "regression"))
                if isinstance(output_config, Mapping)
                else str(getattr(output_config, "task_type", "regression"))
            )
        if output_task not in {"binary", "multiclass", "ordinal"}:
            continue
        output_name = target_names[index]
        inverse = _inverse_category_map(category_maps, output_name)
        decoded = _decode_indices(encoded, inverse)
        columns[f"{output_name}_label"] = decoded

    return pd.DataFrame(columns)


__all__ = [
    "_DATAFRAME_RETURN_TYPES",
    "_LABEL_RETURN_TYPES",
    "_prediction_tensor_and_index",
    "classification_prediction_dataframe",
]
