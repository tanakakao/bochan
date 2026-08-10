from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import wraps
from typing import Any

import torch
from torch import Tensor

from .converter import dataframe_to_tensors
from .optimizer import TabularBayesianOptimizer

_CLASSIFICATION_TASK_TYPES = {"binary", "multiclass", "ordinal"}
_LABEL_RETURN_TYPES = {
    "label",
    "labels",
    "predicted_label",
    "predicted_labels",
    "classification_labels",
}
_DATAFRAME_RETURN_TYPES = {
    "dataframe",
    "df",
    "mean_variance_dataframe",
    "mean_variance_df",
}
_ORIGINAL_PREDICT_ATTR = "_bochan_predict_before_tabular_labels"


def _call_with_fallbacks(calls: Sequence[Callable[[], Any]]) -> Any:
    """Call model accessors while tolerating optional keyword differences."""

    last_error: TypeError | None = None
    for call in calls:
        try:
            return call()
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No prediction accessor call was provided.")


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


def _output_task_types(optimizer: TabularBayesianOptimizer) -> list[str]:
    """Resolve one task type for every fitted tabular target column."""

    if optimizer.dataset is None:
        return []
    target_names = list(optimizer.dataset.target_names)
    model = getattr(optimizer.bo, "model", None)
    if model is None:
        bundle = getattr(optimizer.bo, "bundle", None)
        model = getattr(bundle, "model", None)

    task_types = getattr(model, "task_types", None)
    if callable(task_types):
        task_types = task_types()
    if task_types is not None:
        values = [str(value).lower() for value in task_types]
        if len(values) == len(target_names):
            return values

    bundle = getattr(optimizer.bo, "bundle", None)
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    metadata_task_types = metadata.get("output_task_types")
    if metadata_task_types is not None:
        values = [str(value).lower() for value in metadata_task_types]
        if len(values) == len(target_names):
            return values

    task_type = str(getattr(bundle, "task_type", optimizer.model_config.task_type)).lower()
    if task_type != "hybrid":
        return [task_type] * len(target_names)

    multi_output_config = getattr(optimizer.model_config, "multi_output_config", None)
    output_configs = getattr(multi_output_config, "output_configs", None)
    if output_configs is not None and len(output_configs) == len(target_names):
        resolved = []
        for item in output_configs:
            if isinstance(item, Mapping):
                resolved.append(str(item.get("task_type", "regression")).lower())
            elif isinstance(item, str):
                resolved.append(str(item).lower())
            else:
                resolved.append(str(getattr(item, "task_type", "regression")).lower())
        return resolved

    return ["regression"] * len(target_names)


def _select_output_probability_tensor(
    value: Any,
    *,
    output_index: int,
    num_outputs: int,
) -> Tensor:
    """Select one output while preserving the final class dimension."""

    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if num_outputs > 1 and tensor.ndim >= 3 and tensor.shape[-2] == num_outputs:
        tensor = tensor[..., output_index, :]
    return tensor


def _reduce_probabilities_to_rows(
    probabilities: Tensor,
    *,
    n_rows: int,
) -> Tensor:
    """Reduce sample, batch, or input-perturbation axes to ``n_rows x classes``."""

    probabilities = probabilities.detach()
    while probabilities.ndim > 2:
        if probabilities.shape[-2] == n_rows:
            probabilities = probabilities.reshape(-1, n_rows, probabilities.shape[-1]).mean(dim=0)
        elif probabilities.shape[0] == n_rows:
            probabilities = probabilities.reshape(n_rows, -1, probabilities.shape[-1]).mean(dim=1)
        else:
            probabilities = probabilities.mean(dim=0)

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)
    if probabilities.ndim != 2:
        raise RuntimeError(
            "Class probabilities could not be reduced to a two-dimensional table. "
            f"Got shape={tuple(probabilities.shape)}."
        )

    if probabilities.shape[0] != n_rows:
        if probabilities.shape[0] % n_rows != 0:
            raise RuntimeError(
                "Class probabilities could not be aligned with prediction rows. "
                f"probabilities.shape={tuple(probabilities.shape)}, n_rows={n_rows}."
            )
        probabilities = probabilities.reshape(n_rows, -1, probabilities.shape[-1]).mean(dim=1)
    return probabilities


def _class_probabilities_for_output(
    optimizer: TabularBayesianOptimizer,
    X: Tensor,
    *,
    output_index: int,
    task_type: str,
    posterior_kwargs: Mapping[str, Any],
    n_rows: int,
) -> Tensor | None:
    """Return class probabilities for one binary, ordinal, or multiclass output."""

    model = getattr(optimizer.bo, "model", None)
    if model is None:
        bundle = getattr(optimizer.bo, "bundle", None)
        model = getattr(bundle, "model", None)
    if model is None:
        return None

    num_outputs = len(optimizer.dataset.target_names) if optimizer.dataset is not None else 1
    class_probs_list = getattr(model, "class_probs_list", None)
    if callable(class_probs_list):
        values = _call_output_accessor(
            class_probs_list,
            X,
            output_index=output_index,
            kwargs=posterior_kwargs,
        )
        if len(values) != 1:
            raise RuntimeError(
                "class_probs_list must return exactly one tensor when one output is requested. "
                f"Got {len(values)} values for output_index={output_index}."
            )
        probabilities = _select_output_probability_tensor(
            values[0],
            output_index=output_index,
            num_outputs=num_outputs,
        )
        return _reduce_probabilities_to_rows(probabilities, n_rows=n_rows)

    class_probs = getattr(model, "class_probs", None)
    if callable(class_probs):
        probabilities = _call_single_accessor(class_probs, X, kwargs=posterior_kwargs)
        probabilities = _select_output_probability_tensor(
            probabilities,
            output_index=output_index,
            num_outputs=num_outputs,
        )
        return _reduce_probabilities_to_rows(probabilities, n_rows=n_rows)

    probability_posterior = getattr(model, "probability_posterior", None)
    posterior = None
    if callable(probability_posterior):
        try:
            posterior = _call_output_accessor(
                probability_posterior,
                X,
                output_index=output_index,
                kwargs=posterior_kwargs,
            )
        except (TypeError, NotImplementedError):
            posterior = _call_single_accessor(
                probability_posterior,
                X,
                kwargs=posterior_kwargs,
            )
    elif task_type in _CLASSIFICATION_TASK_TYPES:
        posterior_fn = getattr(model, "posterior", None)
        if callable(posterior_fn):
            try:
                posterior = _call_output_accessor(
                    posterior_fn,
                    X,
                    output_index=output_index,
                    kwargs=posterior_kwargs,
                )
            except (TypeError, NotImplementedError):
                posterior = _call_single_accessor(
                    posterior_fn,
                    X,
                    kwargs=posterior_kwargs,
                )

    mean = getattr(posterior, "mean", None)
    if mean is None:
        return None
    probabilities = _select_output_probability_tensor(
        mean,
        output_index=output_index,
        num_outputs=num_outputs,
    )
    probabilities = _reduce_probabilities_to_rows(probabilities, n_rows=n_rows)
    if task_type == "binary" and probabilities.shape[-1] == 1:
        p1 = probabilities[..., 0].clamp(0.0, 1.0)
        probabilities = torch.stack([1.0 - p1, p1], dim=-1)
    return probabilities


def _predicted_classes_from_model(
    optimizer: TabularBayesianOptimizer,
    X: Tensor,
    *,
    output_index: int,
    posterior_kwargs: Mapping[str, Any],
    n_rows: int,
) -> Tensor | None:
    """Fallback to a model's discrete predict_class accessor."""

    model = getattr(optimizer.bo, "model", None)
    if model is None:
        bundle = getattr(optimizer.bo, "bundle", None)
        model = getattr(bundle, "model", None)
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
            raise RuntimeError(
                "Predicted classes could not be aligned with prediction rows. "
                f"predicted.shape={tuple(tensor.shape)}, n_rows={n_rows}."
            )
        tensor = tensor.reshape(n_rows, -1).to(torch.double).mean(dim=1).round()
    return tensor.to(torch.long)


def _inverse_target_map(
    optimizer: TabularBayesianOptimizer,
    target_name: Any,
) -> Mapping[int, Any] | None:
    if optimizer.dataset is None:
        return None
    inverse_maps = dict(getattr(optimizer.dataset, "inverse_target_category_maps", None) or {})
    mapping = inverse_maps.get(target_name)
    if mapping is None:
        mapping = inverse_maps.get(str(target_name))
    return mapping


def classification_prediction_dataframe(
    optimizer: TabularBayesianOptimizer,
    X: Tensor,
    *,
    posterior_kwargs: Mapping[str, Any] | None = None,
    binary_threshold: float = 0.5,
):
    """Build decoded prediction-label columns for all classification outputs."""

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

    for output_index, (target_name, task_type) in enumerate(zip(target_names, task_types, strict=True)):
        if task_type not in _CLASSIFICATION_TASK_TYPES:
            continue

        probabilities = _class_probabilities_for_output(
            optimizer,
            X,
            output_index=output_index,
            task_type=task_type,
            posterior_kwargs=posterior_kwargs,
            n_rows=n_rows,
        )
        predicted_probability: Tensor | None = None
        if probabilities is not None:
            probabilities = probabilities.clamp_min(0.0)
            if task_type == "binary":
                if probabilities.shape[-1] == 1:
                    p1 = probabilities[..., 0].clamp(0.0, 1.0)
                else:
                    p1 = probabilities[..., 1].clamp(0.0, 1.0)
                predicted_class = (p1 >= threshold).to(torch.long)
                predicted_probability = torch.where(predicted_class == 1, p1, 1.0 - p1)
            else:
                denominator = probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                probabilities = probabilities / denominator
                predicted_probability, predicted_class = probabilities.max(dim=-1)
        else:
            predicted_class = _predicted_classes_from_model(
                optimizer,
                X,
                output_index=output_index,
                posterior_kwargs=posterior_kwargs,
                n_rows=n_rows,
            )
            if predicted_class is None:
                raise AttributeError(
                    f"Could not obtain class probabilities or predicted classes for output {target_name!r}."
                )

        class_indices = predicted_class.detach().cpu().to(torch.long).tolist()
        inverse_map = _inverse_target_map(optimizer, target_name)
        decoded_labels = [
            inverse_map.get(int(class_index), int(class_index)) if inverse_map is not None else int(class_index)
            for class_index in class_indices
        ]
        prefix = str(target_name)
        columns[f"{prefix}_predicted_class_index"] = class_indices
        columns[f"{prefix}_predicted_label"] = decoded_labels
        if predicted_probability is not None:
            columns[f"{prefix}_predicted_probability"] = predicted_probability.detach().cpu().tolist()

    return pd.DataFrame(columns)


def _prediction_tensor_and_index(
    optimizer: TabularBayesianOptimizer,
    data: Any,
) -> tuple[Tensor, Any | None]:
    """Convert prediction input exactly as the base tabular predict method does."""

    if optimizer.dataset is None:
        raise RuntimeError("No fitted tabular dataset found. Call fit() first.")
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and isinstance(data, pd.DataFrame):
        config = replace(
            optimizer.data_config,
            target_cols=None,
            input_cols=optimizer.dataset.feature_names,
        )
        return dataframe_to_tensors(data, config).X, data.index

    X = data if torch.is_tensor(data) else torch.as_tensor(data)
    return X, None


def apply_tabular_prediction_labels() -> None:
    """Patch tabular ``predict`` to append decoded classification labels."""

    cls = TabularBayesianOptimizer
    if hasattr(cls, _ORIGINAL_PREDICT_ATTR):
        return

    original_predict = cls.predict
    setattr(cls, _ORIGINAL_PREDICT_ATTR, original_predict)

    @wraps(original_predict)
    def _predict(
        self,
        data: Any,
        *,
        return_type: str = "dataframe",
        include_input: bool = False,
        return_dataframe_input: bool = False,
        posterior_kwargs: dict[str, Any] | None = None,
        include_prediction_labels: bool = True,
        binary_threshold: float = 0.5,
        **kwargs: Any,
    ) -> Any:
        normalized_return_type = str(return_type).lower()
        labels_only = normalized_return_type in _LABEL_RETURN_TYPES
        dataframe_return = normalized_return_type in _DATAFRAME_RETURN_TYPES

        if not labels_only:
            result = original_predict(
                self,
                data,
                return_type=return_type,
                include_input=include_input,
                return_dataframe_input=return_dataframe_input,
                posterior_kwargs=posterior_kwargs,
                **kwargs,
            )
            if not include_prediction_labels or not dataframe_return:
                return result
            if return_dataframe_input:
                prediction_df, returned_input = result
            else:
                prediction_df = result
                returned_input = None
        else:
            prediction_df = None
            returned_input = data if return_dataframe_input else None

        X, original_index = _prediction_tensor_and_index(self, data)
        labels_df = classification_prediction_dataframe(
            self,
            X,
            posterior_kwargs=posterior_kwargs,
            binary_threshold=binary_threshold,
        )
        if labels_only and labels_df.shape[1] == 0:
            raise ValueError("The fitted optimizer has no binary, multiclass, or ordinal outputs.")
        if original_index is not None:
            labels_df.index = original_index

        if prediction_df is None:
            output_df = labels_df
            if include_input:
                input_df = self._prediction_input_to_dataframe(data, X)
                if original_index is not None:
                    input_df.index = original_index
                output_df = input_df.join(output_df)
        elif labels_df.shape[1] == 0:
            output_df = prediction_df
        else:
            attrs = dict(getattr(prediction_df, "attrs", {}) or {})
            output_df = prediction_df.join(labels_df)
            output_df.attrs.update(attrs)

        if return_dataframe_input:
            return output_df, returned_input
        return output_df

    cls.predict = _predict


__all__ = [
    "apply_tabular_prediction_labels",
    "classification_prediction_dataframe",
]
