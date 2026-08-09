"""Raw-space permutation importance orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from .config import FeatureGroup, FeatureImportanceConfig
from .diagnostics import extract_model_diagnostics
from .result_types import (
    FeatureImportanceEntry,
    FeatureImportanceResult,
    ImportanceMethodResult,
    ImportanceSummary,
    OutputFeatureImportanceResult,
)


def _as_prediction(value: Any) -> Tensor:
    """Convert a public prediction value to a detached two-dimensional tensor."""

    value = torch.as_tensor(value).detach()
    return value.unsqueeze(-1) if value.ndim == 1 else value


def _aggregate_expanded_rows(value: Tensor, X: Tensor, *, kind: str) -> Tensor:
    """Reduce one-to-many evaluation rows back to the nominal input rows.

    BoTorch input transforms such as ``InputPerturbation`` evaluate each nominal
    row at multiple perturbed locations. Permutation-importance metrics are
    defined against the nominal target rows, so predictions must be averaged
    across the expansion before scoring.
    """

    n_rows = int(X.shape[0])
    if n_rows <= 0 or value.shape[0] == n_rows:
        return value
    if value.shape[0] % n_rows != 0:
        raise ValueError(
            f"{kind} rows must match feature-importance inputs or be an integer "
            "one-to-many expansion; "
            f"got values={value.shape[0]}, inputs={n_rows}."
        )
    expansion = value.shape[0] // n_rows
    return value.reshape(n_rows, expansion, *value.shape[1:]).mean(dim=1)


def _select_class_probability_output(
    value: Any,
    X: Tensor,
    *,
    output: int,
) -> Tensor:
    """Normalize class probabilities to ``[n, num_classes]`` for one output."""

    probability = torch.as_tensor(value).detach()
    if probability.ndim >= X.ndim + 1:
        if probability.shape[-2] == 1:
            probability = probability.squeeze(-2)
        else:
            if output >= probability.shape[-2]:
                raise IndexError(
                    f"output={output} is out of range for class probabilities "
                    f"with shape={tuple(probability.shape)}."
                )
            probability = probability[..., output, :]
    probability = _aggregate_expanded_rows(
        probability,
        X,
        kind="Class probability",
    )
    if probability.ndim != 2 or probability.shape[0] != X.shape[0]:
        raise ValueError(
            "Class probabilities must have shape [n, num_classes] for feature "
            f"importance. Got X.shape={tuple(X.shape)}, "
            f"probability.shape={tuple(probability.shape)}."
        )
    if probability.shape[-1] < 2:
        raise ValueError(
            "Multiclass feature importance requires one probability column per "
            f"class. Got shape={tuple(probability.shape)}."
        )
    return probability


def _class_probability_prediction(
    source: Any,
    X: Tensor,
    *,
    task: str,
    output: int,
) -> Tensor | None:
    """Return output-specific class probabilities when the model exposes them."""

    if task not in {"multiclass", "ordinal"}:
        return None
    model = getattr(source, "model", None) or source

    class_probs_list = getattr(model, "class_probs_list", None)
    if callable(class_probs_list):
        try:
            values = class_probs_list(X, output_indices=[output])
        except TypeError:
            try:
                values = class_probs_list(X=X, output_indices=[output])
            except TypeError:
                values = class_probs_list(X)
        if not isinstance(values, (list, tuple)) or not values:
            raise TypeError(
                "class_probs_list must return a non-empty list or tuple of tensors."
            )
        selected = values[0] if len(values) == 1 else values[output]
        return _select_class_probability_output(selected, X, output=0)

    class_probs = getattr(model, "class_probs", None)
    if callable(class_probs):
        try:
            value = class_probs(X=X)
        except TypeError:
            value = class_probs(X)
        return _select_class_probability_output(value, X, output=output)
    return None


def _predict(source: Any, X: Tensor, *, task: str, output: int) -> Tensor:
    """Obtain an output-appropriate prediction through public raw-input APIs."""

    class_probability = _class_probability_prediction(
        source,
        X,
        task=task,
        output=output,
    )
    if class_probability is not None:
        return class_probability

    if hasattr(source, "predict"):
        try:
            result = source.predict(X, return_result=True)
        except TypeError:
            result = source.predict(X)
        value = getattr(result, "mean", result)
    else:
        probability = getattr(source, "predict_proba", None)
        if callable(probability):
            value = probability(X)
        else:
            posterior = source.posterior(X)
            value = posterior.mean
    return _aggregate_expanded_rows(
        _as_prediction(value),
        X,
        kind="Prediction",
    )


def _multiclass_target_indices(target: Tensor, num_classes: int) -> Tensor:
    """Validate encoded class labels before indexing a probability matrix."""

    indices = target.long()
    if not torch.equal(target, indices.to(dtype=target.dtype)):
        raise ValueError("Multiclass targets must be integer-encoded class indices.")
    if indices.numel() and (
        int(indices.min().item()) < 0 or int(indices.max().item()) >= num_classes
    ):
        minimum = int(indices.min().item())
        maximum = int(indices.max().item())
        raise ValueError(
            "Multiclass prediction and target class counts do not match: "
            f"prediction has {num_classes} probability columns, while target "
            f"indices range from {minimum} to {maximum}."
        )
    return indices


def _score(
    y: Tensor, prediction: Tensor, task: str, scoring: str | Callable[..., Any], output: int
) -> tuple[float, str, str]:
    """Score one output using the task's stable default metric."""
    target = y[:, output] if y.ndim > 1 else y
    pred = prediction
    if callable(scoring):
        value = scoring(target, pred, task_type=task, output_name=f"output_{output}", metadata={})
        return float(value), getattr(scoring, "__name__", "callable"), "minimize"
    name = scoring.lower().replace(" ", "_") if scoring != "auto" else "auto"
    classification = task in {"binary", "classification", "multiclass", "ordinal"}
    if task == "ordinal":
        if pred.shape[-1] > 1:
            probability = pred.clamp_min(1e-7)
            probability = probability / probability.sum(-1, keepdim=True)
            ranks = torch.arange(pred.shape[-1], device=pred.device, dtype=probability.dtype)
            ordinal_prediction = (probability * ranks).sum(-1)
            metric_name = "expected_rank_mae"
        else:
            ordinal_prediction = pred[:, 0]
            metric_name = "class_index_mae"
        if name in {"auto", "class_index_mae", "expected_rank_mae", "mae"}:
            return float((target - ordinal_prediction).abs().mean()), metric_name, "minimize"
        if name == "accuracy":
            return (
                float((ordinal_prediction.round() == target).float().mean()),
                "accuracy",
                "maximize",
            )
        raise ValueError(f"Scoring {scoring!r} is not valid for task {task!r}.")
    if task == "binary" or (task == "classification" and pred.shape[-1] == 1):
        probability = pred[:, 0].clamp(1e-7, 1 - 1e-7)
        if name in {"auto", "log_loss", "binary_log_loss"}:
            return (
                float(-(target * probability.log() + (1 - target) * (1 - probability).log()).mean()),
                "log_loss",
                "minimize",
            )
        labels = (probability >= 0.5).to(target.dtype)
    elif task == "multiclass" or (classification and pred.shape[-1] > 1):
        probability = pred.clamp_min(1e-7)
        probability = probability / probability.sum(-1, keepdim=True)
        class_indices = _multiclass_target_indices(target, probability.shape[-1])
        if name in {"auto", "log_loss", "multiclass_log_loss", "ordinal_log_loss"}:
            return (
                float(
                    -probability[
                        torch.arange(len(target), device=target.device),
                        class_indices,
                    ].log().mean()
                ),
                "multiclass_log_loss",
                "minimize",
            )
        labels = probability.argmax(-1).to(target.dtype)
    else:
        column = pred[:, output] if pred.shape[-1] > output else pred[:, 0]
        if name in {"r2"}:
            denom = ((target - target.mean()) ** 2).sum()
            value = 1 - ((target - column) ** 2).sum() / denom
            return float(value), "r2", "maximize"
        if name in {"mae", "class_index_mae", "expected_rank_mae"} or task == "ordinal":
            return float((target - column).abs().mean()), "class_index_mae" if task == "ordinal" else "mae", "minimize"
        return float(torch.sqrt(torch.mean((target - column) ** 2))), "rmse", "minimize"
    if name == "accuracy":
        return float((labels == target).float().mean()), "accuracy", "maximize"
    raise ValueError(f"Scoring {scoring!r} is not valid for task {task!r}.")


def _groups(d: int, names: tuple[str, ...], config: FeatureImportanceConfig) -> list[FeatureGroup]:
    """Validate and return individual features plus optional joint groups."""
    result = [FeatureGroup(names[i], (i,), "categorical" if False else "design") for i in range(d)]
    occupied: set[int] = set()
    for group in config.feature_groups or ():
        if any(i < 0 or i >= d for i in group.indices):
            raise ValueError(f"Feature group {group.name!r} contains an out-of-range index.")
        overlap = occupied.intersection(group.indices)
        if overlap:
            raise ValueError(f"Feature groups must not overlap; repeated indices: {sorted(overlap)}")
        occupied.update(group.indices)
        result.append(group)
    return result


def compute_feature_importance(
    *,
    model: Any | None = None,
    predictor: Any | None = None,
    X: Any,
    y: Any,
    task_type: str | Sequence[str] = "regression",
    feature_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    cat_dims: Sequence[int] | None = None,
    config: FeatureImportanceConfig | None = None,
    training_data: bool = False,
) -> FeatureImportanceResult:
    """Compute reproducible raw-space permutation importance.

    Args:
        model: Fitted model exposing ``posterior`` or ``predict_proba``.
        predictor: Fitted object exposing raw-space ``predict`` (preferred).
        X: Raw evaluation inputs of shape ``[n, d]``.
        y: Evaluation targets.
        task_type: One task name per output, or one shared task name.
        feature_names: Raw column names.
        output_names: Optional output names.
        cat_dims: Raw categorical columns; their observed values are permuted intact.
        config: Inspection settings.
        training_data: Whether evaluation uses training observations.

    Returns:
        Output-oriented importance and lightweight diagnostics.
    """
    config = config or FeatureImportanceConfig()
    source = predictor if predictor is not None else model
    if source is None:
        raise ValueError("Exactly one usable model or predictor must be supplied.")
    X_tensor, y_tensor = torch.as_tensor(X), torch.as_tensor(y)
    if X_tensor.ndim != 2 or y_tensor.shape[0] != X_tensor.shape[0]:
        raise ValueError("X must be two-dimensional and X and y must have equal observation counts.")
    y_tensor = y_tensor.unsqueeze(-1) if y_tensor.ndim == 1 else y_tensor
    names = tuple(feature_names or (f"feature_{i}" for i in range(X_tensor.shape[1])))
    if len(names) != X_tensor.shape[1]:
        raise ValueError("feature_names length must equal the raw input dimension.")
    tasks = [task_type] * y_tensor.shape[1] if isinstance(task_type, str) else list(task_type)
    if len(tasks) != y_tensor.shape[1]:
        raise ValueError("task_type metadata must match the number of y columns.")
    if config.compute_classwise_importance and not all(t in {"binary", "classification", "multiclass"} for t in tasks):
        raise ValueError("compute_classwise_importance is valid only for classification tasks.")
    out_names = tuple(output_names or (f"output_{i}" for i in range(y_tensor.shape[1])))
    if len(out_names) != y_tensor.shape[1]:
        raise ValueError("output_names length must match the number of y columns.")
    groups = _groups(X_tensor.shape[1], names, config)
    warning_list = ["Feature importance was evaluated on training data and may be optimistic."] if training_data else []
    diagnostic_model = model or getattr(predictor, "model", None)
    diagnostics, diagnostic_warnings = (
        extract_model_diagnostics(
            diagnostic_model,
            methods=tuple(dict.fromkeys(config.diagnostic_methods)),
            feature_names=names,
            cat_dims=tuple(cat_dims or ()),
        )
        if diagnostic_model is not None
        else ({}, [])
    )
    if diagnostic_warnings and config.unsupported_method_policy == "raise":
        raise RuntimeError(" ".join(diagnostic_warnings))
    if config.unsupported_method_policy == "skip":
        diagnostic_warnings = []
    warning_list.extend(diagnostic_warnings)
    outputs: dict[str, OutputFeatureImportanceResult] = {}
    for output, (output_name, task) in enumerate(zip(out_names, tasks, strict=True)):
        task_name = str(task)
        with torch.no_grad():
            baseline_prediction = _predict(
                source,
                X_tensor,
                task=task_name,
                output=output,
            )
        baseline, metric_name, direction = _score(
            y_tensor,
            baseline_prediction,
            task_name,
            config.scoring,
            output,
        )
        if config.scoring_direction != "auto":
            direction = config.scoring_direction
        entries: dict[str, FeatureImportanceEntry] = {}
        generator = torch.Generator(device=X_tensor.device)
        if config.random_state is not None:
            generator.manual_seed(config.random_state)
        for group in groups:
            values = []
            for _ in range(config.n_repeats):
                permutation = torch.randperm(len(X_tensor), generator=generator, device=X_tensor.device)
                permuted = X_tensor.clone()
                idx = list(group.indices)
                permuted[:, idx] = X_tensor[permutation][:, idx]
                with torch.no_grad():
                    prediction = _predict(
                        source,
                        permuted,
                        task=task_name,
                        output=output,
                    )
                score, _, _ = _score(
                    y_tensor,
                    prediction,
                    task_name,
                    config.scoring,
                    output,
                )
                values.append(score - baseline if direction == "minimize" else baseline - score)
            tensor = torch.tensor(values, dtype=torch.float64)
            summary = ImportanceSummary(
                tensor if config.return_per_repeat_values else None,
                float(tensor.mean()),
                float(tensor.std(unbiased=False)),
                float(tensor.min()),
                float(tensor.max()),
                float(tensor.median()),
            )
            role = (
                group.role
                if len(group.indices) > 1
                else (config.feature_roles or {}).get(
                    group.indices[0], "categorical" if group.indices[0] in (cat_dims or ()) else "design"
                )
            )
            entries[group.name] = FeatureImportanceEntry(
                group.name,
                group.indices,
                tuple(names[i] for i in group.indices),
                "group"
                if len(group.indices) > 1
                else ("categorical" if group.indices[0] in (cat_dims or ()) else "continuous"),
                role,
                summary,
                baseline,
                metric_name,
                direction,
                {
                    "permutation_strategy": "joint_row_permutation" if len(group.indices) > 1 else "row_permutation",
                    "evaluation_space": "raw",
                },
            )
        ordered = sorted(entries.values(), key=lambda item: item.importance.mean, reverse=True)
        positive_total = sum(max(item.importance.mean, 0.0) for item in ordered)
        for rank, entry in enumerate(ordered, 1):
            entry.importance.rank = rank
            if config.normalize_importance:
                entry.importance.normalized_mean = (
                    max(entry.importance.mean, 0.0) / positive_total if positive_total else 0.0
                )
        method = ImportanceMethodResult(
            "permutation",
            entries,
            {metric_name: baseline},
            metadata={"prediction_calls": 1 + len(groups) * config.n_repeats},
        )
        outputs[output_name] = OutputFeatureImportanceResult(
            output_name,
            task_name,
            {"permutation": method},
            model_diagnostics=diagnostics.copy(),
            warnings=diagnostic_warnings.copy(),
        )
    return FeatureImportanceResult(
        outputs,
        tuple(config.predictive_methods),
        tuple(dict.fromkeys(config.diagnostic_methods)),
        "raw",
        config.n_repeats,
        names,
        warning_list,
        {"random_state": config.random_state},
    )
