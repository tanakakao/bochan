"""Cross-validation support for the regular :mod:`bochan.api` interface."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .configs import FitConfig, ModelConfig, PredictionResult


@dataclass
class CrossValidationConfig:
    """Configure model-independent cross-validation.

    Args:
        splitter: ``"auto"``, ``"kfold"``, ``"stratified"``, ``"loo"``, or
            an sklearn-compatible object exposing ``split``.
        n_splits: Number of folds for K-fold splitters.
        shuffle: Whether K-fold splitters shuffle samples.
        random_state: Seed used by shuffled built-in splitters.
        groups: Optional groups forwarded to a custom splitter.
        stratify_output: Hybrid output name or index used for stratification.
    """

    splitter: str | Any = "auto"
    n_splits: int = 5
    shuffle: bool = True
    random_state: int | None = 0
    groups: Any | None = None
    stratify_output: str | int | None = None
    return_train_predictions: bool = True
    return_fold_predictions: bool = True
    return_models: bool = False
    classification_average: str = "auto"
    classification_threshold: float = 0.5
    positive_class: Any = 1
    zero_division: int = 0
    mape_zero_policy: str = "warn_nan"
    mape_epsilon: float = 1e-8
    error_policy: str = "raise"
    feature_importance_config: Any | None = None
    feature_names: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate settings before any model is trained."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if not 0.0 <= self.classification_threshold <= 1.0:
            raise ValueError("classification_threshold must be between 0 and 1.")
        if self.classification_average not in {"auto", "binary", "micro", "macro", "weighted"}:
            raise ValueError("classification_average must be auto, binary, micro, macro, or weighted.")
        if self.zero_division not in {0, 1}:
            raise ValueError("zero_division must be 0 or 1.")
        if self.mape_zero_policy not in {"warn_nan", "ignore", "clip"}:
            raise ValueError("mape_zero_policy must be warn_nan, ignore, or clip.")
        if self.mape_epsilon <= 0:
            raise ValueError("mape_epsilon must be positive.")
        if self.error_policy not in {"raise", "warn"}:
            raise ValueError("error_policy must be raise or warn.")


@dataclass
class MetricSummary:
    """Summary of one metric over folds; NaNs are ignored when possible."""

    values: Tensor
    mean: float
    std: float
    minimum: float
    maximum: float


@dataclass
class CVPredictionResult:
    """Predictions and two deliberately distinct uncertainty quantities."""

    indices: Tensor
    y_true: Tensor
    y_pred: Tensor
    predictive_mean: Tensor | None = None
    predictive_std: Tensor | None = None
    fold_prediction_std: Tensor | None = None
    probabilities: Tensor | None = None
    variance_kind: str | None = None
    fold_indices: Tensor | None = None
    prediction_count: Tensor | None = None


@dataclass
class CVFoldResult:
    """Metrics and predictions for one output in one fold."""

    fold: int
    train_indices: Tensor
    test_indices: Tensor
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    train_predictions: CVPredictionResult | None
    test_predictions: CVPredictionResult | None
    feature_importance: Any | None = None


@dataclass
class OutputCrossValidationResult:
    """Complete cross-validation result for one model output."""

    task_type: str
    folds: list[CVFoldResult]
    train_metric_summary: dict[str, MetricSummary]
    test_metric_summary: dict[str, MetricSummary]
    aggregated_train_predictions: CVPredictionResult | None
    oof_predictions: CVPredictionResult
    oof_metrics: dict[str, float]


@dataclass
class CrossValidationResult:
    """Cross-validation results using one common shape for single/multi output."""

    outputs: dict[str, OutputCrossValidationResult]
    splitter_name: str
    n_splits: int
    warnings: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    models: list[Any] | None = None
    feature_importance: Any | None = None

    def _single(self) -> OutputCrossValidationResult:
        """Return the sole output or reject an ambiguous convenience access."""
        if len(self.outputs) != 1:
            raise RuntimeError("This convenience property is available only for single-output results.")
        return next(iter(self.outputs.values()))

    @property
    def output(self) -> OutputCrossValidationResult:
        """Return the single output result."""
        return self._single()

    @property
    def test_metric_summary(self) -> dict[str, MetricSummary]:
        """Return validation summaries for a single-output result."""
        return self._single().test_metric_summary

    @property
    def oof_predictions(self) -> CVPredictionResult:
        """Return OOF predictions for a single-output result."""
        return self._single().oof_predictions


def clone_model_config_for_evaluation(config: ModelConfig) -> ModelConfig:
    """Deep-copy model configuration so trainable objects are fold-local.

    Args:
        config: Source model configuration.

    Returns:
        An independent configuration. Classes and functions remain shared under
        Python's normal ``deepcopy`` semantics, while modules and transforms do not.
    """
    return copy.deepcopy(config)


def clone_fit_config_for_evaluation(config: FitConfig | None) -> FitConfig | None:
    """Deep-copy a fit configuration for a fold."""
    return copy.deepcopy(config)


def _as_2d(value: Any) -> Tensor:
    tensor = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor


def _output_layout(config: ModelConfig, train_Y: Tensor) -> tuple[list[str], list[str]]:
    n = _as_2d(train_Y).shape[-1]
    mo = config.multi_output_config
    if mo is None:
        return (
            [f"output_{i}" for i in range(n)],
            [str(config.task_type) for _ in range(n)],
        )
    names = list(mo.output_names or [])
    tasks = list(mo.output_task_types or [])
    output_configs = list(mo.output_configs or [])
    for i in range(n):
        item = output_configs[i] if i < len(output_configs) else None
        if i >= len(names):
            name = getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("name")
            names.append(name or f"output_{i}")
        if i >= len(tasks):
            task = getattr(item, "task_type", None)
            if task is None and isinstance(item, dict):
                task = item.get("task_type")
            tasks.append(str(task or config.task_type))
    return names[:n], tasks[:n]


def _make_splitter(config: CrossValidationConfig, task: str, y: Tensor, warnings: list[str]):
    splitter = config.splitter
    if not isinstance(splitter, str):
        if not callable(getattr(splitter, "split", None)):
            raise ValueError("A custom splitter must expose a callable split method.")
        return splitter, None
    try:
        from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
    except ImportError as exc:
        raise ImportError("Cross-validation requires scikit-learn; install bochan[tabular].") from exc
    key = splitter.lower().replace("-", "_")
    if key in {"loo", "leave_one_out"}:
        return LeaveOneOut(), None
    stratify = key == "stratified" or (key == "auto" and task in {"binary", "multiclass", "ordinal"})
    if key == "auto" and task == "hybrid":
        warnings.append("Hybrid auto splitting fell back to KFold; set stratify_output to stratify explicitly.")
    if key not in {"auto", "kfold", "stratified"}:
        raise ValueError("splitter must be auto, kfold, stratified, loo, or a splitter object.")
    kwargs = {"n_splits": config.n_splits, "shuffle": config.shuffle}
    if config.shuffle:
        kwargs["random_state"] = config.random_state
    return (StratifiedKFold(**kwargs), y.reshape(-1).cpu().numpy()) if stratify else (KFold(**kwargs), None)


def _summary(values: list[float]) -> MetricSummary:
    tensor = torch.tensor(values, dtype=torch.double)
    valid = tensor[~torch.isnan(tensor)]
    if valid.numel() == 0:
        stats = (math.nan,) * 4
    else:
        stats = (float(valid.mean()), float(valid.std(unbiased=False)), float(valid.min()), float(valid.max()))
    return MetricSummary(tensor, *stats)


def _regression_metrics(
    y: Tensor, pred: Tensor, config: CrossValidationConfig, warnings: list[str]
) -> dict[str, float]:
    y, pred = y.double().reshape(-1), pred.double().reshape(-1)
    error = pred - y
    small = y.abs() <= config.mape_epsilon
    if small.any() and config.mape_zero_policy == "warn_nan":
        mape = math.nan
        message = "MAPE is NaN because targets at or below mape_epsilon were encountered."
        if message not in warnings:
            warnings.append(message)
    elif config.mape_zero_policy == "ignore":
        valid = ~small
        mape = float((error[valid].abs() / y[valid].abs()).mean() * 100) if valid.any() else math.nan
    else:
        mape = float((error.abs() / y.abs().clamp_min(config.mape_epsilon)).mean() * 100)
    if y.numel() < 2:
        r2 = math.nan
        message = "Fold R2 is NaN because fewer than two validation samples are available."
        if message not in warnings:
            warnings.append(message)
    else:
        denominator = ((y - y.mean()) ** 2).sum()
        r2 = float(1 - (error.square().sum() / denominator)) if denominator > 0 else math.nan
    return {"rmse": float(error.square().mean().sqrt()), "mae": float(error.abs().mean()), "mape": mape, "r2": r2}


def _classification_metrics(y: Tensor, pred: Tensor, task: str, config: CrossValidationConfig) -> dict[str, float]:
    y, pred = y.reshape(-1), pred.reshape(-1)
    labels = torch.unique(torch.cat((y, pred))).tolist()
    average = config.classification_average
    if average == "auto":
        average = "binary" if task == "binary" else "macro"
    accuracy = float((y == pred).double().mean())
    scores, supports = [], []
    selected = [config.positive_class] if average == "binary" else labels
    for label in selected:
        tp = int(((pred == label) & (y == label)).sum())
        fp = int(((pred == label) & (y != label)).sum())
        fn = int(((pred != label) & (y == label)).sum())
        precision = tp / (tp + fp) if tp + fp else float(config.zero_division)
        recall = tp / (tp + fn) if tp + fn else float(config.zero_division)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else float(config.zero_division)
        scores.append((precision, recall, f1))
        supports.append(int((y == label).sum()))
    if average == "micro":
        precision = recall = f1 = accuracy
    else:
        weights = torch.tensor(supports, dtype=torch.double) if average == "weighted" else torch.ones(len(scores))
        weights = weights / weights.sum().clamp_min(1)
        precision, recall, f1 = [float((torch.tensor([s[i] for s in scores]) * weights).sum()) for i in range(3)]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _prediction_for_output(
    optimizer: Any, X: Tensor, output: int, task: str, config: CrossValidationConfig
) -> CVPredictionResult:
    result: PredictionResult = optimizer.predict(X, return_result=True)
    mean, variance = _as_2d(result.mean), _as_2d(result.variance) if result.variance is not None else None
    model = optimizer.model
    if hasattr(model, "models") and output < len(model.models):
        model = model.models[output]
    probabilities = None
    predict_proba = getattr(model, "predict_proba", None)
    if task in {"multiclass", "ordinal"} and callable(predict_proba):
        probabilities = predict_proba(X)
        if isinstance(probabilities, tuple):
            probabilities = probabilities[0]
        probabilities = _as_2d(probabilities)
        pred = probabilities.argmax(dim=-1)
        predictive_mean = pred.to(probabilities.dtype)
        predictive_std = None
    else:
        column = output if mean.shape[-1] > output else 0
        predictive_mean = mean[..., column].reshape(-1)
        predictive_std = variance[..., column].clamp_min(0).sqrt().reshape(-1) if variance is not None else None
        if task == "binary":
            probabilities = torch.stack((1 - predictive_mean, predictive_mean), dim=-1)
            pred = torch.where(
                predictive_mean >= config.classification_threshold,
                torch.as_tensor(config.positive_class, device=X.device),
                torch.as_tensor(0 if config.positive_class != 0 else 1, device=X.device),
            )
        elif task in {"multiclass", "ordinal"} and mean.shape[-1] > 1:
            probabilities = mean
            pred = mean.argmax(dim=-1)
        else:
            pred = predictive_mean
    empty = torch.empty(0, dtype=torch.long, device=X.device)
    return CVPredictionResult(
        empty,
        empty,
        pred.detach(),
        predictive_mean.detach(),
        predictive_std.detach() if predictive_std is not None else None,
        probabilities=probabilities.detach() if probabilities is not None else None,
        variance_kind=result.variance_kind,
    )


def _aggregate(parts: list[CVPredictionResult], y: Tensor, n: int) -> CVPredictionResult:
    indices = torch.cat([p.indices for p in parts])
    order = torch.argsort(indices)
    unique = torch.unique(indices, sorted=True)
    preds, means, post_stds, fold_stds, counts, truths, probs = [], [], [], [], [], [], []
    has_probs = all(p.probabilities is not None for p in parts)
    all_pred = torch.cat([p.y_pred for p in parts])[order]
    all_mean = torch.cat([p.predictive_mean for p in parts])[order]
    all_std = (
        torch.cat([p.predictive_std for p in parts])[order]
        if all(p.predictive_std is not None for p in parts)
        else None
    )
    all_probs = torch.cat([p.probabilities for p in parts])[order] if has_probs else None
    sorted_idx = indices[order]
    for idx in unique:
        mask = sorted_idx == idx
        vals = all_pred[mask].double()
        preds.append(vals.mean())
        means.append(all_mean[mask].double().mean())
        counts.append(mask.sum())
        fold_stds.append(vals.std(unbiased=False) if mask.sum() > 1 else torch.tensor(math.nan))
        post_stds.append(all_std[mask].mean() if all_std is not None else torch.tensor(math.nan))
        truths.append(y[int(idx)].reshape(()))
        if all_probs is not None:
            probs.append(all_probs[mask].mean(0))
    return CVPredictionResult(
        unique,
        torch.stack(truths),
        torch.stack(preds),
        torch.stack(means),
        torch.stack(post_stds) if all_std is not None else None,
        torch.stack(fold_stds),
        torch.stack(probs) if probs else None,
        parts[0].variance_kind,
        torch.cat([p.fold_indices for p in parts])[order] if all(p.fold_indices is not None for p in parts) else None,
        torch.stack(counts),
    )


def cross_validate_optimizer(
    optimizer: Any,
    train_X: Any,
    train_Y: Any,
    *,
    model_config: ModelConfig | None = None,
    fit_config: FitConfig | None = None,
    cv_config: CrossValidationConfig | None = None,
) -> CrossValidationResult:
    """Run cross-validation without mutating the supplied optimizer.

    Args:
        optimizer: Calling :class:`BayesianOptimizer` instance.
        train_X: Full input tensor.
        train_Y: Full target tensor.
        model_config: Optional per-call model configuration override.
        fit_config: Optional per-call fit configuration override.
        cv_config: Cross-validation settings.

    Returns:
        Output-oriented metrics, predictions, warnings, and optional models.
    """
    config = cv_config or CrossValidationConfig()
    X, Y = torch.as_tensor(train_X), _as_2d(train_Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of rows.")
    base_model = model_config or optimizer.model_config
    base_fit = fit_config if fit_config is not None else optimizer.fit_config
    names, tasks = _output_layout(base_model, Y)
    warnings: list[str] = []
    split_task = str(base_model.task_type)
    strat_y = Y[:, 0]
    if split_task == "hybrid" and config.stratify_output is not None:
        idx = (
            names.index(config.stratify_output)
            if isinstance(config.stratify_output, str)
            else int(config.stratify_output)
        )
        strat_y, split_task = Y[:, idx], tasks[idx]
    splitter, auto_y = _make_splitter(config, split_task, strat_y, warnings)
    split_y = auto_y if auto_y is not None else strat_y.cpu().numpy()
    try:
        splits = list(splitter.split(X.cpu().numpy(), split_y, groups=config.groups))
    except TypeError:
        splits = list(splitter.split(X.cpu().numpy(), split_y))
    per_output: dict[str, list[CVFoldResult]] = {name: [] for name in names}
    fold_importances: list[Any] = []
    models = [] if config.return_models else None
    for fold, (train_idx_raw, test_idx_raw) in enumerate(splits):
        train_idx, test_idx = (
            torch.as_tensor(train_idx_raw, dtype=torch.long),
            torch.as_tensor(test_idx_raw, dtype=torch.long),
        )
        fold_optimizer = type(optimizer)(
            model_config=clone_model_config_for_evaluation(base_model),
            fit_config=clone_fit_config_for_evaluation(base_fit),
            bounds=copy.deepcopy(optimizer.bounds),
            model_registry=optimizer.model_registry,
            acquisition_registry=optimizer.acquisition_registry,
        )
        fold_optimizer.fit(X[train_idx], Y[train_idx])
        fold_importance = None
        if config.feature_importance_config is not None:
            importance_config = copy.deepcopy(config.feature_importance_config)
            if importance_config.random_state is not None:
                importance_config.random_state += fold * 104729
            fold_importance = fold_optimizer.feature_importance(
                X=X[test_idx],
                y=Y[test_idx],
                config=importance_config,
                feature_names=config.feature_names,
                output_names=names,
            )
            fold_importance.metadata["cv_fold"] = fold
            fold_importance.metadata["derived_random_state"] = importance_config.random_state
            fold_importances.append(fold_importance)
        if models is not None:
            models.append(fold_optimizer.model)
        for output, (name, task) in enumerate(zip(names, tasks, strict=True)):
            train_pred = _prediction_for_output(fold_optimizer, X[train_idx], output, task, config)
            test_pred = _prediction_for_output(fold_optimizer, X[test_idx], output, task, config)
            for pred_result, idx in ((train_pred, train_idx), (test_pred, test_idx)):
                pred_result.indices = idx
                pred_result.y_true = Y[idx, output].detach()
                pred_result.fold_indices = torch.full((len(idx),), fold, dtype=torch.long)
                pred_result.prediction_count = torch.ones(len(idx), dtype=torch.long)
            metric = _regression_metrics if task in {"regression", "multi_objective"} else _classification_metrics
            train_metrics = (
                metric(train_pred.y_true, train_pred.y_pred, config, warnings)
                if metric is _regression_metrics
                else metric(train_pred.y_true, train_pred.y_pred, task, config)
            )
            test_metrics = (
                metric(test_pred.y_true, test_pred.y_pred, config, warnings)
                if metric is _regression_metrics
                else metric(test_pred.y_true, test_pred.y_pred, task, config)
            )
            per_output[name].append(
                CVFoldResult(
                    fold,
                    train_idx,
                    test_idx,
                    train_metrics,
                    test_metrics,
                    train_pred,
                    test_pred,
                    fold_importance,
                )
            )
    outputs = {}
    for output, (name, task) in enumerate(zip(names, tasks, strict=True)):
        folds = per_output[name]
        test_parts = []
        train_parts = []
        for fold in range(len(splits)):
            stored = per_output[name][fold]
            if stored.test_predictions is not None:
                test_parts.append(stored.test_predictions)
            if stored.train_predictions is not None:
                train_parts.append(stored.train_predictions)
        oof = _aggregate(test_parts, Y[:, output], len(Y))
        aggregated_train = _aggregate(train_parts, Y[:, output], len(Y)) if config.return_train_predictions else None
        metric = _regression_metrics if task in {"regression", "multi_objective"} else _classification_metrics
        oof_metrics = (
            metric(oof.y_true, oof.y_pred, config, warnings)
            if metric is _regression_metrics
            else metric(oof.y_true, oof.y_pred, task, config)
        )
        metric_names = folds[0].test_metrics
        outputs[name] = OutputCrossValidationResult(
            task,
            folds,
            {k: _summary([f.train_metrics[k] for f in folds]) for k in metric_names},
            {k: _summary([f.test_metrics[k] for f in folds]) for k in metric_names},
            aggregated_train,
            oof,
            oof_metrics,
        )
        if not config.return_fold_predictions:
            for fold_result in folds:
                fold_result.train_predictions = None
                fold_result.test_predictions = None
        elif not config.return_train_predictions:
            for fold_result in folds:
                fold_result.train_predictions = None
    cv_importance = _aggregate_feature_importance(fold_importances) if fold_importances else None
    return CrossValidationResult(
        outputs,
        type(splitter).__name__,
        len(splits),
        warnings,
        {"random_state": config.random_state, "return_models": config.return_models},
        models,
        cv_importance,
    )


def _aggregate_feature_importance(folds: list[Any]) -> Any:
    """Aggregate fold means and ranks without aligning latent diagnostics.

    Args:
        folds: Validation-fold ``FeatureImportanceResult`` objects.

    Returns:
        Cross-validated output-oriented importance result.
    """
    from bochan.inspection.result_types import (
        CrossValidatedFeatureImportanceResult,
        CrossValidatedImportanceSummary,
        CrossValidatedMethodResult,
        CrossValidatedOutputImportance,
    )

    outputs = {}
    for output_name, first_output in folds[0].outputs.items():
        methods = {}
        for method_name in first_output.predictive_methods:
            method_folds = [fold.outputs[output_name].predictive_methods[method_name] for fold in folds]
            entries = {}
            for entry_name in method_folds[0].entries:
                fold_entries = [method.entries[entry_name] for method in method_folds]
                values = torch.tensor([entry.importance.mean for entry in fold_entries], dtype=torch.float64)
                ranks = torch.tensor([entry.importance.rank for entry in fold_entries], dtype=torch.float64)
                entries[entry_name] = CrossValidatedImportanceSummary(
                    values,
                    float(values.mean()),
                    float(values.std(unbiased=False)),
                    float(values.min()),
                    float(values.max()),
                    float(values.median()),
                    float(ranks.mean()),
                    float(ranks.std(unbiased=False)),
                    len(values),
                    [entry.importance.std for entry in fold_entries],
                )
            methods[method_name] = CrossValidatedMethodResult(method_name, entries, method_folds)
        outputs[output_name] = CrossValidatedOutputImportance(
            output_name,
            first_output.task_type,
            methods,
            {},
            [fold.outputs[output_name].model_diagnostics for fold in folds],
            warnings=[warning for fold in folds for warning in fold.outputs[output_name].warnings],
        )
    return CrossValidatedFeatureImportanceResult(
        outputs,
        folds[0].feature_names,
        [warning for fold in folds for warning in fold.warnings],
        {"n_folds": len(folds), "pooled_oof_importance": False},
    )
