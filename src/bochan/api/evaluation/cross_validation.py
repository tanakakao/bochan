"""Cross-validation support for the regular :mod:`bochan.api` interface."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ..configs import FitConfig, ModelConfig, PredictionResult


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
    failure_model: OutputCrossValidationResult | None = None
    failure_feature_importance: Any | None = None

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


def _aggregate_expanded_moments(
    mean: Any,
    variance: Any | None,
    *,
    n_rows: int,
) -> tuple[Tensor, Tensor | None]:
    """Reduce one-to-many evaluation transforms back to nominal CV rows.

    Input transforms such as BoTorch ``InputPerturbation`` expand each nominal
    input row into multiple evaluation points. Cross-validation metrics are
    defined on the nominal observations, so the expanded posterior moments must
    be aggregated before comparing predictions with ``y_true``.
    """

    mean_tensor = _as_2d(mean)
    variance_tensor = _as_2d(variance) if variance is not None else None
    if n_rows <= 0 or mean_tensor.shape[0] == n_rows:
        return mean_tensor, variance_tensor
    if mean_tensor.shape[0] % n_rows != 0:
        raise RuntimeError(
            "Cross-validation prediction rows must match nominal inputs or be an "
            "integer one-to-many expansion; "
            f"got predictions={mean_tensor.shape[0]}, inputs={n_rows}."
        )

    expansion = mean_tensor.shape[0] // n_rows
    grouped_mean = mean_tensor.reshape(n_rows, expansion, *mean_tensor.shape[1:])
    aggregated_mean = grouped_mean.mean(dim=1)
    if variance_tensor is None:
        return aggregated_mean, None
    if variance_tensor.shape != mean_tensor.shape:
        raise RuntimeError(
            "Cross-validation predictive mean and variance must have matching "
            "shapes before one-to-many aggregation."
        )

    grouped_variance = variance_tensor.reshape(
        n_rows, expansion, *variance_tensor.shape[1:]
    )
    second_moment = (grouped_variance + grouped_mean.square()).mean(dim=1)
    aggregated_variance = (second_moment - aggregated_mean.square()).clamp_min(0)
    return aggregated_mean, aggregated_variance


def _aggregate_expanded_probabilities(values: Any, *, n_rows: int) -> Tensor:
    """Average one-to-many class probabilities back to nominal CV rows."""

    probabilities = _as_2d(values)
    if n_rows <= 0 or probabilities.shape[0] == n_rows:
        return probabilities
    if probabilities.shape[0] % n_rows != 0:
        raise RuntimeError(
            "Cross-validation probability rows must match nominal inputs or be "
            "an integer one-to-many expansion; "
            f"got probabilities={probabilities.shape[0]}, inputs={n_rows}."
        )
    expansion = probabilities.shape[0] // n_rows
    return probabilities.reshape(
        n_rows, expansion, *probabilities.shape[1:]
    ).mean(dim=1)


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
    mean, variance = _aggregate_expanded_moments(
        result.mean,
        result.variance,
        n_rows=len(X),
    )
    model = optimizer.model
    if hasattr(model, "models") and output < len(model.models):
        model = model.models[output]
    probabilities = None
    predict_proba = getattr(model, "predict_proba", None)
    if task in {"multiclass", "ordinal"} and callable(predict_proba):
        probabilities = predict_proba(X)
        if isinstance(probabilities, tuple):
            probabilities = probabilities[0]
        probabilities = _aggregate_expanded_probabilities(
            probabilities,
            n_rows=len(X),
        )
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
    train_Yvar: Any | None = None,
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
    Yvar = None if train_Yvar is None else _as_2d(train_Yvar)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of rows.")
    if Yvar is not None and Yvar.shape != Y.shape:
        raise ValueError(
            "train_Yvar must match train_Y shape for cross-validation; "
            f"got train_Y={tuple(Y.shape)!r}, train_Yvar={tuple(Yvar.shape)!r}."
        )
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
        fold_optimizer.fit(
            X[train_idx],
            Y[train_idx],
            None if Yvar is None else Yvar[train_idx],
        )
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



def _slice_observation_data(observation_data: Any, indices: Tensor) -> Any:
    """Return one row-aligned observation subset without changing state semantics."""
    from ..observation import ObservationData

    return ObservationData(
        X=observation_data.X[indices],
        Y=observation_data.Y[indices],
        Yvar=(
            None
            if observation_data.Yvar is None
            else observation_data.Yvar[indices]
        ),
        observed_mask=observation_data.observed_mask[indices],
        failed_mask=observation_data.failed_mask[indices],
        pending_mask=observation_data.pending_mask[indices],
    )


def _observation_cv_groups(
    groups: Any | None,
    objective_indices: Tensor,
    n_rows: int,
) -> Any | None:
    """Align optional splitter groups with the objective-eligible row subset."""
    if groups is None:
        return None
    try:
        n_groups = len(groups)
    except TypeError as exc:
        raise ValueError("Cross-validation groups must be a sized row-aligned value.") from exc
    if n_groups == int(objective_indices.numel()):
        return groups
    if n_groups != int(n_rows):
        raise ValueError(
            "Observation-aware cross-validation groups must match either all "
            "observation rows or the objective-eligible rows; "
            f"got groups={n_groups}, rows={n_rows}, "
            f"objective_rows={int(objective_indices.numel())}."
        )
    import numpy as np

    values = groups.to_numpy() if callable(getattr(groups, "to_numpy", None)) else np.asarray(groups)
    return values[objective_indices.detach().cpu().numpy()]


def _metrics_for_output(
    task: str,
    y_true: Tensor,
    y_pred: Tensor,
    config: CrossValidationConfig,
    warnings: list[str],
) -> dict[str, float]:
    if task in {"regression", "multi_objective"}:
        return _regression_metrics(y_true, y_pred, config, warnings)
    return _classification_metrics(y_true, y_pred, task, config)


def _failure_probability_metrics(
    y_true: Tensor,
    p_success: Tensor,
    config: CrossValidationConfig,
    warnings: list[str],
    *,
    context: str,
) -> dict[str, float]:
    """Score experiment-success probabilities on completed experiment rows."""

    y = y_true.reshape(-1).to(dtype=torch.long)
    probability = p_success.reshape(-1).double().clamp(1e-7, 1.0 - 1e-7)
    prediction = (probability >= config.classification_threshold).to(dtype=y.dtype)
    metrics = _classification_metrics(y, prediction, "binary", config)

    try:
        from sklearn.metrics import log_loss, roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "Failure-model cross-validation requires scikit-learn; install "
            "bochan[tabular]."
        ) from exc

    y_numpy = y.detach().cpu().numpy()
    probability_numpy = probability.detach().cpu().numpy()
    if int(torch.unique(y).numel()) < 2:
        roc_auc = math.nan
        message = (
            f"Failure-model ROC-AUC is NaN for {context} because the validation "
            "targets contain only one class."
        )
        if message not in warnings:
            warnings.append(message)
    else:
        roc_auc = float(roc_auc_score(y_numpy, probability_numpy))
    metrics["roc_auc"] = roc_auc
    metrics["log_loss"] = float(
        log_loss(y_numpy, probability_numpy, labels=[0, 1])
    )
    return metrics


def _failure_prediction(
    model: Any,
    X: Tensor,
    y_true: Tensor,
    indices: Tensor,
    fold: int,
    config: CrossValidationConfig,
) -> CVPredictionResult:
    """Return success-probability predictions using the binary model posterior."""

    posterior = model.posterior(X)
    mean, variance = _aggregate_expanded_moments(
        posterior.mean,
        getattr(posterior, "variance", None),
        n_rows=len(X),
    )
    p_success = mean[..., 0].reshape(-1).clamp(0.0, 1.0)
    predictive_std = (
        variance[..., 0].clamp_min(0.0).sqrt().reshape(-1)
        if variance is not None
        else None
    )
    prediction = (p_success >= config.classification_threshold).to(
        dtype=y_true.dtype
    )
    probabilities = torch.stack((1.0 - p_success, p_success), dim=-1)
    result_indices = indices.detach().cpu()
    return CVPredictionResult(
        indices=result_indices,
        y_true=y_true.detach(),
        y_pred=prediction.detach(),
        predictive_mean=p_success.detach(),
        predictive_std=(
            predictive_std.detach() if predictive_std is not None else None
        ),
        probabilities=probabilities.detach(),
        variance_kind="bernoulli_probability",
        fold_indices=torch.full(
            (len(result_indices),), fold, dtype=torch.long
        ),
        prediction_count=torch.ones(len(result_indices), dtype=torch.long),
    )



class _ObservationFeatureImportancePredictor:
    """Expose one objective output through the raw-space inspection API."""

    def __init__(
        self,
        optimizer: Any,
        *,
        output: int,
        task: str,
        cv_config: CrossValidationConfig,
    ) -> None:
        self.optimizer = optimizer
        self.output = int(output)
        self.task = str(task)
        self.cv_config = cv_config
        model = optimizer.model
        models = getattr(model, "models", None)
        if models is not None and self.output < len(models):
            model = models[self.output]
        self.model = model

    def predict(self, X: Tensor, *, return_result: bool = False) -> Any:
        prediction = _prediction_for_output(
            self.optimizer,
            X,
            self.output,
            self.task,
            self.cv_config,
        )
        if self.task in {"multiclass", "ordinal"} and prediction.probabilities is not None:
            mean = prediction.probabilities
        else:
            mean = prediction.predictive_mean.reshape(-1, 1)
        if not return_result:
            return mean

        class _Result:
            pass

        result = _Result()
        result.mean = mean
        return result


class _FailureFeatureImportancePredictor:
    """Expose P(success) through the raw-space inspection API."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, X: Tensor, *, return_result: bool = False) -> Any:
        posterior = self.model.posterior(X)
        mean, _ = _aggregate_expanded_moments(
            posterior.mean,
            getattr(posterior, "variance", None),
            n_rows=len(X),
        )
        probability = mean[..., 0].reshape(-1, 1).clamp(0.0, 1.0)
        if not return_result:
            return probability

        class _Result:
            pass

        result = _Result()
        result.mean = probability
        return result


def _feature_importance_config_for_fold(
    config: CrossValidationConfig,
    fold: int,
    *,
    offset: int = 0,
) -> Any:
    """Clone FI settings and derive deterministic fold-local permutation seeds."""
    importance_config = copy.deepcopy(config.feature_importance_config)
    if importance_config is None:
        return None
    if getattr(importance_config, "random_state", None) is not None:
        importance_config.random_state = (
            int(importance_config.random_state) + int(fold) * 104729 + int(offset)
        )
    return importance_config


def _observation_output_feature_importance(
    fold_optimizer: Any,
    X: Tensor,
    Y: Tensor,
    indices: Tensor,
    *,
    output: int,
    name: str,
    task: str,
    fold: int,
    config: CrossValidationConfig,
) -> Any:
    """Evaluate one objective only on successful observed validation cells."""
    from bochan.inspection import compute_feature_importance

    importance_config = _feature_importance_config_for_fold(config, fold)
    predictor = _ObservationFeatureImportancePredictor(
        fold_optimizer,
        output=output,
        task=task,
        cv_config=config,
    )
    bundle = getattr(fold_optimizer, "bundle", None)
    cat_dims = tuple(getattr(bundle, "cat_dims", ()) or ())
    result = compute_feature_importance(
        model=predictor.model,
        predictor=predictor,
        X=X[indices],
        y=Y[indices, output : output + 1],
        task_type=task,
        feature_names=config.feature_names,
        output_names=[name],
        cat_dims=cat_dims,
        config=importance_config,
        training_data=False,
    )
    result.metadata.update(
        {
            "cv_fold": fold,
            "derived_random_state": getattr(importance_config, "random_state", None),
            "observation_aware": True,
            "validation_protocol": "successful_observed_cells",
            "n_validation_rows": int(indices.numel()),
        }
    )
    return result


def _failure_feature_importance(
    failure_model: Any,
    X: Tensor,
    success_targets: Tensor,
    indices: Tensor,
    *,
    fold: int,
    config: CrossValidationConfig,
    cat_dims: tuple[int, ...] = (),
) -> Any:
    """Evaluate experiment-success FI only on completed validation rows."""
    from bochan.inspection import compute_feature_importance

    importance_config = _feature_importance_config_for_fold(config, fold, offset=7919)
    predictor = _FailureFeatureImportancePredictor(failure_model)
    result = compute_feature_importance(
        model=failure_model,
        predictor=predictor,
        X=X[indices],
        y=success_targets[indices].reshape(-1, 1),
        task_type="binary",
        feature_names=config.feature_names,
        output_names=["experiment_success"],
        cat_dims=cat_dims,
        config=importance_config,
        training_data=False,
    )
    result.metadata.update(
        {
            "cv_fold": fold,
            "derived_random_state": getattr(importance_config, "random_state", None),
            "observation_aware": True,
            "validation_protocol": "completed_rows",
            "probability_target": "success",
            "n_validation_rows": int(indices.numel()),
        }
    )
    return result


def _combine_observation_feature_importance(
    per_output: dict[str, list[Any]],
) -> Any | None:
    """Combine output-local CV FI aggregates into the standard result contract."""
    available = {name: folds for name, folds in per_output.items() if folds}
    if not available:
        return None

    from bochan.inspection.result_types import CrossValidatedFeatureImportanceResult

    outputs = {}
    warnings: list[str] = []
    feature_names = None
    fold_counts: dict[str, int] = {}
    for name, folds in available.items():
        aggregated = _aggregate_feature_importance(folds)
        outputs.update(aggregated.outputs)
        warnings.extend(aggregated.warnings)
        feature_names = feature_names or aggregated.feature_names
        fold_counts[name] = len(folds)
    return CrossValidatedFeatureImportanceResult(
        outputs=outputs,
        feature_names=feature_names or (),
        warnings=list(dict.fromkeys(warnings)),
        metadata={
            "observation_aware": True,
            "protocol": "output_specific_successful_observed_validation_cells",
            "n_folds_by_output": fold_counts,
            "pooled_oof_importance": False,
        },
    )


def cross_validate_observations(
    optimizer: Any,
    observation_data: Any,
    *,
    model_config: ModelConfig | None = None,
    fit_config: FitConfig | None = None,
    cv_config: CrossValidationConfig | None = None,
    failure_config: Any | None = None,
) -> CrossValidationResult:
    """Cross-validate observed objectives and, optionally, experiment success.

    Without ``failure_config`` this preserves the Phase-5 protocol exactly: folds
    are formed from successful rows carrying at least one observed objective cell.
    With failure modeling enabled, folds are instead formed from all completed
    rows (successful + failed), pending rows remain excluded, and the same row
    split is shared by the objective model and success classifier. Objective
    metrics continue to use only successful observed target cells.
    """
    from ..observation import ObservationData

    if not isinstance(observation_data, ObservationData):
        raise TypeError("observation_data must be an ObservationData instance.")
    config = cv_config or CrossValidationConfig()
    if config.feature_importance_config is not None:
        from bochan.inspection import FeatureImportanceConfig

        if not isinstance(config.feature_importance_config, FeatureImportanceConfig):
            raise ValueError(
                "Observation-aware feature importance requires a "
                "FeatureImportanceConfig instance."
            )

    X, Y = observation_data.X, observation_data.Y
    objective_mask = observation_data.objective_row_mask
    objective_indices = torch.nonzero(objective_mask, as_tuple=False).reshape(-1)
    if int(objective_indices.numel()) < 2:
        raise ValueError(
            "Observation-aware cross-validation requires at least two successful "
            "rows containing an observed objective value."
        )

    base_model = model_config or optimizer.model_config
    base_fit = fit_config if fit_config is not None else optimizer.fit_config
    names, tasks = _output_layout(base_model, Y)
    successful_observed = (
        observation_data.success_mask.unsqueeze(-1) & observation_data.observed_mask
    )
    observed_counts = successful_observed.sum(dim=0)
    missing_outputs = [
        names[index]
        for index, count in enumerate(observed_counts.tolist())
        if int(count) == 0
    ]
    if missing_outputs:
        raise ValueError(
            "Every objective output requires at least one successful observed value "
            f"for cross-validation; missing outputs: {missing_outputs!r}."
        )

    failure_enabled = failure_config is not None
    warnings: list[str] = []
    if failure_enabled:
        split_indices = torch.nonzero(
            observation_data.completed_mask, as_tuple=False
        ).reshape(-1)
        if int(split_indices.numel()) < 2:
            raise ValueError(
                "Failure-model cross-validation requires at least two completed "
                "experiment rows."
            )
        strat_y = observation_data.success_mask[split_indices].to(dtype=Y.dtype)
        if int(torch.unique(strat_y).numel()) < 2:
            raise ValueError(
                "Failure-model cross-validation requires both successful and failed "
                "completed experiments."
            )
        split_task = "binary"
    else:
        split_indices = objective_indices
        strat_y = Y[split_indices, 0]
        split_task = str(base_model.task_type)
        if split_task == "hybrid" and config.stratify_output is not None:
            output_index = (
                names.index(config.stratify_output)
                if isinstance(config.stratify_output, str)
                else int(config.stratify_output)
            )
            if output_index < 0 or output_index >= len(names):
                raise IndexError(
                    f"stratify_output={output_index} is outside [0, {len(names) - 1}]."
                )
            strat_y = Y[split_indices, output_index]
            split_task = tasks[output_index]

    eligible_X = X[split_indices]
    splitter, auto_y = _make_splitter(config, split_task, strat_y, warnings)
    if (
        auto_y is not None
        and torch.is_floating_point(strat_y)
        and not bool(torch.isfinite(strat_y).all())
    ):
        raise ValueError(
            "Stratified observation-aware cross-validation requires the "
            "stratification output to be observed on every eligible row."
        )
    split_y = auto_y if auto_y is not None else strat_y.detach().cpu().numpy()
    groups = _observation_cv_groups(config.groups, split_indices, len(X))
    try:
        splits = list(
            splitter.split(
                eligible_X.detach().cpu().numpy(),
                split_y,
                groups=groups,
            )
        )
    except TypeError:
        splits = list(splitter.split(eligible_X.detach().cpu().numpy(), split_y))
    if not splits:
        raise ValueError("The configured cross-validation splitter produced no folds.")

    per_output: dict[str, list[CVFoldResult]] = {name: [] for name in names}
    objective_importances: dict[str, list[Any]] = {name: [] for name in names}
    failure_folds: list[CVFoldResult] = []
    failure_importances: list[Any] = []
    success_targets = observation_data.success_mask.to(dtype=Y.dtype)
    models = [] if config.return_models else None
    for fold, (train_raw, test_raw) in enumerate(splits):
        train_relative = torch.as_tensor(
            train_raw, dtype=torch.long, device=split_indices.device
        )
        test_relative = torch.as_tensor(
            test_raw, dtype=torch.long, device=split_indices.device
        )
        train_indices = split_indices[train_relative]
        test_indices = split_indices[test_relative]
        train_observations = _slice_observation_data(observation_data, train_indices)
        train_counts = train_observations.observed_mask.sum(dim=0)
        empty_train_outputs = [
            names[index]
            for index, count in enumerate(train_counts.tolist())
            if int(count) == 0
        ]
        if empty_train_outputs:
            raise ValueError(
                "Observation-aware cross-validation cannot train a fold with no "
                "observed values for an output; "
                f"fold={fold}, outputs={empty_train_outputs!r}. Reduce n_splits, "
                "shuffle the folds, or collect more observations for those outputs."
            )
        if failure_enabled:
            train_success = observation_data.success_mask[train_indices]
            if int(torch.unique(train_success).numel()) < 2:
                raise ValueError(
                    "Failure-model cross-validation cannot train a fold without both "
                    "successful and failed completed experiments; "
                    f"fold={fold}. Reduce n_splits, enable shuffling/stratification, "
                    "or collect both experiment outcomes."
                )

        fold_optimizer = type(optimizer)(
            model_config=clone_model_config_for_evaluation(base_model),
            fit_config=clone_fit_config_for_evaluation(base_fit),
            bounds=copy.deepcopy(optimizer.bounds),
            model_registry=optimizer.model_registry,
            acquisition_registry=optimizer.acquisition_registry,
        )
        fit_kwargs = {
            "observation_data": train_observations,
            "model_config": clone_model_config_for_evaluation(base_model),
            "fit_config": clone_fit_config_for_evaluation(base_fit),
        }
        if failure_enabled:
            fit_kwargs["failure_config"] = copy.deepcopy(failure_config)
        fold_optimizer.fit(**fit_kwargs)
        if models is not None:
            models.append(fold_optimizer.model)

        if failure_enabled:
            failure_model = getattr(fold_optimizer, "failure_model", None)
            if failure_model is None:
                raise RuntimeError(
                    "Failure-model cross-validation expected a fitted success "
                    f"classifier in fold={fold}."
                )
            failure_train = _failure_prediction(
                failure_model,
                X[train_indices],
                success_targets[train_indices],
                train_indices,
                fold,
                config,
            )
            failure_test = _failure_prediction(
                failure_model,
                X[test_indices],
                success_targets[test_indices],
                test_indices,
                fold,
                config,
            )
            failure_importance = None
            if config.feature_importance_config is not None:
                bundle = getattr(fold_optimizer, "bundle", None)
                cat_dims = tuple(getattr(bundle, "cat_dims", ()) or ())
                failure_importance = _failure_feature_importance(
                    failure_model,
                    X,
                    success_targets,
                    test_indices,
                    fold=fold,
                    config=config,
                    cat_dims=cat_dims,
                )
                failure_importances.append(failure_importance)
            failure_folds.append(
                CVFoldResult(
                    fold=fold,
                    train_indices=train_indices.detach().cpu(),
                    test_indices=test_indices.detach().cpu(),
                    train_metrics=_failure_probability_metrics(
                        failure_train.y_true,
                        failure_train.predictive_mean,
                        config,
                        warnings,
                        context=f"training fold {fold}",
                    ),
                    test_metrics=_failure_probability_metrics(
                        failure_test.y_true,
                        failure_test.predictive_mean,
                        config,
                        warnings,
                        context=f"validation fold {fold}",
                    ),
                    train_predictions=failure_train,
                    test_predictions=failure_test,
                    feature_importance=failure_importance,
                )
            )

        for output, (name, task) in enumerate(zip(names, tasks, strict=True)):
            train_cell_mask = (
                observation_data.success_mask[train_indices]
                & observation_data.observed_mask[train_indices, output]
            )
            test_cell_mask = (
                observation_data.success_mask[test_indices]
                & observation_data.observed_mask[test_indices, output]
            )
            output_train_indices = train_indices[train_cell_mask]
            output_test_indices = test_indices[test_cell_mask]
            if int(output_test_indices.numel()) == 0:
                warnings.append(
                    f"Output {name!r} has no observed validation target in fold {fold}; "
                    "that output/fold metric was skipped."
                )
                continue

            train_pred = _prediction_for_output(
                fold_optimizer,
                X[output_train_indices],
                output,
                task,
                config,
            )
            test_pred = _prediction_for_output(
                fold_optimizer,
                X[output_test_indices],
                output,
                task,
                config,
            )
            for pred_result, indices in (
                (train_pred, output_train_indices),
                (test_pred, output_test_indices),
            ):
                result_indices = indices.detach().cpu()
                pred_result.indices = result_indices
                pred_result.y_true = Y[indices, output].detach()
                pred_result.fold_indices = torch.full(
                    (len(result_indices),), fold, dtype=torch.long
                )
                pred_result.prediction_count = torch.ones(
                    len(result_indices), dtype=torch.long
                )

            fold_importance = None
            if config.feature_importance_config is not None:
                fold_importance = _observation_output_feature_importance(
                    fold_optimizer,
                    X,
                    Y,
                    output_test_indices,
                    output=output,
                    name=name,
                    task=task,
                    fold=fold,
                    config=config,
                )
                objective_importances[name].append(fold_importance)

            per_output[name].append(
                CVFoldResult(
                    fold=fold,
                    train_indices=output_train_indices.detach().cpu(),
                    test_indices=output_test_indices.detach().cpu(),
                    train_metrics=_metrics_for_output(
                        task,
                        train_pred.y_true,
                        train_pred.y_pred,
                        config,
                        warnings,
                    ),
                    test_metrics=_metrics_for_output(
                        task,
                        test_pred.y_true,
                        test_pred.y_pred,
                        config,
                        warnings,
                    ),
                    train_predictions=train_pred,
                    test_predictions=test_pred,
                    feature_importance=fold_importance,
                )
            )

    outputs: dict[str, OutputCrossValidationResult] = {}
    for output, (name, task) in enumerate(zip(names, tasks, strict=True)):
        folds = per_output[name]
        if not folds:
            raise ValueError(
                f"Output {name!r} has no observed validation targets in any fold."
            )
        test_parts = [
            fold_result.test_predictions
            for fold_result in folds
            if fold_result.test_predictions is not None
        ]
        train_parts = [
            fold_result.train_predictions
            for fold_result in folds
            if fold_result.train_predictions is not None
        ]
        oof = _aggregate(test_parts, Y[:, output], len(Y))
        aggregated_train = (
            _aggregate(train_parts, Y[:, output], len(Y))
            if config.return_train_predictions
            else None
        )
        metric_names = list(folds[0].test_metrics)
        outputs[name] = OutputCrossValidationResult(
            task_type=task,
            folds=folds,
            train_metric_summary={
                key: _summary([fold_result.train_metrics[key] for fold_result in folds])
                for key in metric_names
            },
            test_metric_summary={
                key: _summary([fold_result.test_metrics[key] for fold_result in folds])
                for key in metric_names
            },
            aggregated_train_predictions=aggregated_train,
            oof_predictions=oof,
            oof_metrics=_metrics_for_output(
                task,
                oof.y_true,
                oof.y_pred,
                config,
                warnings,
            ),
        )
        if not config.return_fold_predictions:
            for fold_result in folds:
                fold_result.train_predictions = None
                fold_result.test_predictions = None
        elif not config.return_train_predictions:
            for fold_result in folds:
                fold_result.train_predictions = None

    failure_result = None
    if failure_enabled:
        failure_test_parts = [
            fold_result.test_predictions
            for fold_result in failure_folds
            if fold_result.test_predictions is not None
        ]
        failure_train_parts = [
            fold_result.train_predictions
            for fold_result in failure_folds
            if fold_result.train_predictions is not None
        ]
        failure_oof = _aggregate(failure_test_parts, success_targets, len(X))
        failure_aggregated_train = (
            _aggregate(failure_train_parts, success_targets, len(X))
            if config.return_train_predictions
            else None
        )
        failure_metric_names = list(failure_folds[0].test_metrics)
        failure_result = OutputCrossValidationResult(
            task_type="binary",
            folds=failure_folds,
            train_metric_summary={
                key: _summary(
                    [fold_result.train_metrics[key] for fold_result in failure_folds]
                )
                for key in failure_metric_names
            },
            test_metric_summary={
                key: _summary(
                    [fold_result.test_metrics[key] for fold_result in failure_folds]
                )
                for key in failure_metric_names
            },
            aggregated_train_predictions=failure_aggregated_train,
            oof_predictions=failure_oof,
            oof_metrics=_failure_probability_metrics(
                failure_oof.y_true,
                failure_oof.predictive_mean,
                config,
                warnings,
                context="pooled OOF predictions",
            ),
        )
        if not config.return_fold_predictions:
            for fold_result in failure_folds:
                fold_result.train_predictions = None
                fold_result.test_predictions = None
        elif not config.return_train_predictions:
            for fold_result in failure_folds:
                fold_result.train_predictions = None

    objective_feature_importance = _combine_observation_feature_importance(
        objective_importances
    )
    failure_feature_importance = (
        _aggregate_feature_importance(failure_importances)
        if failure_importances
        else None
    )

    success_without_objective = (
        observation_data.success_mask & ~observation_data.observed_mask.any(dim=-1)
    )
    metadata = {
        "random_state": config.random_state,
        "return_models": config.return_models,
        "observation_aware": True,
        "objective_protocol": "successful_observed_cells",
        "failure_model_evaluated": failure_enabled,
        "feature_importance_evaluated": config.feature_importance_config is not None,
        "feature_importance_protocol": (
            "output_specific_successful_observed_validation_cells"
            if config.feature_importance_config is not None
            else None
        ),
        "known_observation_variance": observation_data.Yvar is not None,
        "n_rows": int(len(X)),
        "n_objective_rows": int(objective_indices.numel()),
        "n_excluded_failed_rows": int(observation_data.failed_mask.sum()),
        "n_excluded_pending_rows": int(observation_data.pending_mask.sum()),
        "n_excluded_success_without_objective": int(success_without_objective.sum()),
        "observed_per_output": {
            name: int(observed_counts[index]) for index, name in enumerate(names)
        },
    }
    if failure_enabled:
        metadata.update(
            {
                "failure_protocol": "completed_rows_shared_with_objectives",
                "failure_probability_target": "success",
                "n_completed_rows": int(observation_data.completed_mask.sum()),
                "n_success_rows": int(observation_data.success_mask.sum()),
                "n_failed_rows": int(observation_data.failed_mask.sum()),
                "n_failure_validation_rows": int(split_indices.numel()),
                "failure_feature_importance_evaluated": (
                    failure_feature_importance is not None
                ),
            }
        )
    return CrossValidationResult(
        outputs=outputs,
        splitter_name=type(splitter).__name__,
        n_splits=len(splits),
        warnings=warnings,
        metadata=metadata,
        models=models,
        feature_importance=objective_feature_importance,
        failure_model=failure_result,
        failure_feature_importance=failure_feature_importance,
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
            method_folds = [
                fold.outputs[output_name].predictive_methods[method_name]
                for fold in folds
            ]
            entries = {}
            for entry_name in method_folds[0].entries:
                fold_entries = [method.entries[entry_name] for method in method_folds]
                values = torch.tensor(
                    [entry.importance.mean for entry in fold_entries],
                    dtype=torch.float64,
                )
                ranks = torch.tensor(
                    [entry.importance.rank for entry in fold_entries],
                    dtype=torch.float64,
                )
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
            methods[method_name] = CrossValidatedMethodResult(
                method_name, entries, method_folds
            )
        outputs[output_name] = CrossValidatedOutputImportance(
            output_name,
            first_output.task_type,
            methods,
            {},
            [fold.outputs[output_name].model_diagnostics for fold in folds],
            warnings=[
                warning
                for fold in folds
                for warning in fold.outputs[output_name].warnings
            ],
        )
    return CrossValidatedFeatureImportanceResult(
        outputs,
        folds[0].feature_names,
        [warning for fold in folds for warning in fold.warnings],
        {"n_folds": len(folds), "pooled_oof_importance": False},
    )
