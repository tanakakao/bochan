from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label} marker not found")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Observation-aware CV: jointly evaluate objective outputs and success model.
# ---------------------------------------------------------------------------
cv_path = Path("src/bochan/api/evaluation/cross_validation.py")
cv_text = cv_path.read_text(encoding="utf-8")

old_result_tail = '''    metadata: dict[str, Any] = field(default_factory=dict)\n    models: list[Any] | None = None\n    feature_importance: Any | None = None\n'''
new_result_tail = '''    metadata: dict[str, Any] = field(default_factory=dict)\n    models: list[Any] | None = None\n    feature_importance: Any | None = None\n    failure_model: OutputCrossValidationResult | None = None\n'''
cv_text = replace_once(
    cv_text,
    old_result_tail,
    new_result_tail,
    label="CrossValidationResult tail",
)

metrics_marker = '''def cross_validate_observations(\n'''
failure_helpers = r'''def _failure_probability_metrics(
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


'''
if failure_helpers not in cv_text:
    cv_text = replace_once(
        cv_text,
        metrics_marker,
        failure_helpers + metrics_marker,
        label="observation CV marker",
    )

start = cv_text.index("def cross_validate_observations(\n")
end = cv_text.index("\ndef _aggregate_feature_importance", start)
new_observation_cv = r'''def cross_validate_observations(
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
        raise ValueError(
            "Observation-aware cross-validation does not yet support fold feature "
            "importance because partial target cells require an output-specific "
            "importance protocol."
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
    failure_folds: list[CVFoldResult] = []
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

    success_without_objective = (
        observation_data.success_mask & ~observation_data.observed_mask.any(dim=-1)
    )
    metadata = {
        "random_state": config.random_state,
        "return_models": config.return_models,
        "observation_aware": True,
        "objective_protocol": "successful_observed_cells",
        "failure_model_evaluated": failure_enabled,
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
            }
        )
    return CrossValidationResult(
        outputs=outputs,
        splitter_name=type(splitter).__name__,
        n_splits=len(splits),
        warnings=warnings,
        metadata=metadata,
        models=models,
        feature_importance=None,
        failure_model=failure_result,
    )

'''
cv_text = cv_text[:start] + new_observation_cv + cv_text[end:]
cv_path.write_text(cv_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Public optimizer method: pass optional failure config into observation CV.
# ---------------------------------------------------------------------------
optimizer_path = Path("src/bochan/api/optimizer/__init__.py")
optimizer_text = optimizer_path.read_text(encoding="utf-8")
optimizer_text = replace_once(
    optimizer_text,
    '''        fit_config: FitConfig | None = None,\n        cv_config: Any | None = None,\n    ) -> Any:\n        """Cross-validate successful observed target cells without mutating state."""\n''',
    '''        fit_config: FitConfig | None = None,\n        cv_config: Any | None = None,\n        failure_config: ExperimentFailureConfig | None = None,\n    ) -> Any:\n        """Cross-validate observed objectives and optional experiment success."""\n''',
    label="optimizer observation CV signature",
)
optimizer_text = replace_once(
    optimizer_text,
    '''            fit_config=fit_config,\n            cv_config=cv_config,\n        )\n\n    def cross_validate(self, *args: Any, **kwargs: Any) -> Any:\n''',
    '''            fit_config=fit_config,\n            cv_config=cv_config,\n            failure_config=failure_config,\n        )\n\n    def cross_validate(self, *args: Any, **kwargs: Any) -> Any:\n''',
    label="optimizer observation CV call",
)
optimizer_path.write_text(optimizer_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tabular fit: resolve failure config before CV and reuse it for final fit.
# ---------------------------------------------------------------------------
fitting_path = Path("src/bochan/tabular/optimizer/fitting.py")
fitting_text = fitting_path.read_text(encoding="utf-8")
fitting_text = replace_once(
    fitting_text,
    '''    resolved_cv = resolve_cv_config(cv_config) if cv_config is not None else owner.cv_config\n    owner.cross_validation_result_ = None\n''',
    '''    resolved_cv = resolve_cv_config(cv_config) if cv_config is not None else owner.cv_config\n    resolved_failure_config = owner.observation.resolve_failure_config(failure_config)\n    owner.cross_validation_result_ = None\n''',
    label="tabular pre-CV failure config",
)
fitting_text = replace_once(
    fitting_text,
    '''                fit_config=owner.fit_config,\n                cv_config=resolved_cv or CrossValidationConfig(),\n            )\n''',
    '''                fit_config=owner.fit_config,\n                cv_config=resolved_cv or CrossValidationConfig(),\n                failure_config=resolved_failure_config,\n            )\n''',
    label="tabular observation CV failure config",
)
fitting_text = fitting_text.replace(
    "    resolved_failure_config = owner.observation.resolve_failure_config(failure_config)\n    if uses_observation_conversion:\n",
    "    if uses_observation_conversion:\n",
    1,
)
fitting_path.write_text(fitting_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Focused tests.
# ---------------------------------------------------------------------------
test_path = Path("tests/test_material_train_yvar_phase7.py")
test_path.write_text(r'''from types import SimpleNamespace

import pytest
import torch

from bochan.api.configs import FitConfig, ModelConfig
from bochan.api.evaluation.cross_validation import (
    CrossValidationConfig,
    cross_validate_observations,
)
from bochan.api.observation import ExperimentFailureConfig, ObservationData


class _FakeFailureModel:
    def posterior(self, X):
        probability = torch.sigmoid(X[:, :1])
        return SimpleNamespace(
            mean=probability,
            variance=probability * (1.0 - probability),
        )


class _FakeOptimizer:
    fit_observations = []
    fit_failure_configs = []

    def __init__(
        self,
        *,
        model_config=None,
        fit_config=None,
        bounds=None,
        model_registry=None,
        acquisition_registry=None,
    ):
        self.model_config = model_config or ModelConfig(
            task_type="regression", model_type="base"
        )
        self.fit_config = fit_config or FitConfig()
        self.bounds = bounds
        self.model_registry = model_registry
        self.acquisition_registry = acquisition_registry
        self.model = object()
        self.failure_model = None

    def fit(
        self,
        *,
        observation_data,
        model_config=None,
        fit_config=None,
        failure_config=None,
    ):
        type(self).fit_observations.append(observation_data)
        type(self).fit_failure_configs.append(failure_config)
        self.model = object()
        self.failure_model = _FakeFailureModel() if failure_config is not None else None
        return self

    def predict(self, X, *, return_result=False):
        mean = X[:, :1]
        variance = torch.full_like(mean, 0.04)
        result = SimpleNamespace(
            mean=mean,
            variance=variance,
            variance_kind="posterior",
        )
        return result if return_result else mean


def _optimizer():
    return _FakeOptimizer(
        model_config=ModelConfig(task_type="regression", model_type="base"),
        fit_config=FitConfig(),
    )


def _observations_with_failure_and_noise():
    X = torch.linspace(-2.0, 2.0, 9, dtype=torch.double).unsqueeze(-1)
    failed = torch.tensor(
        [False, True, False, True, False, True, False, True, False]
    )
    pending = torch.tensor(
        [False, False, False, False, False, False, False, False, True]
    )
    Y = torch.tensor(
        [[-2.0], [float("nan")], [-1.0], [float("nan")], [0.0],
         [float("nan")], [1.0], [float("nan")], [float("nan")]],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [[0.01], [float("nan")], [0.02], [float("nan")], [0.03],
         [float("nan")], [0.04], [float("nan")], [float("nan")]],
        dtype=torch.double,
    )
    return ObservationData(
        X=X,
        Y=Y,
        Yvar=Yvar,
        failed_mask=failed,
        pending_mask=pending,
    )


def test_phase7_failure_cv_uses_completed_rows_and_preserves_objective_masks():
    _FakeOptimizer.fit_observations.clear()
    _FakeOptimizer.fit_failure_configs.clear()
    observations = _observations_with_failure_and_noise()

    result = cross_validate_observations(
        _optimizer(),
        observations,
        cv_config=CrossValidationConfig(n_splits=2, random_state=7),
        failure_config=ExperimentFailureConfig(),
    )

    assert result.failure_model is not None
    assert result.metadata["failure_model_evaluated"] is True
    assert result.metadata["failure_probability_target"] == "success"
    assert result.metadata["n_completed_rows"] == 8
    assert result.metadata["n_failure_validation_rows"] == 8
    assert result.metadata["n_excluded_pending_rows"] == 1
    assert result.failure_model.oof_predictions.indices.tolist() == list(range(8))
    assert result.output.oof_predictions.indices.tolist() == [0, 2, 4, 6]
    assert set(result.failure_model.oof_metrics) >= {
        "accuracy",
        "f1",
        "roc_auc",
        "log_loss",
    }
    assert all(config is not None for config in _FakeOptimizer.fit_failure_configs)
    assert all(
        fold_observations.Yvar is not None
        for fold_observations in _FakeOptimizer.fit_observations
    )


def test_phase7_without_failure_config_preserves_phase5_protocol():
    observations = _observations_with_failure_and_noise()
    result = cross_validate_observations(
        _optimizer(),
        observations,
        cv_config=CrossValidationConfig(n_splits=2, random_state=3),
    )

    assert result.failure_model is None
    assert result.metadata["failure_model_evaluated"] is False
    assert "failure_protocol" not in result.metadata
    assert result.metadata["n_objective_rows"] == 4
    assert result.output.oof_predictions.indices.tolist() == [0, 2, 4, 6]


def test_phase7_one_class_validation_sets_roc_auc_nan_and_warns():
    X = torch.arange(6, dtype=torch.double).unsqueeze(-1)
    failed = torch.tensor([False, False, True, True, False, True])
    Y = torch.tensor(
        [[0.0], [1.0], [float("nan")], [float("nan")], [4.0], [float("nan")]],
        dtype=torch.double,
    )
    observations = ObservationData(X=X, Y=Y, failed_mask=failed)

    result = cross_validate_observations(
        _optimizer(),
        observations,
        cv_config=CrossValidationConfig(
            splitter="kfold",
            n_splits=3,
            shuffle=False,
        ),
        failure_config=ExperimentFailureConfig(),
    )

    assert result.failure_model is not None
    auc_values = result.failure_model.test_metric_summary["roc_auc"].values
    assert torch.isnan(auc_values).any()
    assert any("ROC-AUC is NaN" in warning for warning in result.warnings)


def test_phase7_rejects_fold_without_both_failure_classes():
    X = torch.arange(4, dtype=torch.double).unsqueeze(-1)
    failed = torch.tensor([False, False, False, True])
    Y = torch.tensor([[0.0], [1.0], [2.0], [float("nan")]], dtype=torch.double)
    observations = ObservationData(X=X, Y=Y, failed_mask=failed)

    with pytest.raises(ValueError, match="both successful and failed"):
        cross_validate_observations(
            _optimizer(),
            observations,
            cv_config=CrossValidationConfig(
                splitter="kfold",
                n_splits=2,
                shuffle=False,
            ),
            failure_config=ExperimentFailureConfig(),
        )
''', encoding="utf-8")


doc_path = Path("docs/material_train_yvar_phase7.md")
doc_path.write_text(r'''# Material `train_Yvar` Phase 7

Phase 7 extends observation-aware cross-validation to the experiment
success/failure classifier that can accompany physical-experiment Bayesian
optimization.

## Protocol

When no failure model is configured, the Phase-5 protocol is unchanged: folds
are built only from successful rows carrying at least one observed objective.

When `ExperimentFailureConfig` is supplied:

- the split universe is every completed experiment (`success` + `failed`);
- pending experiments are excluded;
- one row split is shared by the objective model and success classifier;
- objective training and scoring still use only successful observed target cells;
- known per-cell `Yvar` remains aligned through fold slicing;
- the classifier predicts `P(success)` and reports OOF probability metrics;
- `splitter="auto"` stratifies the shared folds by success/failure status;
- a training fold containing only one outcome is rejected explicitly;
- ROC-AUC is reported as `NaN` with a warning when a validation fold contains
  only one class, while accuracy/F1/log-loss remain available.

The classifier result is exposed separately as
`CrossValidationResult.failure_model`, so existing objective-output consumers of
`CrossValidationResult.outputs` remain backward compatible.

## Metrics

The failure-model result includes the normal binary classification metrics plus
probability-aware `roc_auc` and `log_loss`. The probability column is ordered as
`[P(failure), P(success)]`; `predictive_mean` is `P(success)`.
''', encoding="utf-8")
