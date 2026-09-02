from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"pattern not found in {path}: {old[:120]!r}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1) Observation-aware evaluator.
path = Path("src/bochan/api/evaluation/cross_validation.py")
text = path.read_text(encoding="utf-8")
anchor = "\n\ndef _aggregate_feature_importance(folds: list[Any]) -> Any:\n"
if anchor not in text:
    raise SystemExit("cross-validation insertion anchor not found")
implementation = r'''


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


def cross_validate_observations(
    optimizer: Any,
    observation_data: Any,
    *,
    model_config: ModelConfig | None = None,
    fit_config: FitConfig | None = None,
    cv_config: CrossValidationConfig | None = None,
) -> CrossValidationResult:
    """Cross-validate an objective model over successful observed target cells.

    A single row split is shared by all outputs. Pending and failed experiments are
    excluded from objective validation. For partially observed multi-output data,
    each output is scored only where that target cell is observed. Known ``Yvar``
    remains row/cell aligned inside every fold and is passed through
    :class:`ObservationData` to model construction.
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

    eligible_X = X[objective_indices]
    eligible_Y = Y[objective_indices]
    warnings: list[str] = []
    split_task = str(base_model.task_type)
    strat_y = eligible_Y[:, 0]
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
        strat_y, split_task = eligible_Y[:, output_index], tasks[output_index]

    splitter, auto_y = _make_splitter(config, split_task, strat_y, warnings)
    if auto_y is not None and torch.is_floating_point(strat_y):
        if not bool(torch.isfinite(strat_y).all()):
            raise ValueError(
                "Stratified observation-aware cross-validation requires the "
                "stratification output to be observed on every objective-eligible row."
            )
    split_y = auto_y if auto_y is not None else strat_y.detach().cpu().numpy()
    groups = _observation_cv_groups(config.groups, objective_indices, len(X))
    try:
        splits = list(splitter.split(eligible_X.detach().cpu().numpy(), split_y, groups=groups))
    except TypeError:
        splits = list(splitter.split(eligible_X.detach().cpu().numpy(), split_y))
    if not splits:
        raise ValueError("The configured cross-validation splitter produced no folds.")

    per_output: dict[str, list[CVFoldResult]] = {name: [] for name in names}
    models = [] if config.return_models else None
    for fold, (train_raw, test_raw) in enumerate(splits):
        train_relative = torch.as_tensor(
            train_raw, dtype=torch.long, device=objective_indices.device
        )
        test_relative = torch.as_tensor(
            test_raw, dtype=torch.long, device=objective_indices.device
        )
        train_indices = objective_indices[train_relative]
        test_indices = objective_indices[test_relative]
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

        fold_optimizer = type(optimizer)(
            model_config=clone_model_config_for_evaluation(base_model),
            fit_config=clone_fit_config_for_evaluation(base_fit),
            bounds=copy.deepcopy(optimizer.bounds),
            model_registry=optimizer.model_registry,
            acquisition_registry=optimizer.acquisition_registry,
        )
        fold_optimizer.fit(
            observation_data=train_observations,
            model_config=clone_model_config_for_evaluation(base_model),
            fit_config=clone_fit_config_for_evaluation(base_fit),
        )
        if models is not None:
            models.append(fold_optimizer.model)

        for output, (name, task) in enumerate(zip(names, tasks, strict=True)):
            train_cell_mask = observation_data.observed_mask[train_indices, output]
            test_cell_mask = observation_data.observed_mask[test_indices, output]
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
            fold.test_predictions
            for fold in folds
            if fold.test_predictions is not None
        ]
        train_parts = [
            fold.train_predictions
            for fold in folds
            if fold.train_predictions is not None
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
                key: _summary([fold.train_metrics[key] for fold in folds])
                for key in metric_names
            },
            test_metric_summary={
                key: _summary([fold.test_metrics[key] for fold in folds])
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

    success_without_objective = (
        observation_data.success_mask & ~observation_data.observed_mask.any(dim=-1)
    )
    metadata = {
        "random_state": config.random_state,
        "return_models": config.return_models,
        "observation_aware": True,
        "objective_protocol": "successful_observed_cells",
        "failure_model_evaluated": False,
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
    return CrossValidationResult(
        outputs=outputs,
        splitter_name=type(splitter).__name__,
        n_splits=len(splits),
        warnings=warnings,
        metadata=metadata,
        models=models,
        feature_importance=None,
    )
'''
text = text.replace(anchor, implementation + anchor, 1)
path.write_text(text, encoding="utf-8")

# 2) Public optimizer convenience API.
path = Path("src/bochan/api/optimizer/__init__.py")
text = path.read_text(encoding="utf-8")
anchor = "    def cross_validate(self, *args: Any, **kwargs: Any) -> Any:\n"
if anchor not in text:
    raise SystemExit("optimizer cross_validate anchor not found")
method = r'''    def cross_validate_observations(
        self,
        observation_data: ObservationData | None = None,
        *,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
        cv_config: Any | None = None,
    ) -> Any:
        """Cross-validate successful observed target cells without mutating state."""
        observations = observation_data or self.observations
        if observations is None:
            raise ValueError(
                "observation_data is required when the optimizer has no stored "
                "observation state."
            )
        from ..evaluation.cross_validation import cross_validate_observations

        return cross_validate_observations(
            self,
            observations,
            model_config=model_config,
            fit_config=fit_config,
            cv_config=cv_config,
        )

'''
text = text.replace(anchor, method + anchor, 1)
path.write_text(text, encoding="utf-8")

# 3) Route tabular observation workflows to the observation-aware evaluator.
replace_once(
    "src/bochan/tabular/optimizer/fitting.py",
    '''    run_cv = owner.cross_validation if cross_validation is None else bool(cross_validation)\n    if owner.observation.uses_observation_conversion(resolved) and run_cv:\n        raise ValueError(\n            "Cross-validation requires an observation-aware validation protocol."\n        )\n    dataset = to_dataset(owner, fit_data, y, data_config=resolved)\n''',
    '''    run_cv = owner.cross_validation if cross_validation is None else bool(cross_validation)\n    uses_observation_conversion = owner.observation.uses_observation_conversion(resolved)\n    dataset = to_dataset(owner, fit_data, y, data_config=resolved)\n''',
)
replace_once(
    "src/bochan/tabular/optimizer/fitting.py",
    '''    owner.cross_validation_result_ = None\n    if run_cv:\n        owner.cross_validation_result_ = owner.bo.cross_validate(\n            dataset.X,\n            dataset.Y,\n            dataset.Yvar,\n            model_config=model_config,\n            fit_config=owner.fit_config,\n            cv_config=resolved_cv or CrossValidationConfig(),\n        )\n    resolved_failure_config = owner.observation.resolve_failure_config(failure_config)\n    if owner.observation.uses_observation_conversion(resolved):\n''',
    '''    owner.cross_validation_result_ = None\n    if run_cv:\n        if uses_observation_conversion:\n            owner.cross_validation_result_ = owner.bo.cross_validate_observations(\n                dataset.observation_data(),\n                model_config=model_config,\n                fit_config=owner.fit_config,\n                cv_config=resolved_cv or CrossValidationConfig(),\n            )\n        else:\n            owner.cross_validation_result_ = owner.bo.cross_validate(\n                dataset.X,\n                dataset.Y,\n                dataset.Yvar,\n                model_config=model_config,\n                fit_config=owner.fit_config,\n                cv_config=resolved_cv or CrossValidationConfig(),\n            )\n    resolved_failure_config = owner.observation.resolve_failure_config(failure_config)\n    if uses_observation_conversion:\n''',
)

# 4) Replace the legacy rejection test with a supported observation-aware CV test.
path = Path("tests/test_tabular_observation_states.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "from bochan.api import ExperimentFailureConfig, FitConfig\n",
    "from bochan.api import CrossValidationConfig, ExperimentFailureConfig, FitConfig\n",
    1,
)
old = r'''def test_tabular_observation_mode_rejects_generic_cross_validation() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.0],
            "strength": [1.0, None, 3.0],
        }
    )
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x"],
        target_cols=["strength"],
        target_missing_strategy="keep",
        skip_fit=True,
    )

    with pytest.raises(ValueError, match="observation-aware validation"):
        optimizer.fit(data, cross_validation=True)
'''
new = r'''def test_tabular_observation_mode_supports_cross_validation() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "strength": [1.0, 1.2, None, 1.6, 1.8, 2.0],
        }
    )
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x"],
        target_cols=["strength"],
        target_missing_strategy="keep",
        skip_fit=True,
    )

    optimizer.fit(
        data,
        cross_validation=True,
        cv_config=CrossValidationConfig(n_splits=2, shuffle=False),
    )

    result = optimizer.cross_validation_result_
    assert result is not None
    assert result.metadata["observation_aware"] is True
    assert result.metadata["n_objective_rows"] == 5
    assert result.output.oof_predictions.indices.tolist() == [0, 1, 3, 4, 5]
'''
if old not in text:
    raise SystemExit("legacy observation CV rejection test not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")

# 5) Focused Phase 5 tests.
Path("tests/test_material_train_yvar_phase5.py").write_text(
r'''from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.api import CrossValidationConfig, FitConfig, ModelConfig, ObservationData, PredictionResult
from bochan.api.evaluation.cross_validation import cross_validate_observations


class _FakeObservationCVOptimizer:
    captured_observations: list[ObservationData] = []

    def __init__(
        self,
        model_config: ModelConfig,
        fit_config: FitConfig | None = None,
        *,
        bounds=None,
        model_registry=None,
        acquisition_registry=None,
    ) -> None:
        self.model_config = model_config
        self.fit_config = fit_config
        self.bounds = bounds
        self.model_registry = model_registry
        self.acquisition_registry = acquisition_registry
        self.model = SimpleNamespace()

    def fit(
        self,
        *,
        observation_data: ObservationData,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
    ) -> _FakeObservationCVOptimizer:
        self.model_config = model_config or self.model_config
        self.fit_config = fit_config or self.fit_config
        self.observation_data = observation_data
        type(self).captured_observations.append(observation_data)
        return self

    def predict(self, X, *, return_result: bool = False):
        x = torch.as_tensor(X).reshape(-1, 1)
        mean = torch.cat((x + 1.0, 2.0 * x + 1.0), dim=-1)
        result = PredictionResult(
            posterior=None,
            mean=mean,
            variance=torch.full_like(mean, 0.04),
            task_type="multi_objective",
            variance_kind="predictive",
        )
        return result if return_result else mean


def _partial_known_noise_observations() -> ObservationData:
    X = torch.arange(8, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor(
        [
            [1.0, float("nan")],
            [2.0, 3.0],
            [float("nan"), 5.0],
            [4.0, 7.0],
            [5.0, float("nan")],
            [float("nan"), 11.0],
            [7.0, 13.0],
            [float("nan"), float("nan")],
        ],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [
            [0.10, float("nan")],
            [0.11, 0.21],
            [float("nan"), 0.22],
            [0.13, 0.23],
            [0.14, float("nan")],
            [float("nan"), 0.25],
            [0.16, 0.26],
            [float("nan"), float("nan")],
        ],
        dtype=torch.double,
    )
    return ObservationData.from_status(
        X,
        Y,
        status=["success"] * 6 + ["failed", "pending"],
        Yvar=Yvar,
    )


def test_observation_cv_scores_only_successful_observed_cells_and_keeps_yvar() -> None:
    _FakeObservationCVOptimizer.captured_observations = []
    optimizer = _FakeObservationCVOptimizer(
        ModelConfig(
            task_type="multi_objective",
            model_type="base",
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )
    observations = _partial_known_noise_observations()

    result = cross_validate_observations(
        optimizer,
        observations,
        cv_config=CrossValidationConfig(n_splits=3, shuffle=False),
    )

    assert result.metadata["observation_aware"] is True
    assert result.metadata["failure_model_evaluated"] is False
    assert result.metadata["known_observation_variance"] is True
    assert result.metadata["n_objective_rows"] == 6
    assert result.metadata["n_excluded_failed_rows"] == 1
    assert result.metadata["n_excluded_pending_rows"] == 1
    assert result.outputs["output_0"].oof_predictions.indices.tolist() == [0, 1, 3, 4]
    assert result.outputs["output_1"].oof_predictions.indices.tolist() == [1, 2, 3, 5]
    assert result.outputs["output_0"].oof_metrics["rmse"] == pytest.approx(0.0)
    assert result.outputs["output_1"].oof_metrics["rmse"] == pytest.approx(0.0)

    assert len(_FakeObservationCVOptimizer.captured_observations) == 3
    for fold_observations in _FakeObservationCVOptimizer.captured_observations:
        assert not bool(fold_observations.failed_mask.any())
        assert not bool(fold_observations.pending_mask.any())
        assert fold_observations.Yvar is not None
        valid = fold_observations.observed_mask
        assert bool(torch.isfinite(fold_observations.Yvar[valid]).all())
        assert bool((fold_observations.Yvar[valid] > 0.0).all())


def test_observation_cv_rejects_fold_without_training_values_for_an_output() -> None:
    X = torch.arange(4, dtype=torch.double).unsqueeze(-1)
    observations = ObservationData(
        X=X,
        Y=torch.tensor(
            [[1.0, 1.0], [2.0, 3.0], [3.0, float("nan")], [4.0, float("nan")]],
            dtype=torch.double,
        ),
    )
    optimizer = _FakeObservationCVOptimizer(
        ModelConfig(
            task_type="multi_objective",
            model_type="base",
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )

    with pytest.raises(ValueError, match="no observed values.*output_1"):
        cross_validate_observations(
            optimizer,
            observations,
            cv_config=CrossValidationConfig(n_splits=2, shuffle=False),
        )


def test_observation_cv_rejects_feature_importance_until_output_protocol_exists() -> None:
    optimizer = _FakeObservationCVOptimizer(
        ModelConfig(
            task_type="multi_objective",
            model_type="base",
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )

    with pytest.raises(ValueError, match="feature importance"):
        cross_validate_observations(
            optimizer,
            _partial_known_noise_observations(),
            cv_config=CrossValidationConfig(feature_importance_config=object()),
        )
''',
encoding="utf-8",
)

# 6) Phase 5 behavior note.
Path("docs/material_train_yvar_phase5.md").write_text(
r'''# Material `train_Yvar` Phase 5: observation-aware cross-validation

Phase 5 extends the observation-state workflow introduced in Phase 4 with an
objective-model cross-validation protocol for partial, pending, failed, and
known-noise observations.

## Protocol

- Folds are formed only from successful rows containing at least one observed
  objective cell.
- Failed and pending experiments are excluded from objective-model CV and are
  not treated as target observations.
- One row split is shared by all outputs, while each output is scored only on
  cells observed for that output.
- Known per-observation variance (`Yvar` / `target_variance_cols`) is sliced with
  the same fold rows and target-cell masks used for model fitting.
- Every fold must retain at least one training observation for every objective
  output. A clear error is raised otherwise.
- A fold with no validation observation for one output skips that output/fold
  metric; the other observed outputs remain valid.
- OOF predictions retain original row indices so sparse output coverage can be
  inspected directly.

## Tabular API

`cross_validation=True` now works with observation conversion such as
`target_missing_strategy="keep"`, `experiment_status_col`, and
`target_variance_cols`. Ordinary fully observed tabular CV keeps the existing
row-wise path unchanged.

## Scope boundary

This phase evaluates the objective model only. Failure/success classifiers are
not cross-validated together with the objective model. Fold-level feature
importance is also intentionally rejected for observation-aware CV until an
output-specific partial-target importance protocol is defined.
''',
encoding="utf-8",
)
