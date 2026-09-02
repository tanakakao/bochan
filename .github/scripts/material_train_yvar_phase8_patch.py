from pathlib import Path


PATH = Path("src/bochan/api/evaluation/cross_validation.py")
text = PATH.read_text(encoding="utf-8")


def replace_once(old: str, new: str, *, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    text = text.replace(old, new, 1)


replace_once(
    "    feature_importance: Any | None = None\n"
    "    failure_model: OutputCrossValidationResult | None = None\n",
    "    feature_importance: Any | None = None\n"
    "    failure_model: OutputCrossValidationResult | None = None\n"
    "    failure_feature_importance: Any | None = None\n",
    label="CrossValidationResult failure FI field",
)

replace_once(
    "    config = cv_config or CrossValidationConfig()\n"
    "    if config.feature_importance_config is not None:\n"
    "        raise ValueError(\n"
    "            \"Observation-aware cross-validation does not yet support fold feature \"\n"
    "            \"importance because partial target cells require an output-specific \"\n"
    "            \"importance protocol.\"\n"
    "        )\n\n"
    "    X, Y = observation_data.X, observation_data.Y\n",
    "    config = cv_config or CrossValidationConfig()\n\n"
    "    X, Y = observation_data.X, observation_data.Y\n",
    label="remove Phase 5 FI rejection",
)

helpers = r'''

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
'''

replace_once(
    "\ndef cross_validate_observations(\n",
    helpers + "\n\ndef cross_validate_observations(\n",
    label="insert Phase 8 FI helpers",
)

replace_once(
    "    per_output: dict[str, list[CVFoldResult]] = {name: [] for name in names}\n"
    "    failure_folds: list[CVFoldResult] = []\n"
    "    success_targets = observation_data.success_mask.to(dtype=Y.dtype)\n",
    "    per_output: dict[str, list[CVFoldResult]] = {name: [] for name in names}\n"
    "    objective_importances: dict[str, list[Any]] = {name: [] for name in names}\n"
    "    failure_folds: list[CVFoldResult] = []\n"
    "    failure_importances: list[Any] = []\n"
    "    success_targets = observation_data.success_mask.to(dtype=Y.dtype)\n",
    label="initialize observation FI collectors",
)

replace_once(
    "            failure_folds.append(\n"
    "                CVFoldResult(\n"
    "                    fold=fold,\n"
    "                    train_indices=train_indices.detach().cpu(),\n"
    "                    test_indices=test_indices.detach().cpu(),\n"
    "                    train_metrics=_failure_probability_metrics(\n"
    "                        failure_train.y_true,\n"
    "                        failure_train.predictive_mean,\n"
    "                        config,\n"
    "                        warnings,\n"
    "                        context=f\"training fold {fold}\",\n"
    "                    ),\n"
    "                    test_metrics=_failure_probability_metrics(\n"
    "                        failure_test.y_true,\n"
    "                        failure_test.predictive_mean,\n"
    "                        config,\n"
    "                        warnings,\n"
    "                        context=f\"validation fold {fold}\",\n"
    "                    ),\n"
    "                    train_predictions=failure_train,\n"
    "                    test_predictions=failure_test,\n"
    "                )\n"
    "            )\n",
    "            failure_importance = None\n"
    "            if config.feature_importance_config is not None:\n"
    "                bundle = getattr(fold_optimizer, \"bundle\", None)\n"
    "                cat_dims = tuple(getattr(bundle, \"cat_dims\", ()) or ())\n"
    "                failure_importance = _failure_feature_importance(\n"
    "                    failure_model,\n"
    "                    X,\n"
    "                    success_targets,\n"
    "                    test_indices,\n"
    "                    fold=fold,\n"
    "                    config=config,\n"
    "                    cat_dims=cat_dims,\n"
    "                )\n"
    "                failure_importances.append(failure_importance)\n"
    "            failure_folds.append(\n"
    "                CVFoldResult(\n"
    "                    fold=fold,\n"
    "                    train_indices=train_indices.detach().cpu(),\n"
    "                    test_indices=test_indices.detach().cpu(),\n"
    "                    train_metrics=_failure_probability_metrics(\n"
    "                        failure_train.y_true,\n"
    "                        failure_train.predictive_mean,\n"
    "                        config,\n"
    "                        warnings,\n"
    "                        context=f\"training fold {fold}\",\n"
    "                    ),\n"
    "                    test_metrics=_failure_probability_metrics(\n"
    "                        failure_test.y_true,\n"
    "                        failure_test.predictive_mean,\n"
    "                        config,\n"
    "                        warnings,\n"
    "                        context=f\"validation fold {fold}\",\n"
    "                    ),\n"
    "                    train_predictions=failure_train,\n"
    "                    test_predictions=failure_test,\n"
    "                    feature_importance=failure_importance,\n"
    "                )\n"
    "            )\n",
    label="attach failure FI",
)

replace_once(
    "            per_output[name].append(\n"
    "                CVFoldResult(\n"
    "                    fold=fold,\n"
    "                    train_indices=output_train_indices.detach().cpu(),\n"
    "                    test_indices=output_test_indices.detach().cpu(),\n",
    "            fold_importance = None\n"
    "            if config.feature_importance_config is not None:\n"
    "                fold_importance = _observation_output_feature_importance(\n"
    "                    fold_optimizer,\n"
    "                    X,\n"
    "                    Y,\n"
    "                    output_test_indices,\n"
    "                    output=output,\n"
    "                    name=name,\n"
    "                    task=task,\n"
    "                    fold=fold,\n"
    "                    config=config,\n"
    "                )\n"
    "                objective_importances[name].append(fold_importance)\n\n"
    "            per_output[name].append(\n"
    "                CVFoldResult(\n"
    "                    fold=fold,\n"
    "                    train_indices=output_train_indices.detach().cpu(),\n"
    "                    test_indices=output_test_indices.detach().cpu(),\n",
    label="compute objective FI",
)

replace_once(
    "                    train_predictions=train_pred,\n"
    "                    test_predictions=test_pred,\n"
    "                )\n"
    "            )\n\n"
    "    outputs: dict[str, OutputCrossValidationResult] = {}\n",
    "                    train_predictions=train_pred,\n"
    "                    test_predictions=test_pred,\n"
    "                    feature_importance=fold_importance,\n"
    "                )\n"
    "            )\n\n"
    "    outputs: dict[str, OutputCrossValidationResult] = {}\n",
    label="attach objective FI to fold",
)

replace_once(
    "    success_without_objective = (\n"
    "        observation_data.success_mask & ~observation_data.observed_mask.any(dim=-1)\n"
    "    )\n",
    "    objective_feature_importance = _combine_observation_feature_importance(\n"
    "        objective_importances\n"
    "    )\n"
    "    failure_feature_importance = (\n"
    "        _aggregate_feature_importance(failure_importances)\n"
    "        if failure_importances\n"
    "        else None\n"
    "    )\n\n"
    "    success_without_objective = (\n"
    "        observation_data.success_mask & ~observation_data.observed_mask.any(dim=-1)\n"
    "    )\n",
    label="aggregate Phase 8 FI",
)

replace_once(
    "        \"failure_model_evaluated\": failure_enabled,\n"
    "        \"known_observation_variance\": observation_data.Yvar is not None,\n",
    "        \"failure_model_evaluated\": failure_enabled,\n"
    "        \"feature_importance_evaluated\": config.feature_importance_config is not None,\n"
    "        \"feature_importance_protocol\": (\n"
    "            \"output_specific_successful_observed_validation_cells\"\n"
    "            if config.feature_importance_config is not None\n"
    "            else None\n"
    "        ),\n"
    "        \"known_observation_variance\": observation_data.Yvar is not None,\n",
    label="Phase 8 metadata",
)

replace_once(
    "                \"n_failure_validation_rows\": int(split_indices.numel()),\n"
    "            }\n"
    "        )\n",
    "                \"n_failure_validation_rows\": int(split_indices.numel()),\n"
    "                \"failure_feature_importance_evaluated\": (\n"
    "                    failure_feature_importance is not None\n"
    "                ),\n"
    "            }\n"
    "        )\n",
    label="failure FI metadata",
)

replace_once(
    "        feature_importance=None,\n"
    "        failure_model=failure_result,\n"
    "    )\n",
    "        feature_importance=objective_feature_importance,\n"
    "        failure_model=failure_result,\n"
    "        failure_feature_importance=failure_feature_importance,\n"
    "    )\n",
    label="return Phase 8 FI",
)

PATH.write_text(text, encoding="utf-8")
