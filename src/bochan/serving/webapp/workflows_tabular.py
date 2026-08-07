"""React Web workflow backed by :class:`TabularBayesianOptimizer`."""

from __future__ import annotations

import logging
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from bochan.api.candidate_uniqueness import count_unique_candidate_rows
from bochan.desktop.services import (
    _build_repair_config,
    _encode_features,
    _postprocess_candidates,
    _requires_best_f,
    _requires_beta,
)

from .logging import current_request_id, get_logger, log_event
from .search_settings import (
    botorch_linear_constraints,
    build_target_constraint_config,
    feature_constraint_results,
    normalize_feature_constraints,
    resolve_search_method,
)
from .tabular_backend import encoded_features_from_tabular, fit_tabular_optimizer
from .target_results import (
    _batch_acq_value,
    _build_feature_importance_visualizations,
    _build_visualizations,
    _candidate_rows,
    _figure_payload,
)
from .target_roles import (
    apply_target_roles,
    best_observed,
    level_set_thresholds,
    objective_values_direct,
    objective_weights,
    optimized_settings,
    optimized_targets,
    output_spec_kwargs,
    select_optimized_values,
    target_directions,
)
from .target_settings import (
    _as_2d,
    _build_outcome_constraint_config,
    _clean_rows,
    _encode_targets,
    _model_kwargs,
    _reference_point,
    _resolve_target_settings,
    _resolve_targets,
    _validate_columns,
)

LOGGER = get_logger("workflow")
_NATIVE_MULTITASK_MODEL_TYPE_ALIASES = {
    "multitask": "multitask",
    "beta_multitask": "beta_wide_multitask",
    "gamma_multitask": "gamma_wide_multitask",
    "poisson_multitask": "poisson_wide_multitask",
    "negative_binomial_multitask": "negative_binomial_wide_multitask",
}


def _acquisition_family(acqf_kwargs: dict[str, Any]) -> str:
    family = str(acqf_kwargs.pop("web_family", "bayesian_optimization")).lower()
    if family not in {
        "bayesian_optimization",
        "active_learning",
        "level_set_estimation",
    }:
        raise ValueError(f"Unknown acquisition family: {family!r}.")
    return family


def _normalized_acquisition_name(name: str) -> str:
    return "".join(character for character in str(name).lower() if character.isalnum())


def _set_active_learning_kwargs(
    acqf_kwargs: dict[str, Any],
    *,
    acq_key: str,
    train_x: Any,
    task_type: str,
    multi_output: bool,
    output_weights: Any | None = None,
) -> None:
    """Attach task/output-aware Active Learning constructor arguments."""
    task = str(task_type).lower()
    if multi_output:
        if output_weights is not None:
            acqf_kwargs.setdefault("output_weights", output_weights)
        if task == "regression":
            acqf_kwargs.setdefault("output_reduction", "weighted_mean")
        else:
            acqf_kwargs.setdefault("output_mode", "weighted_mean")

    if acq_key in {"nipv", "qnipv"}:
        acqf_kwargs.setdefault("mc_points", train_x)
        # Classification / ordinal NIPV implementations also expose observed
        # exclusion. True Gaussian regression NIPV does not accept X_observed.
        if task != "regression":
            acqf_kwargs.setdefault("X_observed", train_x)
        return

    acqf_kwargs.setdefault("X_observed", train_x)


def _set_active_learning_reference_kwargs(
    acqf_kwargs: dict[str, object],
    *,
    acq_key: str,
    train_x: object,
) -> None:
    """Backward-compatible Regression single-output AL reference helper."""
    _set_active_learning_kwargs(
        acqf_kwargs,
        acq_key=acq_key,
        train_x=train_x,
        task_type="regression",
        multi_output=False,
    )


def _request_with_constraints(request: Any, constraints: list[Any]) -> Any:
    if hasattr(request, "model_copy"):
        return request.model_copy(update={"constraints": constraints})
    values = dict(vars(request))
    values["constraints"] = constraints
    return SimpleNamespace(**values)


def _resolve_optimizer_value(name: str) -> Any:
    if name != "thompson_sampling":
        return name
    from bochan.optim import optimize_thompson_sampling

    return optimize_thompson_sampling


def _resolve_direct_multitask_model_type(model_type: str) -> str | None:
    """Resolve a Web multitask option to its shared-design wide model key."""

    return _NATIVE_MULTITASK_MODEL_TYPE_ALIASES.get(str(model_type))


def _is_direct_multitask_model(model_type: str) -> bool:
    """Return whether Web should fit one native correlated multitask model."""

    return _resolve_direct_multitask_model_type(model_type) is not None


def _direct_multitask_model_config_kwargs(
    *,
    model_type: str,
    input_transform_config: Any,
    outcome_transform: Any,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build kwargs for a native wide multi-output model."""

    return {
        "task_type": "multi_objective",
        "model_type": str(model_type),
        "cat_dims": None,
        "input_transform_config": input_transform_config,
        "outcome_transform": outcome_transform,
        "model_kwargs": model_kwargs,
    }


def _multi_objective_config(
    *,
    hybrid_model: bool,
    objective_targets: list[str],
    objective_settings: list[dict[str, Any]],
    target_columns: list[str],
    ObjectiveConfig: Any,
) -> Any:
    if hybrid_model:
        return ObjectiveConfig(
            mode="multi_output",
            outputs=objective_targets,
            directions=["maximize"] * len(objective_targets),
            weights=[1.0] * len(objective_targets),
        )
    return ObjectiveConfig(
        mode="multi_output",
        outputs=[target_columns.index(target) for target in objective_targets],
        directions=[str(setting["direction"]) for setting in objective_settings],
        weights=[1.0] * len(objective_targets),
        eq_targets=[
            float(setting["value"]) if setting["goal"] == "target" and not setting.get("legacy") else None
            for setting in objective_settings
        ],
    )


def _serializable_target_settings(
    target_settings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "target": setting["target"],
            "task_type": setting["task_type"],
            "optimize": bool(setting["optimize"]),
            "direction": setting["direction"],
            "goal": setting["goal"],
            "value": setting["value"],
            "target_class": setting.get("target_class"),
            "target_classes": setting.get("target_classes", []),
            "class_order": setting.get("class_order", []),
            "target_values": setting.get("target_values", []),
        }
        for setting in target_settings
    ]


def _response_task_type(
    target_settings: list[dict[str, Any]],
    target_columns: list[str],
) -> str:
    task_types = [setting["task_type"] for setting in target_settings]
    if len(set(task_types)) > 1:
        return "hybrid"
    if task_types[0] == "regression":
        return "multi_objective" if len(target_columns) > 1 else "regression"
    return str(task_types[0])


def _candidate_distance_tolerances(
    encoded: dict[str, Any],
    *,
    relative_distance: float,
) -> list[float]:
    """Resolve final-space minimum distances for every encoded feature."""

    ratio = float(relative_distance)
    if ratio < 0 or ratio > 1:
        raise ValueError("minimum_candidate_distance_ratio must be between 0 and 1.")
    lower = [float(value) for value in encoded["bounds"][0]]
    upper = [float(value) for value in encoded["bounds"][1]]
    categorical = {int(index) for index in encoded["cat_dims"]}
    fixed = {int(index) for index in encoded["fixed_features"]}
    steps = {int(index): abs(float(value)) for index, value in encoded["steps"].items()}

    tolerances: list[float] = []
    for index, (low, high) in enumerate(zip(lower, upper, strict=True)):
        if index in categorical or index in fixed:
            tolerances.append(0.0)
            continue
        range_distance = abs(high - low) * ratio
        resolution_distance = 0.5 * steps.get(index, 0.0)
        tolerances.append(max(range_distance, resolution_distance, 1e-12))
    return tolerances


def run_regression_web_workflow(request: Any, store: Any) -> dict[str, Any]:
    """Fit tabular target-specific models and generate constrained candidates."""

    import torch

    from bochan.api import (
        AcquisitionConfig,
        DataContext,
        FitConfig,
        InputTransformConfig,
        ModelConfig,
        MultiOutputConfig,
        ObjectiveConfig,
        OptimizeConfig,
        OutputConfig,
    )
    from bochan.serving.fastapi.converters import to_serializable

    workflow_started = perf_counter()
    timings_ms: dict[str, float] = {}
    request_id = current_request_id()

    target_columns, legacy_directions = _resolve_targets(request)
    target_settings, user_model_kwargs = _resolve_target_settings(
        request,
        target_columns=target_columns,
        directions=legacy_directions,
    )
    target_settings, user_model_kwargs = apply_target_roles(
        target_settings,
        user_model_kwargs,
        directions=legacy_directions,
    )
    objective_settings = optimized_settings(target_settings)
    objective_targets = optimized_targets(target_settings)
    directions = target_directions(target_settings)
    multi_objective = len(objective_targets) > 1

    acqf_kwargs = dict(request.acquisition.acqf_kwargs or {})
    acquisition_family = _acquisition_family(acqf_kwargs)
    requested_acq_name = str(request.acquisition.name)
    acq_key = _normalized_acquisition_name(requested_acq_name)

    search_method_name = str(request.optimizer.name or "normal")
    resolved_optimizer, optimizer_kwargs, search_requests_nsgaii = resolve_search_method(
        search_method_name,
        multi_objective=multi_objective,
    )
    use_nsgaii = search_requests_nsgaii or acq_key in {"nsgaii", "nsga2"}
    if use_nsgaii and acquisition_family != "bayesian_optimization":
        raise ValueError("NSGA-II is available only for multi-objective Bayesian optimization.")

    requested_model_type = str(request.model_type)
    model_type = "rrp" if requested_model_type == "robust" else requested_model_type
    display_model_type = "robust" if model_type == "rrp" else model_type

    preparation_started = perf_counter()
    record = store.get(request.dataset_id)
    data = record.data.copy()
    feature_columns = list(request.feature_columns)
    _validate_columns(data, feature_columns, target_columns)
    data = _clean_rows(
        data,
        feature_columns,
        target_columns,
        drop_missing=request.drop_missing,
    )

    # These helpers now provide Web search metadata only. The actual DataFrame-to-
    # tensor conversion and category encoding used for fitting is performed by
    # TabularBayesianOptimizer below.
    encoded_features = _encode_features(
        data=data,
        feature_columns=feature_columns,
        search_space=list(request.search_space or []),
    )
    encoded_targets, target_metadata = _encode_targets(data, target_settings)

    feature_constraints = normalize_feature_constraints(
        list(request.constraints or []),
        feature_columns=feature_columns,
    )
    processing_request = _request_with_constraints(request, feature_constraints)
    equality_constraints, inequality_constraints = botorch_linear_constraints(
        feature_constraints,
        feature_columns=feature_columns,
    )

    internal_tasks = [target_metadata[target]["internal_task"] for target in target_columns]
    direct_multitask_model_type = _resolve_direct_multitask_model_type(model_type)
    direct_multitask = direct_multitask_model_type is not None
    if direct_multitask:
        if acquisition_family != "bayesian_optimization":
            raise ValueError("multitask is currently available only with Bayesian optimization in the Web workbench.")
        if len(target_columns) < 2:
            raise ValueError("multitask requires at least two target columns.")
        if any(task != "regression" for task in internal_tasks):
            raise ValueError("The Web workbench currently enables multitask only for homogeneous regression targets.")
        if encoded_features["cat_dims"]:
            raise ValueError("multitask is not registered for mixed categorical inputs in multi-objective regression.")

    provisional_train_x = torch.as_tensor(encoded_features["X"], dtype=torch.double)
    provisional_bounds = torch.as_tensor(encoded_features["bounds"], dtype=torch.double)
    timings_ms["prepare"] = round(
        (perf_counter() - preparation_started) * 1000,
        3,
    )

    importance_settings = getattr(request, "feature_importance", None)
    importance_enabled = bool(importance_settings and importance_settings.enabled)
    importance_source = None
    effective_cv_config = dict(request.cv_config or {})
    if importance_enabled:
        importance_source = importance_settings.source
        if importance_source == "auto":
            importance_source = "cross_validation" if request.cross_validation else "training"
        if importance_source == "cross_validation" and not request.cross_validation:
            raise ValueError("Cross-validation feature importance requires cross_validation=true.")
        if importance_source == "cross_validation":
            effective_cv_config["feature_importance_config"] = importance_settings.config.model_dump()

    input_transform_config = InputTransformConfig(
        normalize=request.normalize,
        perturbation=request.input_perturbation,
        n_w=request.n_w,
        std=request.perturbation_std,
        bounds=provisional_bounds,
        categorical_idx=encoded_features["cat_dims"] or None,
    )

    if direct_multitask:
        assert direct_multitask_model_type is not None
        model_config = ModelConfig(
            **_direct_multitask_model_config_kwargs(
                model_type=direct_multitask_model_type,
                input_transform_config=input_transform_config,
                outcome_transform=request.outcome_transform,
                model_kwargs=_model_kwargs(
                    user_model_kwargs,
                    model_type=direct_multitask_model_type,
                    n_features=int(provisional_train_x.shape[1]),
                ),
            )
        )
        hybrid_model = False
    else:
        output_configs = []
        for target in target_columns:
            metadata = target_metadata[target]
            output_configs.append(
                OutputConfig(
                    task_type=str(metadata["internal_task"]),
                    model_type=model_type,
                    name=target,
                    model_kwargs=_model_kwargs(
                        user_model_kwargs,
                        model_type=model_type,
                        n_features=int(provisional_train_x.shape[1]),
                        target_meta=metadata,
                    ),
                    output_spec_kwargs=output_spec_kwargs(metadata),
                )
            )
        model_config = ModelConfig(
            task_type="hybrid",
            model_type=model_type,
            cat_dims=encoded_features["cat_dims"] or None,
            input_transform_config=input_transform_config,
            outcome_transform=request.outcome_transform,
            model_kwargs={},
            multi_output_config=MultiOutputConfig(
                output_configs=output_configs,
                output_names=target_columns,
                use_hybrid=True,
            ),
        )
        hybrid_model = True

    fit_config = FitConfig(
        maxiter=request.fit_maxiter,
        num_epochs=request.fit_maxiter,
    )

    fit_started = perf_counter()
    tabular_optimizer = fit_tabular_optimizer(
        data=data,
        feature_columns=feature_columns,
        target_columns=target_columns,
        encoded_features=encoded_features,
        target_metadata=target_metadata,
        model_config=model_config,
        fit_config=fit_config,
        cross_validation=request.cross_validation,
        cv_config=effective_cv_config or None,
    )
    dataset = tabular_optimizer.dataset
    if dataset is None or dataset.Y is None:
        raise RuntimeError("TabularBayesianOptimizer did not produce fitted X/Y data.")
    optimizer = tabular_optimizer.bo
    train_x = dataset.X
    train_y = dataset.Y
    bounds = dataset.bounds
    if bounds is None:
        raise RuntimeError("TabularBayesianOptimizer did not resolve search bounds.")
    encoded_features = encoded_features_from_tabular(
        tabular_optimizer,
        encoded_features,
    )
    timings_ms["fit"] = round((perf_counter() - fit_started) * 1000, 3)

    importance_result = None
    importance_warnings: list[str] = []
    importance_summary: list[dict[str, Any]] = []
    importance_visualizations: list[dict[str, Any]] = []
    model_diagnostics: dict[str, Any] = {}
    if importance_enabled:
        importance_started = perf_counter()
        try:
            if importance_source == "cross_validation":
                importance_result = getattr(
                    tabular_optimizer.cross_validation_result_,
                    "feature_importance",
                    None,
                )
                if importance_result is None:
                    raise RuntimeError("Cross-validation did not return feature importance.")
                from bochan.visualization import (
                    cross_validated_feature_importance_dataframe,
                )

                frames = [
                    cross_validated_feature_importance_dataframe(importance_result, output_name=name)
                    for name in importance_result.outputs
                ]
            else:
                importance_result = tabular_optimizer.feature_importance(
                    config=importance_settings.config.model_dump(exclude={"feature_groups"}),
                    feature_groups=[group.model_dump() for group in importance_settings.config.feature_groups],
                )
                frames = [
                    tabular_optimizer.feature_importance_dataframe(importance_result, output_name=name)
                    for name in importance_result.outputs
                ]
                model_diagnostics = {
                    name: output.model_diagnostics for name, output in importance_result.outputs.items()
                }
            for frame in frames:
                importance_summary.extend(frame.to_dict(orient="records"))
            importance_warnings.extend(getattr(importance_result, "warnings", []))
        except Exception as exc:
            if importance_settings.config.error_policy == "raise":
                raise
            importance_warnings.append(f"Feature importance failed: {exc}")
            importance_result = None
        timings_ms["feature_importance"] = round(
            (perf_counter() - importance_started) * 1000,
            3,
        )
        if importance_result is not None:
            view_started = perf_counter()
            importance_visualizations, view_warnings = _build_feature_importance_visualizations(
                importance_result,
                visualization_config=importance_settings.visualization,
            )
            importance_warnings.extend(view_warnings)
            timings_ms["feature_importance_visualization"] = round(
                (perf_counter() - view_started) * 1000,
                3,
            )

    if hybrid_model:
        objective_values_full = _as_2d(
            optimizer.model.posterior(train_x, output_mode="objective").mean,
            n_rows=int(train_x.shape[0]),
        ).detach()
    else:
        objective_values_full = objective_values_direct(
            train_y,
            target_settings,
        ).detach()
    objective_values = select_optimized_values(
        objective_values_full,
        target_columns=target_columns,
        objective_targets=objective_targets,
    )

    outcome_constraint_config = build_target_constraint_config(
        processing_request,
        target_settings=target_settings,
        target_metadata=target_metadata,
        target_columns=target_columns,
        directions=directions,
        hybrid_model=hybrid_model,
    )

    objective_config: Any | None
    acq_name = "nsgaii" if use_nsgaii else requested_acq_name
    acq_key = _normalized_acquisition_name(acq_name)

    if acquisition_family == "bayesian_optimization":
        if not multi_objective:
            if acq_key not in {"ei", "nei", "pi", "ucb"}:
                raise ValueError(f"Single-objective Bayesian optimization requires EI, PI, or UCB, got {acq_name}.")
            if _requires_best_f(acq_name.upper()) and "best_f" not in acqf_kwargs:
                acqf_kwargs["best_f"] = objective_values[:, 0].max()
            if _requires_beta(acq_name.upper()) and "beta" not in acqf_kwargs:
                acqf_kwargs["beta"] = request.acquisition.beta

            active_setting = objective_settings[0]
            active_target = objective_targets[0]
            if hybrid_model:
                objective_config = ObjectiveConfig(
                    mode="scalar",
                    output=active_target,
                    direction="maximize",
                )
            else:
                objective_config = ObjectiveConfig(
                    mode="scalar",
                    output=target_columns.index(active_target),
                    direction=str(active_setting["direction"]),
                    eq_target=(
                        float(active_setting["value"])
                        if active_setting["goal"] == "target" and not active_setting.get("legacy")
                        else None
                    ),
                )
            data_context = DataContext(
                X_baseline=train_x,
                Y_baseline=objective_values,
                best_f=objective_values[:, 0].max(),
            )
        else:
            if acq_key not in {
                "ehvi",
                "nehvi",
                "nparego",
                "qnparego",
                "nsgaii",
                "nsga2",
            }:
                raise ValueError(
                    f"Multi-objective Bayesian optimization requires EHVI, NEHVI, NParEGO, or NSGA-II, got {acq_name}."
                )
            ref_point = _reference_point(objective_values)
            partitioning = None
            if acq_key == "ehvi":
                from botorch.utils.multi_objective.box_decompositions.non_dominated import (
                    NondominatedPartitioning,
                )

                partitioning = NondominatedPartitioning(
                    ref_point=ref_point,
                    Y=objective_values,
                )
            objective_config = _multi_objective_config(
                hybrid_model=hybrid_model,
                objective_targets=objective_targets,
                objective_settings=objective_settings,
                target_columns=target_columns,
                ObjectiveConfig=ObjectiveConfig,
            )
            data_context = DataContext(
                X_baseline=train_x,
                Y_baseline=objective_values,
                ref_point=ref_point,
                partitioning=partitioning,
            )
    elif acquisition_family == "active_learning":
        if acq_key not in {
            "variance",
            "predictiveentropy",
            "bald",
            "nipv",
            "qnipv",
        }:
            raise ValueError(f"Active learning supports variance, predictive_entropy, BALD, or NIPV, got {acq_name}.")
        objective_config = None
        homogeneous_task = (
            internal_tasks[0]
            if internal_tasks and all(task == internal_tasks[0] for task in internal_tasks)
            else "hybrid"
        )
        _set_active_learning_kwargs(
            acqf_kwargs,
            acq_key=acq_key,
            train_x=train_x,
            task_type=homogeneous_task,
            multi_output=len(target_columns) > 1,
            output_weights=objective_weights(
                target_columns=target_columns,
                objective_targets=objective_targets,
            ),
        )
        data_context = DataContext(
            X_baseline=train_x,
            Y_baseline=objective_values,
        )
    else:
        if acq_key not in {"straddle", "boundaryvariance", "icu"}:
            raise ValueError(f"Level-set estimation supports straddle, boundary_variance, or ICU, got {acq_name}.")
        objective_config = None
        acqf_kwargs.setdefault(
            "thresholds",
            level_set_thresholds(
                target_columns=target_columns,
                target_metadata=target_metadata,
                objective_targets=objective_targets,
            ),
        )
        acqf_kwargs.setdefault(
            "output_weights",
            objective_weights(
                target_columns=target_columns,
                objective_targets=objective_targets,
            ),
        )
        acqf_kwargs.setdefault("output_reduction", "weighted_mean")
        acqf_kwargs.setdefault("X_observed", train_x)
        data_context = DataContext(
            X_baseline=train_x,
            Y_baseline=objective_values,
        )

    acq_config = AcquisitionConfig(
        name=acq_name,
        objective_config=objective_config,
        outcome_constraint_config=outcome_constraint_config,
        acqf_kwargs=acqf_kwargs,
    )
    repair_config = _build_repair_config(
        request=processing_request,
        encoded=encoded_features,
        bounds=bounds,
    )
    minimum_distance_ratio = float(getattr(request.optimizer, "minimum_candidate_distance_ratio", 1e-3))
    duplicate_tolerances = _candidate_distance_tolerances(
        encoded_features,
        relative_distance=minimum_distance_ratio,
    )

    def final_candidate_postprocess(value: Any) -> Any:
        return _postprocess_candidates(
            value,
            request=processing_request,
            encoded=encoded_features,
        )

    opt_config = OptimizeConfig(
        q=request.optimizer.q,
        num_restarts=request.optimizer.num_restarts,
        raw_samples=request.optimizer.raw_samples,
        sequential=request.optimizer.sequential,
        optimizer=_resolve_optimizer_value(resolved_optimizer),
        optimizer_kwargs=optimizer_kwargs,
        repair_config=repair_config,
        fixed_features=encoded_features["fixed_features"] or None,
        inequality_constraints=inequality_constraints or None,
        equality_constraints=equality_constraints or None,
        duplicate_tolerances=duplicate_tolerances,
        final_candidate_postprocess=final_candidate_postprocess,
    )

    candidate_started = perf_counter()
    candidate_result = tabular_optimizer.candidate(
        acq_config,
        opt_config,
        data_context=data_context,
        return_result=True,
    )
    raw_candidates = candidate_result.candidates
    raw_acq_value = candidate_result.acq_value
    candidates = _postprocess_candidates(
        raw_candidates,
        request=processing_request,
        encoded=encoded_features,
    )
    batch_acq_value = _batch_acq_value(
        raw_acq_value,
        int(candidates.shape[0]),
    )
    unique_candidate_count = count_unique_candidate_rows(
        candidates,
        tolerance=opt_config.duplicate_tolerance,
        tolerances=duplicate_tolerances,
    )
    uniqueness_warning = (
        None
        if unique_candidate_count == int(request.optimizer.q)
        else (
            f"要求した{request.optimizer.q}件のうち、最終実験条件で異なる候補は"
            f"{unique_candidate_count}件です。探索範囲、step、制約、または最小距離を"
            "見直してください。"
        )
    )
    timings_ms["candidate"] = round(
        (perf_counter() - candidate_started) * 1000,
        3,
    )

    prediction_started = perf_counter()
    rows = _candidate_rows(
        optimizer=optimizer,
        candidates=candidates,
        acq_value=raw_acq_value,
        encoded=encoded_features,
        target_columns=target_columns,
        target_settings=target_settings,
        target_metadata=target_metadata,
        hybrid_model=hybrid_model,
    )
    for row in rows:
        feature_results = feature_constraint_results(
            row["values"],
            feature_constraints,
        )
        row["constraints"].extend(feature_results)
        row["constraints_ok"] = bool(row["constraints_ok"]) and all(result["ok"] for result in feature_results)
    timings_ms["prediction"] = round(
        (perf_counter() - prediction_started) * 1000,
        3,
    )

    visualization_started = perf_counter()
    visualizations, visualization_warnings = _build_visualizations(
        optimizer=optimizer,
        train_x=train_x,
        original_targets=data[target_columns],
        target_columns=target_columns,
        target_metadata=target_metadata,
        hybrid_model=hybrid_model,
    )
    timings_ms["visualization"] = round(
        (perf_counter() - visualization_started) * 1000,
        3,
    )

    best_observed_by_target = best_observed(
        data[target_columns],
        encoded_targets,
        target_settings,
        target_metadata,
    )
    best_observed_value: float | dict[str, float]
    if len(target_columns) == 1:
        best_observed_value = best_observed_by_target[target_columns[0]]
    else:
        best_observed_value = best_observed_by_target

    task_type = _response_task_type(target_settings, target_columns)
    serializable_settings = _serializable_target_settings(target_settings)
    timings_ms["total"] = round(
        (perf_counter() - workflow_started) * 1000,
        3,
    )
    log_event(
        LOGGER,
        logging.INFO,
        "workflow_completed",
        "Tabular target workflow completed",
        dataset_id=record.dataset_id,
        model_type=display_model_type,
        task_type=task_type,
        target_columns=target_columns,
        optimized_targets=objective_targets,
        target_settings=serializable_settings,
        acquisition_family=acquisition_family,
        acquisition=acq_name,
        search_method=search_method_name,
        n_feature_constraints=len(feature_constraints),
        n_candidates=len(rows),
        optimizer_backend="TabularBayesianOptimizer",
        timings_ms=timings_ms,
    )

    first_target = objective_targets[0]
    internal_model_type = direct_multitask_model_type if direct_multitask else model_type
    return {
        "dataset_id": record.dataset_id,
        "dataset_name": record.name,
        "task_type": task_type,
        "model_type": display_model_type,
        "n_train": int(train_x.shape[0]),
        "n_features": int(train_x.shape[1]),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "target_column": first_target,
        "target_settings": serializable_settings,
        "directions": directions,
        "direction": directions[first_target],
        "outcome_constraints": [],
        "cat_dims": encoded_features["cat_dims"],
        "category_maps": to_serializable(encoded_features["category_maps"]),
        "target_metadata": to_serializable(target_metadata),
        "best_observed": best_observed_value,
        "best_observed_by_target": best_observed_by_target,
        "bounds": to_serializable(bounds),
        "raw_acq_value": to_serializable(raw_acq_value),
        "batch_acq_value": batch_acq_value,
        "candidates": rows,
        "visualizations": visualizations,
        "visualization_warnings": visualization_warnings,
        "feature_importance": to_serializable(importance_result),
        "feature_importance_source": importance_source,
        "feature_importance_summary": to_serializable(importance_summary),
        "feature_importance_visualizations": importance_visualizations,
        "feature_importance_warnings": list(dict.fromkeys(importance_warnings)),
        "model_diagnostics": to_serializable(model_diagnostics),
        "metadata": {
            "request_id": request_id,
            "dropped_rows": int(record.profile["n_rows"] - len(data)),
            "acquisition_family": acquisition_family,
            "acquisition": acq_name,
            "requested_acquisition": requested_acq_name,
            "optimizer": resolved_optimizer,
            "search_method": search_method_name,
            "optimizer_backend": "TabularBayesianOptimizer",
            "tabular_feature_names": list(dataset.feature_names),
            "tabular_target_names": list(dataset.target_names),
            "repair_enabled": repair_config is not None,
            "candidate_uniqueness": {
                "requested_q": int(request.optimizer.q),
                "unique_count": unique_candidate_count,
                "sequential": bool(request.optimizer.sequential),
                "minimum_distance_ratio": minimum_distance_ratio,
                "per_feature_tolerances": dict(zip(feature_columns, duplicate_tolerances, strict=True)),
                "warning": uniqueness_warning,
            },
            "n_feature_constraints": len(feature_constraints),
            "n_targets": len(target_columns),
            "n_optimized_targets": len(objective_targets),
            "target_columns": target_columns,
            "optimized_targets": objective_targets,
            "constraint_only_targets": [target for target in target_columns if target not in objective_targets],
            "target_settings": serializable_settings,
            "internal_tasks": internal_tasks,
            "internal_model_type": internal_model_type,
            "hybrid_model": hybrid_model,
            "timings_ms": timings_ms,
            "cross_validation": to_serializable(tabular_optimizer.cross_validation_result_),
        },
    }


__all__ = [
    "_build_outcome_constraint_config",
    "_figure_payload",
    "_resolve_target_settings",
    "_resolve_targets",
    "run_regression_web_workflow",
]
