"""Web workflow orchestration for heterogeneous target optimization."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from bochan.desktop.services import (
    _build_repair_config,
    _encode_features,
    _postprocess_candidates,
    _requires_best_f,
    _requires_beta,
)

from .logging import current_request_id, get_logger, log_event
from .target_results import _build_visualizations, _candidate_rows, _figure_payload
from .target_roles import (
    apply_target_roles,
    best_observed,
    build_target_constraint_config,
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
    return str(name).replace("-", "_").replace(" ", "_").lower()


def run_regression_web_workflow(request: Any, store: Any) -> dict[str, Any]:
    """Fit target-specific models, generate candidates, and serialize results."""

    import torch

    from bochan.api import (
        AcquisitionConfig,
        BayesianOptimizer,
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

    acqf_kwargs = dict(request.acquisition.acqf_kwargs or {})
    acquisition_family = _acquisition_family(acqf_kwargs)
    acq_name = str(request.acquisition.name)
    acq_key = _normalized_acquisition_name(acq_name)

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
    encoded_features = _encode_features(
        data=data,
        feature_columns=feature_columns,
        search_space=list(request.search_space or []),
    )
    encoded_targets, target_metadata = _encode_targets(data, target_settings)

    internal_tasks = [
        target_metadata[target]["internal_task"] for target in target_columns
    ]
    direct_multitask = model_type == "multitask"
    if direct_multitask:
        if acquisition_family != "bayesian_optimization":
            raise ValueError(
                "multitask is currently available only with Bayesian optimization "
                "in the Web workbench."
            )
        if len(target_columns) < 2:
            raise ValueError("multitask requires at least two target columns.")
        if any(task != "regression" for task in internal_tasks):
            raise ValueError(
                "The Web workbench currently enables multitask only for "
                "homogeneous regression targets."
            )
        if encoded_features["cat_dims"]:
            raise ValueError(
                "multitask is not registered for mixed categorical inputs in "
                "multi-objective regression."
            )

    train_x = torch.as_tensor(encoded_features["X"], dtype=torch.double)
    train_y = torch.as_tensor(
        encoded_targets.to_numpy(dtype=float),
        dtype=torch.double,
    )
    bounds = torch.as_tensor(encoded_features["bounds"], dtype=torch.double)
    timings_ms["prepare"] = round(
        (perf_counter() - preparation_started) * 1000,
        3,
    )

    input_transform_config = InputTransformConfig(
        normalize=request.normalize,
        perturbation=request.input_perturbation,
        n_w=request.n_w,
        std=request.perturbation_std,
        bounds=bounds,
        categorical_idx=encoded_features["cat_dims"] or None,
    )

    if direct_multitask:
        model_config = ModelConfig(
            task_type="multi_objective",
            model_type="multitask",
            cat_dims=None,
            input_transform_config=input_transform_config,
            outcome_transform=request.outcome_transform,
            model_kwargs=_model_kwargs(
                user_model_kwargs,
                model_type=model_type,
                n_features=int(train_x.shape[1]),
            ),
        )
        hybrid_model = False
    else:
        output_configs = []
        for target in target_columns:
            meta = target_metadata[target]
            output_configs.append(
                OutputConfig(
                    task_type=str(meta["internal_task"]),
                    model_type=model_type,
                    name=target,
                    model_kwargs=_model_kwargs(
                        user_model_kwargs,
                        model_type=model_type,
                        n_features=int(train_x.shape[1]),
                        target_meta=meta,
                    ),
                    output_spec_kwargs=output_spec_kwargs(meta),
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
    optimizer = BayesianOptimizer(
        model_config=model_config,
        fit_config=fit_config,
        bounds=bounds,
    )

    fit_started = perf_counter()
    optimizer.fit(train_x, train_y)
    timings_ms["fit"] = round((perf_counter() - fit_started) * 1000, 3)

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
        request,
        target_settings=target_settings,
        target_metadata=target_metadata,
        target_columns=target_columns,
        directions=directions,
        hybrid_model=hybrid_model,
    )

    objective_config: Any | None
    if acquisition_family == "bayesian_optimization":
        if len(objective_targets) == 1:
            if acq_key not in {"ei", "nei", "ucb"}:
                raise ValueError(
                    "Single-objective Bayesian optimization requires EI, NEI, or "
                    f"UCB, got {acq_name}."
                )
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
                        if active_setting["goal"] == "target"
                        and not active_setting.get("legacy")
                        else None
                    ),
                )
            data_context = DataContext(
                X_baseline=train_x,
                Y_baseline=objective_values,
                best_f=objective_values[:, 0].max(),
            )
        else:
            if acq_key not in {"ehvi", "nehvi"}:
                raise ValueError(
                    "Multi-objective Bayesian optimization requires EHVI or NEHVI, "
                    f"got {acq_name}."
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
            if hybrid_model:
                objective_config = ObjectiveConfig(
                    mode="multi_output",
                    outputs=objective_targets,
                    directions=["maximize"] * len(objective_targets),
                    weights=[1.0] * len(objective_targets),
                )
            else:
                objective_config = ObjectiveConfig(
                    mode="multi_output",
                    outputs=[
                        target_columns.index(target) for target in objective_targets
                    ],
                    directions=[
                        str(setting["direction"])
                        for setting in objective_settings
                    ],
                    weights=[1.0] * len(objective_targets),
                    eq_targets=[
                        float(setting["value"])
                        if setting["goal"] == "target"
                        and not setting.get("legacy")
                        else None
                        for setting in objective_settings
                    ],
                )
            data_context = DataContext(
                X_baseline=train_x,
                Y_baseline=objective_values,
                ref_point=ref_point,
                partitioning=partitioning,
            )
    elif acquisition_family == "active_learning":
        if acq_key not in {"variance", "predictive_entropy", "bald"}:
            raise ValueError(
                "Active learning supports variance, predictive_entropy, or BALD, "
                f"got {acq_name}."
            )
        objective_config = None
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
    else:
        if acq_key not in {"straddle", "boundary_variance"}:
            raise ValueError(
                "Level-set estimation supports straddle or boundary_variance, "
                f"got {acq_name}."
            )
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
        request=request,
        encoded=encoded_features,
        bounds=bounds,
    )
    opt_config = OptimizeConfig(
        q=request.optimizer.q,
        num_restarts=request.optimizer.num_restarts,
        raw_samples=request.optimizer.raw_samples,
        sequential=request.optimizer.sequential,
        optimizer=request.optimizer.name,
        repair_config=repair_config,
        fixed_features=encoded_features["fixed_features"] or None,
    )

    candidate_started = perf_counter()
    candidate_result = optimizer.candidate(
        acq_config,
        opt_config,
        data_context=data_context,
        return_result=True,
    )
    raw_candidates = candidate_result.candidates
    raw_acq_value = candidate_result.acq_value
    candidates = _postprocess_candidates(
        raw_candidates,
        request=request,
        encoded=encoded_features,
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

    task_types = [setting["task_type"] for setting in target_settings]
    if len(set(task_types)) > 1:
        task_type = "hybrid"
    elif task_types[0] == "regression":
        task_type = "multi_objective" if len(target_columns) > 1 else "regression"
    else:
        task_type = task_types[0]

    serializable_settings = [
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
    timings_ms["total"] = round(
        (perf_counter() - workflow_started) * 1000,
        3,
    )
    log_event(
        LOGGER,
        logging.INFO,
        "workflow_completed",
        "Target workflow completed",
        dataset_id=record.dataset_id,
        model_type=display_model_type,
        task_type=task_type,
        target_columns=target_columns,
        optimized_targets=objective_targets,
        target_settings=serializable_settings,
        acquisition_family=acquisition_family,
        acquisition=acq_name,
        n_candidates=len(rows),
        timings_ms=timings_ms,
    )

    first_target = objective_targets[0]
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
        "category_maps": encoded_features["category_maps"],
        "target_metadata": to_serializable(target_metadata),
        "best_observed": best_observed_value,
        "best_observed_by_target": best_observed_by_target,
        "bounds": to_serializable(bounds),
        "raw_acq_value": to_serializable(raw_acq_value),
        "candidates": rows,
        "visualizations": visualizations,
        "visualization_warnings": visualization_warnings,
        "metadata": {
            "request_id": request_id,
            "dropped_rows": int(record.profile["n_rows"] - len(data)),
            "acquisition_family": acquisition_family,
            "acquisition": acq_name,
            "optimizer": request.optimizer.name,
            "repair_enabled": repair_config is not None,
            "n_targets": len(target_columns),
            "n_optimized_targets": len(objective_targets),
            "target_columns": target_columns,
            "optimized_targets": objective_targets,
            "constraint_only_targets": [
                target for target in target_columns if target not in objective_targets
            ],
            "target_settings": serializable_settings,
            "internal_tasks": internal_tasks,
            "internal_model_type": model_type,
            "hybrid_model": hybrid_model,
            "timings_ms": timings_ms,
        },
    }


__all__ = [
    "_build_outcome_constraint_config",
    "_figure_payload",
    "_resolve_target_settings",
    "_resolve_targets",
    "run_regression_web_workflow",
]
