"""Web workflow orchestration for single- and multi-objective regression."""

from __future__ import annotations

import json
import logging
import re
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

LOGGER = get_logger("workflow")


def _figure_payload(
    figure: Any,
    *,
    figure_id: str,
    title: str,
    description: str,
) -> dict[str, Any]:
    """Convert a Plotly figure into a JSON-safe web response payload."""

    figure.update_layout(
        title=title,
        autosize=True,
        width=None,
        margin=dict(l=60, r=30, t=70, b=60),
    )
    return {
        "id": figure_id,
        "title": title,
        "description": description,
        "figure": json.loads(figure.to_json()),
    }


def _safe_figure_id(value: str) -> str:
    """Return a stable HTML-friendly identifier fragment."""

    normalized = re.sub(r"[^0-9A-Za-z_-]+", "-", str(value)).strip("-")
    return normalized or "target"


def _resolve_targets(request: Any) -> tuple[list[str], dict[str, str]]:
    """Resolve backward-compatible target and direction settings."""

    target_columns = [str(value) for value in list(request.target_columns or [])]
    if not target_columns and request.target_column:
        target_columns = [str(request.target_column)]
    if not target_columns:
        raise ValueError("At least one target column is required.")
    if len(set(target_columns)) != len(target_columns):
        raise ValueError("target_columns must not contain duplicates.")

    requested_directions = dict(request.directions or {})
    directions = {
        target: str(requested_directions.get(target, request.direction or "maximize"))
        for target in target_columns
    }
    invalid = {
        target: direction
        for target, direction in directions.items()
        if direction not in {"maximize", "minimize"}
    }
    if invalid:
        raise ValueError(f"Invalid target directions: {invalid}")
    return target_columns, directions


def _validate_regression_columns(
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
) -> None:
    """Validate selected feature and numeric target columns."""

    import pandas as pd

    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    missing = [
        column
        for column in [*feature_columns, *target_columns]
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")
    overlap = sorted(set(feature_columns).intersection(target_columns))
    if overlap:
        raise ValueError(
            f"Target columns must not be included in feature columns: {overlap}"
        )
    non_numeric = [
        target
        for target in target_columns
        if not pd.api.types.is_numeric_dtype(data[target])
    ]
    if non_numeric:
        raise ValueError(f"Regression targets must be numeric: {non_numeric}")


def _clean_regression_rows(
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
    *,
    drop_missing: bool,
) -> Any:
    """Drop or reject rows missing any selected feature or target."""

    selected = [*feature_columns, *target_columns]
    if not drop_missing:
        if data[selected].isna().any().any():
            raise ValueError(
                "Missing values are present. Enable drop_missing or clean the dataset first."
            )
        return data.reset_index(drop=True)
    return data.dropna(subset=selected).reset_index(drop=True)


def _direction_signs(
    target_columns: list[str],
    directions: dict[str, str],
    *,
    dtype: Any,
    device: Any,
) -> Any:
    """Return +1 for maximize and -1 for minimize targets."""

    import torch

    return torch.as_tensor(
        [1.0 if directions[target] == "maximize" else -1.0 for target in target_columns],
        dtype=dtype,
        device=device,
    )


def _build_outcome_constraint_config(
    request: Any,
    *,
    target_columns: list[str],
    directions: dict[str, str],
) -> Any | None:
    """Convert original-scale target constraints to transformed model outputs."""

    constraints = list(request.outcome_constraints or [])
    if not constraints:
        return None

    from bochan.api import OutcomeConstraintConfig

    target_to_index = {target: index for index, target in enumerate(target_columns)}
    output_indices: list[int] = []
    operators: list[str] = []
    thresholds: list[float] = []
    for constraint in constraints:
        target = str(constraint.target)
        if target not in target_to_index:
            raise ValueError(
                f"Outcome constraint target is not selected: {target}"
            )
        operator = str(constraint.operator)
        threshold = float(constraint.value)
        if directions[target] == "minimize":
            operator = "le" if operator == ">=" else "ge"
            threshold = -threshold
        else:
            operator = "ge" if operator == ">=" else "le"
        output_indices.append(target_to_index[target])
        operators.append(operator)
        thresholds.append(threshold)

    return OutcomeConstraintConfig(
        output_indices=output_indices,
        operators=operators,
        thresholds=thresholds,
    )


def _model_kwargs(
    request: Any,
    *,
    model_type: str,
    n_features: int,
) -> dict[str, Any]:
    """Add safe defaults for models requiring a latent dimension."""

    kwargs = dict(request.model_kwargs or {})
    if model_type in {"pca", "rembo"}:
        kwargs.setdefault("n_components", max(1, min(2, int(n_features))))
    return kwargs


def _reference_point(train_y: Any) -> Any:
    """Build a dominated reference point in transformed maximize space."""

    import torch

    lower = train_y.min(dim=0).values
    upper = train_y.max(dim=0).values
    scale = (upper - lower).abs()
    fallback = torch.maximum(lower.abs(), upper.abs()).clamp_min(1.0)
    margin = torch.where(scale > 1e-12, scale * 0.1, fallback * 0.1)
    return lower - margin


def _broadcast_acq_values(acq_value: Any, n: int) -> list[float | None]:
    """Broadcast scalar acquisition values to candidate rows."""

    try:
        values = acq_value.detach().cpu().reshape(-1).tolist()
    except Exception:
        values = [acq_value]
    values = [float(value) for value in values if value is not None]
    if not values:
        return [None for _ in range(n)]
    if len(values) == 1:
        return values * n
    if len(values) < n:
        return values + [values[-1]] * (n - len(values))
    return values[:n]


def _constraint_results(
    predictions: dict[str, dict[str, float]],
    constraints: list[Any],
) -> list[dict[str, Any]]:
    """Evaluate displayed candidate means against original-scale constraints."""

    results: list[dict[str, Any]] = []
    tolerance = 1e-8
    for constraint in constraints:
        target = str(constraint.target)
        mean = float(predictions[target]["mean"])
        threshold = float(constraint.value)
        if constraint.operator == ">=":
            ok = mean >= threshold - tolerance
            violation = max(threshold - mean, 0.0)
        else:
            ok = mean <= threshold + tolerance
            violation = max(mean - threshold, 0.0)
        results.append(
            {
                "target": target,
                "operator": str(constraint.operator),
                "value": threshold,
                "predicted_mean": mean,
                "ok": bool(ok),
                "violation": float(violation),
            }
        )
    return results


def _candidate_rows_multi(
    *,
    candidates: Any,
    acq_value: Any,
    mean: Any,
    std: Any,
    encoded: dict[str, Any],
    request: Any,
    target_columns: list[str],
    directions: dict[str, str],
) -> list[dict[str, Any]]:
    """Build candidate rows containing predictions for every selected target."""

    from bochan.serving.fastapi.converters import to_serializable

    n_candidates = int(candidates.shape[0])
    flat_mean = mean.detach().cpu().reshape(n_candidates, -1)
    flat_std = std.detach().cpu().reshape(n_candidates, -1)
    if flat_mean.shape[1] < len(target_columns):
        raise RuntimeError(
            "Prediction output width is smaller than the selected target count: "
            f"{flat_mean.shape[1]} < {len(target_columns)}"
        )

    candidate_values = candidates.detach().cpu().tolist()
    acq_values = _broadcast_acq_values(acq_value, n_candidates)
    inverse_maps = encoded["inverse_category_maps"]
    feature_columns = encoded["feature_columns"]
    constraints = list(request.outcome_constraints or [])

    rows: list[dict[str, Any]] = []
    for row_index, values in enumerate(candidate_values):
        decoded: dict[str, Any] = {}
        encoded_values: dict[str, float] = {}
        for feature_index, column in enumerate(feature_columns):
            value = float(values[feature_index])
            encoded_values[column] = value
            if column in inverse_maps:
                decoded[column] = inverse_maps[column].get(
                    int(round(value)), str(int(round(value)))
                )
            else:
                decoded[column] = value

        predictions: dict[str, dict[str, float]] = {}
        for target_index, target in enumerate(target_columns):
            sign = 1.0 if directions[target] == "maximize" else -1.0
            predictions[target] = {
                "mean": float(flat_mean[row_index, target_index]) * sign,
                "std": float(flat_std[row_index, target_index]),
            }
        constraint_results = _constraint_results(predictions, constraints)
        first_prediction = predictions[target_columns[0]]
        rows.append(
            {
                "rank": row_index + 1,
                "values": decoded,
                "encoded_values": encoded_values,
                "acq_value": acq_values[row_index],
                "predictions": predictions,
                "predicted_target_mean": first_prediction["mean"],
                "predicted_target_std": first_prediction["std"],
                "constraints_ok": all(item["ok"] for item in constraint_results),
                "constraints": constraint_results,
                "raw": {"candidate": to_serializable(candidates[row_index])},
            }
        )
    return rows


def _build_target_visualizations(
    *,
    optimizer: Any,
    candidate_result: Any,
    candidates: Any,
    encoded: dict[str, Any],
    target_data: Any,
    target_columns: list[str],
    target_column: str,
    direction_sign: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build YY and response-surface figures for one output of a multi-output model."""

    import pandas as pd

    from bochan.visualization import (
        grid_1d_plot,
        grid_2d,
        prediction_dataframe,
        show_1dplot_with_pred,
        show_scatter_with_acqf,
        show_yyplot,
    )

    feature_columns = list(encoded["feature_columns"])
    train_x = optimizer.train_X
    if train_x is None:
        raise RuntimeError("Optimizer has no training inputs for visualization.")

    x_df = pd.DataFrame(train_x.detach().cpu().numpy(), columns=feature_columns)
    y_df = pd.DataFrame({target_column: target_data.to_numpy(dtype=float)})
    prefix = _safe_figure_id(target_column)
    figures: list[dict[str, Any]] = []
    warnings: list[str] = []

    pred_mean, pred_std = prediction_dataframe(
        optimizer,
        train_x,
        target_cols=target_columns,
    )
    pred_mean[target_column] = pred_mean[target_column] * direction_sign

    candidate_mean, candidate_std = prediction_dataframe(
        optimizer,
        candidates,
        target_cols=target_columns,
    )
    candidate_df = pd.DataFrame(
        candidates.detach().cpu().numpy(), columns=feature_columns
    )
    candidate_df[f"{target_column}_mean"] = (
        candidate_mean[target_column].to_numpy() * direction_sign
    )
    candidate_df[f"{target_column}_std"] = candidate_std[target_column].to_numpy()

    try:
        yy_figure = show_yyplot(
            y=y_df,
            target=target_column,
            preds=(pred_mean[[target_column]], pred_std[[target_column]]),
            df_cand=candidate_df,
        )
        figures.append(
            _figure_payload(
                yy_figure,
                figure_id=f"{prefix}-yyplot",
                title=f"{target_column}: 実測値と予測値",
                description="学習データに対する予測平均と不確かさを実測値と比較します。",
            )
        )
    except Exception as exc:
        warnings.append(f"{target_column}のYY plotを生成できませんでした: {exc}")
        LOGGER.warning(
            "YY plot generation failed",
            exc_info=True,
            extra={"event": "visualization_failed", "target_column": target_column},
        )

    fixed_indices = {int(index) for index in encoded["fixed_features"]}
    numeric_features = [
        feature_columns[int(index)]
        for index in encoded["numeric_indices"]
        if int(index) not in fixed_indices
    ]

    if numeric_features:
        feature = numeric_features[0]
        try:
            mean_1d, std_1d, x_grid = grid_1d_plot(
                optimizer,
                feature,
                feature_cols=feature_columns,
                target_cols=target_columns,
                n=80,
            )
            mean_1d[target_column] = mean_1d[target_column] * direction_sign
            figure_1d = show_1dplot_with_pred(
                feature=feature,
                target=target_column,
                data_1d_plot=(mean_1d, std_1d, x_grid),
                X=x_df,
                y=y_df,
                df_cand=candidate_df,
            )
            figures.append(
                _figure_payload(
                    figure_1d,
                    figure_id=f"{prefix}-prediction-1d",
                    title=f"{target_column}: {feature}に対する予測曲線",
                    description="他の説明変数を代表値に固定した予測平均、±1σ、入力データ、候補点です。",
                )
            )
        except Exception as exc:
            warnings.append(
                f"{target_column}の1次元予測グラフを生成できませんでした: {exc}"
            )
            LOGGER.warning(
                "One-dimensional visualization generation failed",
                exc_info=True,
                extra={"event": "visualization_failed", "target_column": target_column},
            )

    if len(numeric_features) >= 2:
        feature_1, feature_2 = numeric_features[:2]
        try:
            z_values, grid_1, grid_2 = grid_2d(
                optimizer,
                [feature_1, feature_2],
                target_col=target_column,
                feature_cols=feature_columns,
                target_cols=target_columns,
                candidate_result=candidate_result,
                n=40,
                show_type="pred",
            )
            z_values = z_values * direction_sign
            figure_2d = show_scatter_with_acqf(
                feature_col1=feature_1,
                feature_col2=feature_2,
                target_col=target_column,
                data_2d_plot=(z_values, grid_1, grid_2),
                X=x_df,
                y=y_df,
                df_cand=candidate_df,
                show_type="pred",
            )
            figures.append(
                _figure_payload(
                    figure_2d,
                    figure_id=f"{prefix}-prediction-2d",
                    title=f"{target_column}: {feature_1} × {feature_2} 予測分布",
                    description="他の説明変数を代表値に固定した予測平均の等高線に入力データと候補点を重ねます。",
                )
            )
        except Exception as exc:
            warnings.append(
                f"{target_column}の2次元予測グラフを生成できませんでした: {exc}"
            )
            LOGGER.warning(
                "Two-dimensional visualization generation failed",
                exc_info=True,
                extra={"event": "visualization_failed", "target_column": target_column},
            )

    return figures, warnings


def _build_regression_visualizations(
    *,
    optimizer: Any,
    candidate_result: Any,
    candidates: Any,
    encoded: dict[str, Any],
    targets: Any,
    target_columns: list[str],
    directions: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build result figures for every selected target."""

    figures: list[dict[str, Any]] = []
    warnings: list[str] = []
    for target in target_columns:
        sign = 1.0 if directions[target] == "maximize" else -1.0
        try:
            target_figures, target_warnings = _build_target_visualizations(
                optimizer=optimizer,
                candidate_result=candidate_result,
                candidates=candidates,
                encoded=encoded,
                target_data=targets[target],
                target_columns=target_columns,
                target_column=target,
                direction_sign=sign,
            )
            figures.extend(target_figures)
            warnings.extend(target_warnings)
        except Exception as exc:
            warnings.append(f"{target}の可視化を初期化できませんでした: {exc}")
            LOGGER.exception(
                "Visualization initialization failed",
                extra={"event": "visualization_batch_failed", "target_column": target},
            )
    return figures, warnings


def run_regression_web_workflow(request: Any, store: Any) -> dict[str, Any]:
    """Fit regression outputs, generate candidates, and create result figures."""

    import pandas as pd
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
    )
    from bochan.serving.fastapi.converters import to_serializable

    workflow_started = perf_counter()
    timings_ms: dict[str, float] = {}
    request_id = current_request_id()
    target_columns, directions = _resolve_targets(request)
    model_type = "rrp" if str(request.model_type) == "robust" else str(request.model_type)
    display_model_type = "robust" if model_type == "rrp" else model_type
    task_type = "multi_objective" if len(target_columns) > 1 else "regression"

    log_event(
        LOGGER,
        logging.INFO,
        "workflow_started",
        "Regression workflow started",
        dataset_id=request.dataset_id,
        model_type=display_model_type,
        task_type=task_type,
        target_columns=target_columns,
        directions=directions,
        acquisition=request.acquisition.name,
        optimizer=request.optimizer.name,
        q=request.optimizer.q,
    )

    preparation_started = perf_counter()
    record = store.get(request.dataset_id)
    data = record.data.copy()
    feature_columns = list(request.feature_columns)
    _validate_regression_columns(data, feature_columns, target_columns)
    data = _clean_regression_rows(
        data,
        feature_columns,
        target_columns,
        drop_missing=request.drop_missing,
    )
    encoded = _encode_features(
        data=data,
        feature_columns=feature_columns,
        search_space=list(request.search_space or []),
    )
    if model_type == "multitask":
        if len(target_columns) < 2:
            raise ValueError("multitask requires at least two target columns.")
        if encoded["cat_dims"]:
            raise ValueError(
                "multitask is not registered for mixed categorical inputs in multi-objective regression."
            )

    targets = data[target_columns].apply(pd.to_numeric, errors="coerce")
    if targets.isna().any().any():
        invalid = [target for target in target_columns if targets[target].isna().any()]
        raise ValueError(
            f"Target columns contain non-numeric values after conversion: {invalid}"
        )

    train_x = torch.as_tensor(encoded["X"], dtype=torch.double)
    signs = _direction_signs(
        target_columns,
        directions,
        dtype=train_x.dtype,
        device=train_x.device,
    )
    train_y_original = torch.as_tensor(
        targets.to_numpy(dtype=float), dtype=torch.double
    )
    train_y = train_y_original * signs
    bounds = torch.as_tensor(encoded["bounds"], dtype=torch.double)
    timings_ms["prepare"] = round(
        (perf_counter() - preparation_started) * 1000, 3
    )

    multi_output_config = None
    if len(target_columns) > 1 and model_type != "multitask":
        multi_output_config = MultiOutputConfig(output_names=target_columns)

    model_config = ModelConfig(
        task_type=task_type,
        model_type=model_type,
        cat_dims=encoded["cat_dims"] or None,
        input_transform_config=InputTransformConfig(
            normalize=request.normalize,
            perturbation=request.input_perturbation,
            n_w=request.n_w,
            std=request.perturbation_std,
            bounds=bounds,
            categorical_idx=encoded["cat_dims"] or None,
        ),
        outcome_transform=request.outcome_transform,
        model_kwargs=_model_kwargs(
            request,
            model_type=model_type,
            n_features=int(train_x.shape[1]),
        ),
        multi_output_config=multi_output_config,
    )
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
    log_event(
        LOGGER,
        logging.INFO,
        "model_fit_started",
        "Model fitting started",
        model_type=display_model_type,
        task_type=task_type,
        n_train=int(train_x.shape[0]),
        n_features=int(train_x.shape[1]),
        n_targets=len(target_columns),
        fit_maxiter=request.fit_maxiter,
    )
    optimizer.fit(train_x, train_y)
    timings_ms["fit"] = round((perf_counter() - fit_started) * 1000, 3)

    acq_name = str(request.acquisition.name).upper()
    acqf_kwargs = dict(request.acquisition.acqf_kwargs or {})
    outcome_constraint_config = _build_outcome_constraint_config(
        request,
        target_columns=target_columns,
        directions=directions,
    )
    data_context = DataContext(
        X_baseline=train_x,
        Y_baseline=train_y,
        best_f=train_y[:, 0].max(),
    )

    if len(target_columns) == 1:
        if acq_name not in {"EI", "NEI", "UCB"}:
            raise ValueError(
                f"Single-objective regression requires EI, NEI, or UCB, got {acq_name}."
            )
        if _requires_best_f(acq_name) and "best_f" not in acqf_kwargs:
            acqf_kwargs["best_f"] = train_y.max()
        if _requires_beta(acq_name) and "beta" not in acqf_kwargs:
            acqf_kwargs["beta"] = request.acquisition.beta
        acq_config = AcquisitionConfig(
            name=acq_name,
            objective_config=ObjectiveConfig(
                mode="scalar",
                output=0,
                direction="maximize",
            ),
            outcome_constraint_config=outcome_constraint_config,
            acqf_kwargs=acqf_kwargs,
        )
    else:
        if acq_name not in {"EHVI", "NEHVI"}:
            raise ValueError(
                f"Multi-objective regression requires EHVI or NEHVI, got {acq_name}."
            )
        ref_point = _reference_point(train_y)
        partitioning = None
        if acq_name == "EHVI":
            from botorch.utils.multi_objective.box_decompositions.non_dominated import (
                NondominatedPartitioning,
            )

            partitioning = NondominatedPartitioning(
                ref_point=ref_point,
                Y=train_y,
            )
        data_context = DataContext(
            X_baseline=train_x,
            Y_baseline=train_y,
            ref_point=ref_point,
            partitioning=partitioning,
        )
        acq_config = AcquisitionConfig(
            name=acq_name,
            objective_config=ObjectiveConfig(
                mode="multi_output",
                outputs=list(range(len(target_columns))),
                directions=["maximize"] * len(target_columns),
                weights=[1.0] * len(target_columns),
            ),
            outcome_constraint_config=outcome_constraint_config,
            acqf_kwargs=acqf_kwargs,
        )

    repair_config = _build_repair_config(
        request=request,
        encoded=encoded,
        bounds=bounds,
    )
    opt_config = OptimizeConfig(
        q=request.optimizer.q,
        num_restarts=request.optimizer.num_restarts,
        raw_samples=request.optimizer.raw_samples,
        sequential=request.optimizer.sequential,
        optimizer=request.optimizer.name,
        repair_config=repair_config,
        fixed_features=encoded["fixed_features"] or None,
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
        encoded=encoded,
    )
    timings_ms["candidate"] = round(
        (perf_counter() - candidate_started) * 1000, 3
    )

    prediction_started = perf_counter()
    mean, variance = optimizer.predict(candidates, return_type="mean_variance")
    std = variance.clamp_min(0).sqrt()
    rows = _candidate_rows_multi(
        candidates=candidates,
        acq_value=raw_acq_value,
        mean=mean,
        std=std,
        encoded=encoded,
        request=request,
        target_columns=target_columns,
        directions=directions,
    )
    timings_ms["prediction"] = round(
        (perf_counter() - prediction_started) * 1000, 3
    )

    visualization_started = perf_counter()
    visualizations, visualization_warnings = _build_regression_visualizations(
        optimizer=optimizer,
        candidate_result=candidate_result,
        candidates=candidates,
        encoded=encoded,
        targets=targets,
        target_columns=target_columns,
        directions=directions,
    )
    timings_ms["visualization"] = round(
        (perf_counter() - visualization_started) * 1000, 3
    )

    best_observed_by_target = {
        target: (
            float(targets[target].max())
            if directions[target] == "maximize"
            else float(targets[target].min())
        )
        for target in target_columns
    }
    best_observed: float | dict[str, float]
    if len(target_columns) == 1:
        best_observed = best_observed_by_target[target_columns[0]]
    else:
        best_observed = best_observed_by_target

    timings_ms["total"] = round(
        (perf_counter() - workflow_started) * 1000, 3
    )
    log_event(
        LOGGER,
        logging.INFO,
        "workflow_completed",
        "Regression workflow completed",
        dataset_id=record.dataset_id,
        model_type=display_model_type,
        task_type=task_type,
        target_columns=target_columns,
        acquisition=acq_name,
        n_candidates=len(rows),
        n_visualizations=len(visualizations),
        n_visualization_warnings=len(visualization_warnings),
        timings_ms=timings_ms,
    )

    serialized_constraints = [
        constraint.model_dump() if hasattr(constraint, "model_dump") else dict(constraint)
        for constraint in list(request.outcome_constraints or [])
    ]
    first_target = target_columns[0]
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
        "directions": directions,
        "direction": directions[first_target],
        "outcome_constraints": serialized_constraints,
        "cat_dims": encoded["cat_dims"],
        "category_maps": encoded["category_maps"],
        "best_observed": best_observed,
        "best_observed_by_target": best_observed_by_target,
        "bounds": to_serializable(bounds),
        "raw_acq_value": to_serializable(raw_acq_value),
        "candidates": rows,
        "visualizations": visualizations,
        "visualization_warnings": visualization_warnings,
        "metadata": {
            "request_id": request_id,
            "dropped_rows": int(record.profile["n_rows"] - len(data)),
            "acquisition": acq_name,
            "optimizer": request.optimizer.name,
            "repair_enabled": repair_config is not None,
            "n_targets": len(target_columns),
            "target_columns": target_columns,
            "directions": directions,
            "outcome_constraints": serialized_constraints,
            "internal_model_type": model_type,
            "timings_ms": timings_ms,
        },
    }


__all__ = [
    "_build_outcome_constraint_config",
    "_figure_payload",
    "_resolve_targets",
    "run_regression_web_workflow",
]
