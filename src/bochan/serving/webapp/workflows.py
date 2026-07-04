"""Web-specific workflow orchestration with Plotly visualization output."""

from __future__ import annotations

import json
from typing import Any

from bochan.desktop.services import (
    _build_repair_config,
    _candidate_rows,
    _clean_regression_rows,
    _encode_features,
    _postprocess_candidates,
    _requires_best_f,
    _requires_beta,
    _validate_regression_columns,
)


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


def _build_regression_visualizations(
    *,
    optimizer: Any,
    candidate_result: Any,
    candidates: Any,
    encoded: dict[str, Any],
    target: Any,
    target_column: str,
    direction_sign: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build regression result figures through :mod:`bochan.visualization`."""

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
    train_X = optimizer.train_X
    if train_X is None:
        raise RuntimeError("Optimizer has no training inputs for visualization.")

    X_df = pd.DataFrame(train_X.detach().cpu().numpy(), columns=feature_columns)
    y_df = pd.DataFrame({target_column: target.to_numpy(dtype=float)})

    pred_mean, pred_std = prediction_dataframe(
        optimizer,
        train_X,
        target_cols=[target_column],
    )
    pred_mean[target_column] = pred_mean[target_column] * direction_sign

    candidate_mean, candidate_std = prediction_dataframe(
        optimizer,
        candidates,
        target_cols=[target_column],
    )
    candidate_df = pd.DataFrame(
        candidates.detach().cpu().numpy(),
        columns=feature_columns,
    )
    candidate_df[f"{target_column}_mean"] = (
        candidate_mean[target_column].to_numpy() * direction_sign
    )
    candidate_df[f"{target_column}_std"] = candidate_std[target_column].to_numpy()

    figures: list[dict[str, Any]] = []
    warnings: list[str] = []

    try:
        yy_figure = show_yyplot(
            y=y_df,
            target=target_column,
            preds=(pred_mean, pred_std),
            df_cand=candidate_df,
        )
        figures.append(
            _figure_payload(
                yy_figure,
                figure_id="yyplot",
                title="実測値と予測値",
                description="学習データに対する予測平均と不確かさを、実測値と比較します。",
            )
        )
    except Exception as exc:
        warnings.append(f"YY plotを生成できませんでした: {exc}")

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
                target_cols=[target_column],
                n=80,
            )
            mean_1d[target_column] = mean_1d[target_column] * direction_sign
            figure_1d = show_1dplot_with_pred(
                feature=feature,
                target=target_column,
                data_1d_plot=(mean_1d, std_1d, x_grid),
                X=X_df,
                y=y_df,
                df_cand=candidate_df,
            )
            figures.append(
                _figure_payload(
                    figure_1d,
                    figure_id="prediction-1d",
                    title=f"{feature}に対する予測曲線",
                    description="他の説明変数を代表値に固定したときの予測平均、±1σ、入力データ、候補点です。",
                )
            )
        except Exception as exc:
            warnings.append(f"1次元予測グラフを生成できませんでした: {exc}")

    if len(numeric_features) >= 2:
        feature_1, feature_2 = numeric_features[:2]
        try:
            z_values, grid_1, grid_2 = grid_2d(
                optimizer,
                [feature_1, feature_2],
                target_col=target_column,
                feature_cols=feature_columns,
                target_cols=[target_column],
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
                X=X_df,
                y=y_df,
                df_cand=candidate_df,
                show_type="pred",
            )
            figures.append(
                _figure_payload(
                    figure_2d,
                    figure_id="prediction-2d",
                    title=f"{feature_1} × {feature_2} 予測分布",
                    description="他の説明変数を代表値に固定した予測平均の等高線に、入力データと候補点を重ねます。",
                )
            )
        except Exception as exc:
            warnings.append(f"2次元予測グラフを生成できませんでした: {exc}")

    return figures, warnings


def run_regression_web_workflow(request: Any, store: Any) -> dict[str, Any]:
    """Fit a regression model, generate candidates, and create result figures."""

    import pandas as pd
    import torch

    from bochan.api import (
        AcquisitionConfig,
        BayesianOptimizer,
        FitConfig,
        InputTransformConfig,
        ModelConfig,
        OptimizeConfig,
    )
    from bochan.serving.fastapi.converters import to_serializable

    record = store.get(request.dataset_id)
    data = record.data.copy()
    feature_columns = list(request.feature_columns)
    target_column = request.target_column

    _validate_regression_columns(data, feature_columns, target_column)
    data = _clean_regression_rows(
        data,
        feature_columns,
        target_column,
        drop_missing=request.drop_missing,
    )

    encoded = _encode_features(
        data=data,
        feature_columns=feature_columns,
        search_space=list(request.search_space or []),
    )
    target = pd.to_numeric(data[target_column], errors="coerce")
    if target.isna().any():
        raise ValueError(
            f"Target column contains non-numeric values after conversion: {target_column}"
        )

    direction_sign = 1.0 if request.direction == "maximize" else -1.0
    train_X = torch.as_tensor(encoded["X"], dtype=torch.double)
    train_Y = torch.as_tensor(
        (target.to_numpy(dtype=float) * direction_sign).reshape(-1, 1),
        dtype=torch.double,
    )
    bounds = torch.as_tensor(encoded["bounds"], dtype=torch.double)

    model_config = ModelConfig(
        task_type="regression",
        model_type=request.model_type,
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
        model_kwargs=dict(request.model_kwargs or {}),
    )
    fit_config = FitConfig(maxiter=request.fit_maxiter)

    optimizer = BayesianOptimizer(
        model_config=model_config,
        fit_config=fit_config,
        bounds=bounds,
    )
    optimizer.fit(train_X, train_Y)

    acqf_kwargs = dict(request.acquisition.acqf_kwargs or {})
    acq_name = request.acquisition.name
    if _requires_best_f(acq_name) and "best_f" not in acqf_kwargs:
        acqf_kwargs["best_f"] = train_Y.max()
    if _requires_beta(acq_name) and "beta" not in acqf_kwargs:
        acqf_kwargs["beta"] = request.acquisition.beta

    acq_config = AcquisitionConfig(name=acq_name, acqf_kwargs=acqf_kwargs)
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

    candidate_result = optimizer.candidate(
        acq_config,
        opt_config,
        return_result=True,
    )
    raw_candidates = candidate_result.candidates
    raw_acq_value = candidate_result.acq_value
    candidates = _postprocess_candidates(
        raw_candidates,
        request=request,
        encoded=encoded,
    )

    mean, variance = optimizer.predict(candidates, return_type="mean_variance")
    std = variance.clamp_min(0).sqrt()
    rows = _candidate_rows(
        candidates=candidates,
        acq_value=raw_acq_value,
        mean=mean,
        std=std,
        encoded=encoded,
        request=request,
    )

    try:
        visualizations, visualization_warnings = _build_regression_visualizations(
            optimizer=optimizer,
            candidate_result=candidate_result,
            candidates=candidates,
            encoded=encoded,
            target=target,
            target_column=target_column,
            direction_sign=direction_sign,
        )
    except Exception as exc:
        visualizations = []
        visualization_warnings = [f"可視化を初期化できませんでした: {exc}"]

    best_observed = (
        float(target.max())
        if request.direction == "maximize"
        else float(target.min())
    )
    return {
        "dataset_id": record.dataset_id,
        "dataset_name": record.name,
        "task_type": "regression",
        "model_type": request.model_type,
        "n_train": int(train_X.shape[0]),
        "n_features": int(train_X.shape[1]),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "direction": request.direction,
        "cat_dims": encoded["cat_dims"],
        "category_maps": encoded["category_maps"],
        "best_observed": best_observed,
        "bounds": to_serializable(bounds),
        "raw_acq_value": to_serializable(raw_acq_value),
        "candidates": rows,
        "visualizations": visualizations,
        "visualization_warnings": visualization_warnings,
        "metadata": {
            "dropped_rows": int(record.profile["n_rows"] - len(data)),
            "acquisition": acq_name,
            "optimizer": request.optimizer.name,
            "repair_enabled": repair_config is not None,
        },
    }


__all__ = [
    "_figure_payload",
    "run_regression_web_workflow",
]
