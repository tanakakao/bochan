from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from bochan.api import FitConfig, ModelConfig
from bochan.serving.fastapi.schemas.tabular import TabularFitModelRequest
from bochan.serving.webapp import target_missing_policy as policy
from bochan.serving.webapp import target_settings
from bochan.serving.webapp.tabular_backend import (
    feature_category_maps,
    fit_tabular_optimizer,
    target_category_maps,
)
from bochan.serving.webapp.workflows import (
    _run_regression_web_workflow,
    run_regression_web_workflow,
)
from bochan.tabular import ObservationTabularDataset, TabularBayesianOptimizer


def test_web_workflow_wrapper_calls_source_level_tabular_entrypoint() -> None:
    assert run_regression_web_workflow.__module__.endswith("workflows")
    assert _run_regression_web_workflow.__module__.endswith("workflows_tabular")
    assert not hasattr(policy, "install_workflow_adapters")
    assert not hasattr(policy, "adaptive_multitask_gp")


def test_tabular_fastapi_schema_accepts_feature_imputation_options() -> None:
    fields = getattr(TabularFitModelRequest, "model_fields", {})
    if not fields:
        fields = getattr(TabularFitModelRequest, "__fields__", {})
    assert {
        "missing_strategy",
        "continuous_impute_strategy",
        "categorical_impute_strategy",
        "impute_random_state",
        "impute_max_iter",
        "multiple_impute_sample_posterior",
    }.issubset(fields)


def test_feature_category_maps_restore_numeric_labels() -> None:
    data = pd.DataFrame({"temperature": [10, 20, 10], "y": [1.0, 2.0, 1.5]})
    encoded = {"category_maps": {"temperature": {"10": 0, "20": 1}}}

    maps = feature_category_maps(data, encoded)

    assert maps == {"temperature": {10: 0, 20: 1}}


def test_target_category_maps_preserve_custom_ordinal_order() -> None:
    metadata = {
        "rank": {
            "internal_task": "ordinal",
            "classes": ["high", "medium", "low"],
        },
        "yield": {
            "internal_task": "regression",
            "classes": None,
        },
    }

    assert target_category_maps(metadata) == {
        "rank": {"high": 0, "medium": 1, "low": 2}
    }


def _encoded_features() -> dict[str, object]:
    return {
        "X": [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]],
        "bounds": [[0.0, 0.0], [1.0, 1.0]],
        "feature_columns": ["x", "material"],
        "cat_dims": [1],
        "numeric_indices": [0],
        "category_maps": {"material": {"A": 0, "B": 1}},
        "inverse_category_maps": {"material": {0: "A", 1: "B"}},
        "fixed_features": {},
        "steps": {},
    }


def test_fit_tabular_optimizer_uses_dataframe_backend() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.0],
            "material": ["A", "B", "A"],
            "y": [0.0, 0.8, 0.2],
        }
    )
    target_metadata = {"y": {"internal_task": "regression", "classes": None}}

    optimizer = fit_tabular_optimizer(
        data=data,
        feature_columns=["x", "material"],
        target_columns=["y"],
        encoded_features=_encoded_features(),
        target_metadata=target_metadata,
        model_config=ModelConfig(
            task_type="regression",
            model_type="base",
            outcome_transform=False,
        ),
        fit_config=FitConfig(skip_fit=True),
    )

    assert isinstance(optimizer, TabularBayesianOptimizer)
    assert optimizer.dataset is not None
    assert optimizer.dataset.feature_names == ["x", "material"]
    assert optimizer.dataset.target_names == ["y"]
    assert optimizer.dataset.cat_dims == [1]
    assert optimizer.dataset.category_maps == {"material": {"A": 0, "B": 1}}
    assert optimizer.dataset.X.shape == (3, 2)
    assert optimizer.dataset.Y is not None
    assert optimizer.dataset.Y.shape == (3, 1)


def _request(
    *,
    targets: list[str],
    model_type: str,
    feature_missing: dict[str, object] | None = None,
    categorical_features: list[str] | None = None,
) -> SimpleNamespace:
    categorical = set(categorical_features or [])
    return SimpleNamespace(
        target_columns=targets,
        target_column=targets[0],
        model_type=model_type,
        model_kwargs=(
            {"web_feature_missing": dict(feature_missing)}
            if feature_missing is not None
            else {}
        ),
        search_space=[
            SimpleNamespace(name=name, type="categorical") for name in categorical
        ],
    )


def test_feature_missing_rows_are_dropped_by_default() -> None:
    data = pd.DataFrame({"x": [0.0, None, 2.0], "y": [1.0, 2.0, 3.0]})
    with policy.target_missing_run(
        _request(targets=["y"], model_type="base")
    ) as report:
        cleaned = target_settings._clean_rows(
            data,
            ["x"],
            ["y"],
            drop_missing=True,
        )

    assert cleaned["x"].tolist() == [0.0, 2.0]
    assert report["feature_missing_strategy"] == "drop"
    assert report["dropped_feature_rows"] == 1


def test_feature_missing_values_are_imputed_with_tabular_strategies() -> None:
    data = pd.DataFrame(
        {
            "x": [1.0, None, 3.0],
            "material": ["A", None, "A"],
            "y": [1.0, 2.0, 3.0],
        }
    )
    request = _request(
        targets=["y"],
        model_type="base",
        feature_missing={
            "strategy": "impute",
            "continuous_impute_strategy": "mean",
            "categorical_impute_strategy": "mode",
            "impute_max_iter": 10,
        },
        categorical_features=["material"],
    )
    with policy.target_missing_run(request) as report:
        cleaned = target_settings._clean_rows(
            data,
            ["x", "material"],
            ["y"],
            drop_missing=True,
        )

    assert cleaned["x"].tolist() == [1.0, 2.0, 3.0]
    assert cleaned["material"].tolist() == ["A", "A", "A"]
    assert report["feature_missing_strategy"] == "impute"
    assert report["dropped_feature_rows"] == 0
    assert report["feature_impute_values"] == {"x": 2.0, "material": "A"}


def test_web_feature_missing_settings_are_removed_from_model_kwargs() -> None:
    request = SimpleNamespace(
        model_kwargs={
            "web_feature_missing": {"strategy": "impute"},
            "keep": 1,
        }
    )
    settings, model_kwargs = target_settings._resolve_target_settings(
        request,
        target_columns=["y"],
        directions={"y": "maximize"},
    )

    assert settings[0]["target"] == "y"
    assert model_kwargs == {"keep": 1}


def test_web_multitask_backend_keeps_nan_targets_without_posterior_imputation() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.0],
            "material": ["A", "B", "A"],
            "y1": [1.0, None, 3.0],
            "y2": [None, 2.0, 4.0],
        }
    )
    metadata = {
        "y1": {"internal_task": "regression", "classes": None},
        "y2": {"internal_task": "regression", "classes": None},
    }
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="multitask")
    ) as report:
        optimizer = fit_tabular_optimizer(
            data=data,
            feature_columns=["x", "material"],
            target_columns=["y1", "y2"],
            encoded_features=_encoded_features(),
            target_metadata=metadata,
            model_config=ModelConfig(
                task_type="regression",
                model_type="multitask",
                outcome_transform=False,
            ),
            fit_config=FitConfig(skip_fit=True),
        )

    assert isinstance(optimizer.dataset, ObservationTabularDataset)
    assert torch.isnan(optimizer.dataset.Y).sum().item() == 2
    assert torch.isnan(optimizer.bo.model.train_Y_wide).sum().item() == 2
    assert report["acquisition_baseline_completed"] is False
    assert type(optimizer.bo.model).__name__ == "WideMultiTaskGP"


def test_nan_safe_web_reference_point_uses_each_observed_objective() -> None:
    values = torch.tensor(
        [[1.0, float("nan")], [float("nan"), 4.0], [2.0, 5.0]],
        dtype=torch.double,
    )

    ref = target_settings._reference_point(values)

    assert torch.isfinite(ref).all()
    assert ref[0] < 1.0
    assert ref[1] < 4.0
