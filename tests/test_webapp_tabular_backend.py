from __future__ import annotations

import pandas as pd

from bochan.api import FitConfig, ModelConfig
from bochan.serving.webapp.tabular_backend import (
    feature_category_maps,
    fit_tabular_optimizer,
    target_category_maps,
)
from bochan.serving.webapp.workflows import run_regression_web_workflow
from bochan.tabular import TabularBayesianOptimizer


def test_web_workflow_export_uses_tabular_implementation() -> None:
    assert run_regression_web_workflow.__module__.endswith("workflows_tabular")


def test_feature_category_maps_restore_numeric_labels() -> None:
    data = pd.DataFrame({"temperature": [10, 20, 10], "y": [1.0, 2.0, 1.5]})
    encoded = {
        "category_maps": {"temperature": {"10": 0, "20": 1}},
    }

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


def test_fit_tabular_optimizer_uses_dataframe_backend() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.0],
            "material": ["A", "B", "A"],
            "y": [0.0, 0.8, 0.2],
        }
    )
    encoded_features = {
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
    target_metadata = {
        "y": {
            "internal_task": "regression",
            "classes": None,
        }
    }

    optimizer = fit_tabular_optimizer(
        data=data,
        feature_columns=["x", "material"],
        target_columns=["y"],
        encoded_features=encoded_features,
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
