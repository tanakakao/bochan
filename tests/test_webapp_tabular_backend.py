from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from bochan.api import FitConfig, ModelConfig
from bochan.serving.webapp import target_missing_policy as policy
from bochan.serving.webapp.tabular_backend import (
    feature_category_maps,
    fit_tabular_optimizer,
    target_category_maps,
)
from bochan.serving.webapp.workflows import (
    _run_regression_web_workflow,
    run_regression_web_workflow,
)
from bochan.tabular import TabularBayesianOptimizer


def test_web_workflow_wrapper_calls_tabular_implementation() -> None:
    assert run_regression_web_workflow.__module__.endswith("workflows")
    assert _run_regression_web_workflow.__module__.endswith("workflows_tabular")


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


def _request(*, targets: list[str], model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        target_columns=targets,
        target_column=targets[0],
        model_type=model_type,
    )


def test_single_and_regular_multiobjective_drop_missing_targets() -> None:
    single = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, None]})
    with policy.target_missing_run(_request(targets=["y"], model_type="base")):
        cleaned_single = policy.clean_rows(
            single,
            ["x"],
            ["y"],
            drop_missing=True,
        )
    assert len(cleaned_single) == 1

    multi = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y1": [1.0, None, 3.0],
            "y2": [2.0, 4.0, None],
        }
    )
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="base")
    ):
        cleaned_multi = policy.clean_rows(
            multi,
            ["x"],
            ["y1", "y2"],
            drop_missing=True,
        )
    assert cleaned_multi.to_dict("list") == {
        "x": [0.0],
        "y1": [1.0],
        "y2": [2.0],
    }


def test_multitask_preserves_partial_target_rows() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0, None, 3.0],
            "y1": [1.0, None, 3.0, None],
            "y2": [None, 4.0, 5.0, None],
        }
    )
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="multitask")
    ) as report:
        cleaned = policy.clean_rows(
            data,
            ["x"],
            ["y1", "y2"],
            drop_missing=True,
        )

    assert cleaned["x"].tolist() == [0.0, 1.0]
    assert cleaned["y1"].isna().tolist() == [False, True]
    assert cleaned["y2"].isna().tolist() == [True, False]
    assert report["target_missing_counts"] == {"y1": 1, "y2": 1}
    assert report["dropped_feature_rows"] == 1
    assert report["dropped_all_target_missing_rows"] == 1


def test_multitask_encoder_preserves_regression_nan_cells() -> None:
    data = pd.DataFrame({"y1": [1.0, None], "y2": [None, 2.0]})
    settings = [
        {
            "target": target,
            "task_type": "regression",
            "goal": "none",
            "value": None,
            "legacy": False,
        }
        for target in ["y1", "y2"]
    ]
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="multitask")
    ):
        encoded, metadata = policy.encode_targets(data, settings)

    assert encoded.isna().sum().to_dict() == {"y1": 1, "y2": 1}
    assert metadata["y1"]["internal_task"] == "regression"
    assert metadata["y2"]["internal_task"] == "regression"


def test_adaptive_multitask_selects_wide_or_kronecker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bochan.models.regression.gaussian as gaussian
    import bochan.models.wide_multitask_variants as wide

    class FakeWide:
        def __init__(self, train_X, train_Y, **kwargs):
            self.train_X = train_X
            self.train_Y = train_Y

    class FakeKronecker:
        def __init__(self, train_X, train_Y, **kwargs):
            self.train_X = train_X
            self.train_Y = train_Y

    monkeypatch.setattr(wide, "WideMultiTaskGP", FakeWide)
    monkeypatch.setattr(
        gaussian,
        "PerturbationSupportedKroneckerMultiTaskGP",
        FakeKronecker,
    )
    X = torch.zeros(2, 1, dtype=torch.double)
    incomplete = policy.adaptive_multitask_gp(
        X,
        torch.tensor([[1.0, float("nan")], [2.0, 3.0]], dtype=torch.double),
    )
    complete = policy.adaptive_multitask_gp(
        X,
        torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.double),
    )

    assert isinstance(incomplete, FakeWide)
    assert incomplete.web_multitask_variant == "wide_multitask"
    assert isinstance(complete, FakeKronecker)
    assert complete.web_multitask_variant == "kronecker"


def test_missing_targets_are_completed_only_for_acquisition_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    observed = torch.tensor(
        [[1.0, float("nan")], [float("nan"), 4.0]],
        dtype=torch.double,
    )

    class FakePosterior:
        mean = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.double)

    class FakeModel:
        web_multitask_variant = "wide_multitask"
        web_effective_model_type = "multitask"

        def posterior(self, value):
            assert torch.equal(value, X)
            return FakePosterior()

    fake = SimpleNamespace(
        dataset=SimpleNamespace(X=X, Y=observed.clone()),
        bo=SimpleNamespace(model=FakeModel()),
    )
    monkeypatch.setattr(
        policy,
        "_ORIGINAL_FIT_TABULAR_OPTIMIZER",
        lambda **kwargs: fake,
    )
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="multitask")
    ) as report:
        fitted = policy.fit_tabular_optimizer()

    assert torch.isnan(fitted.web_observed_target_tensor).sum().item() == 2
    assert torch.equal(
        fitted.dataset.Y,
        torch.tensor([[1.0, 2.5], [3.5, 4.0]], dtype=torch.double),
    )
    assert report["acquisition_baseline_completed"] is True
