from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from bochan.serving.webapp import target_missing_policy as policy


def _request(*, targets: list[str], model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        target_columns=targets,
        target_column=targets[0] if targets else None,
        model_type=model_type,
    )


def test_single_objective_drops_missing_target_rows() -> None:
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, None, 3.0]})
    with policy.target_missing_run(_request(targets=["y"], model_type="base")) as report:
        cleaned = policy.clean_rows(
            data,
            ["x"],
            ["y"],
            drop_missing=True,
        )

    assert cleaned.to_dict("list") == {"x": [0.0, 2.0], "y": [1.0, 3.0]}
    assert report["policy"] == "drop_rows"
    assert report["target_missing_detected"] is True


def test_multiobjective_non_multitask_drops_any_missing_target_row() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y1": [1.0, None, 3.0],
            "y2": [2.0, 4.0, None],
        }
    )
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="base")
    ):
        cleaned = policy.clean_rows(
            data,
            ["x"],
            ["y1", "y2"],
            drop_missing=True,
        )

    assert cleaned.to_dict("list") == {"x": [0.0], "y1": [1.0], "y2": [2.0]}


def test_multitask_preserves_partial_targets_and_drops_unusable_rows() -> None:
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
    assert report["policy"] == "wide_multitask"
    assert report["target_missing_detected"] is True
    assert report["target_missing_counts"] == {"y1": 1, "y2": 1}
    assert report["dropped_feature_rows"] == 1
    assert report["dropped_all_target_missing_rows"] == 1


def test_multitask_requires_an_observation_for_every_target() -> None:
    data = pd.DataFrame({"x": [0.0, 1.0], "y1": [1.0, 2.0], "y2": [None, None]})
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="multitask")
    ):
        with pytest.raises(ValueError, match="without observations"):
            policy.clean_rows(
                data,
                ["x"],
                ["y1", "y2"],
                drop_missing=True,
            )


def test_target_encoder_preserves_regression_nan_cells() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y1": [1.0, None],
            "y2": [None, 2.0],
        }
    )
    settings = [
        {
            "target": "y1",
            "task_type": "regression",
            "goal": "none",
            "value": None,
            "legacy": False,
        },
        {
            "target": "y2",
            "task_type": "regression",
            "goal": "none",
            "value": None,
            "legacy": False,
        },
    ]
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="multitask")
    ):
        encoded, metadata = policy.encode_targets(data, settings)

    assert encoded["y1"].isna().tolist() == [False, True]
    assert encoded["y2"].isna().tolist() == [True, False]
    assert metadata["y1"]["internal_task"] == "regression"
    assert metadata["y2"]["internal_task"] == "regression"


def test_adaptive_multitask_uses_wide_for_nan_and_kronecker_for_complete(
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
    assert incomplete.web_effective_model_type == "multitask"
    assert isinstance(complete, FakeKronecker)
    assert complete.web_multitask_variant == "kronecker"
    assert complete.web_effective_model_type == "kronecker"


def test_fit_wrapper_completes_only_acquisition_baseline(
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
