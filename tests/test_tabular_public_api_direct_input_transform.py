from __future__ import annotations

import pandas as pd

from bochan.tabular import TabularBayesianOptimizer


def test_tabular_constructor_accepts_direct_input_transform_kwargs() -> None:
    bo = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x1", "x2"],
        target_cols="y",
        normalize=False,
        perturbation=True,
        n_w=8,
        std=0.05,
    )
    config = bo.model_config.input_transform_config
    assert config is not None
    assert config.normalize is False
    assert config.perturbation is True
    assert config.n_w == 8
    assert config.std == 0.05
    assert "perturbation" not in bo.bo_kwargs
    assert "n_w" not in bo.bo_kwargs
    assert "std" not in bo.bo_kwargs
    assert "normalize" not in bo.bo_kwargs


def test_tabular_fit_accepts_direct_input_transform_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}
    bo = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x1"],
        target_cols="y",
    )

    def fake_fit(X, Y, *, model_config=None, fit_config=None, **kwargs):
        captured["model_config"] = model_config
        return bo.bo

    monkeypatch.setattr(bo.bo, "fit", fake_fit)
    frame = pd.DataFrame({"x1": [0.0, 1.0], "y": [0.0, 1.0]})
    result = bo.fit(frame, perturbation=True, n_w=4, std=0.2)
    assert result is bo
    config = captured["model_config"].input_transform_config
    assert config is not None
    assert config.perturbation is True
    assert config.n_w == 4
    assert config.std == 0.2


def test_tabular_direct_input_transform_overrides_model_config_values() -> None:
    bo = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x1"],
        target_cols="y",
        perturbation=True,
        n_w=4,
    )
    updated = TabularBayesianOptimizer(
        model_config=bo.model_config,
        input_cols=["x1"],
        target_cols="y",
        perturbation=False,
        n_w=2,
    )
    config = updated.model_config.input_transform_config
    assert config is not None
    assert config.perturbation is False
    assert config.n_w == 2
