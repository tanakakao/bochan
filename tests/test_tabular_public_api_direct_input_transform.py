from __future__ import annotations

from bochan.tabular import TabularBayesianOptimizer
from bochan.tabular import optimizer_api


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

    def fake_fit(self, data=None, y=None, *, fit_config=None, **kwargs):
        captured["fit_config"] = fit_config
        captured["kwargs"] = kwargs
        return self

    monkeypatch.setattr(optimizer_api._BaseTabularBayesianOptimizer, "fit", fake_fit)

    bo = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x1"],
        target_cols="y",
    )

    result = bo.fit(object(), perturbation=True, n_w=4, std=0.2)

    assert result is bo
    kwargs = captured["kwargs"]
    config = kwargs["input_transform_config"]

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
