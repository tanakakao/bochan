from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import bochan.models.ordinal.deep.deepkernel_configurable as deepkernel_module
from bochan.models.ordinal.deep import (
    DeepKernelOrdinalGPModel,
    DeepKernelOrdinalMixedGPModel,
)


@pytest.mark.parametrize(
    "model_cls",
    [DeepKernelOrdinalGPModel, DeepKernelOrdinalMixedGPModel],
)
def test_public_deepkernel_make_mll_accepts_beta(model_cls) -> None:
    parameter = inspect.signature(model_cls.make_mll).parameters["beta"]

    assert parameter.default is None


@pytest.mark.parametrize(
    "model_cls",
    [DeepKernelOrdinalGPModel, DeepKernelOrdinalMixedGPModel],
)
def test_make_mll_passes_beta_to_variational_elbo(monkeypatch, model_cls) -> None:
    captured = {}

    def fake_variational_elbo(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(deepkernel_module, "VariationalELBO", fake_variational_elbo)
    model = SimpleNamespace(
        likelihood="likelihood",
        deepkernel="model",
        train_X=SimpleNamespace(shape=(8, 3)),
        use_predictive_log_likelihood=False,
    )

    result = model_cls.make_mll(model, beta=0.01)

    assert result.beta == 0.01
    assert captured == {
        "likelihood": "likelihood",
        "model": "model",
        "num_data": 8,
        "beta": 0.01,
    }


@pytest.mark.parametrize(
    "model_cls",
    [DeepKernelOrdinalGPModel, DeepKernelOrdinalMixedGPModel],
)
def test_make_mll_omits_beta_when_not_specified(monkeypatch, model_cls) -> None:
    captured = {}

    def fake_variational_elbo(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(deepkernel_module, "VariationalELBO", fake_variational_elbo)
    model = SimpleNamespace(
        likelihood="likelihood",
        deepkernel="model",
        train_X=SimpleNamespace(shape=(8, 3)),
        use_predictive_log_likelihood=False,
    )

    model_cls.make_mll(model)

    assert "beta" not in captured


def test_make_mll_preserves_predictive_log_likelihood_choice(monkeypatch) -> None:
    captured = {}

    def fake_predictive_log_likelihood(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        deepkernel_module,
        "PredictiveLogLikelihood",
        fake_predictive_log_likelihood,
    )
    model = SimpleNamespace(
        likelihood="likelihood",
        deepkernel="model",
        train_X=SimpleNamespace(shape=(8, 3)),
        use_predictive_log_likelihood=True,
    )

    result = DeepKernelOrdinalGPModel.make_mll(model, beta=0.25)

    assert result.beta == 0.25
    assert captured["num_data"] == 8
