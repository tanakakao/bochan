from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from bochan.models.ordinal.deep import (
    DeepKernelOrdinalGPModel,
    DeepKernelOrdinalMixedGPModel,
)
from bochan.models.ordinal.deep._mll_beta import enable_make_mll_beta


@pytest.mark.parametrize(
    "model_cls",
    [DeepKernelOrdinalGPModel, DeepKernelOrdinalMixedGPModel],
)
def test_public_deepkernel_make_mll_accepts_beta(model_cls) -> None:
    parameter = inspect.signature(model_cls.make_mll).parameters["beta"]

    assert parameter.default is None


def test_make_mll_beta_adapter_updates_variational_weight() -> None:
    class DummyModel:
        def make_mll(self):
            return SimpleNamespace(beta=1.0)

    model_cls = enable_make_mll_beta(DummyModel)
    mll = model_cls().make_mll(beta=0.01)

    assert mll.beta == 0.01


def test_make_mll_beta_adapter_preserves_default_when_omitted() -> None:
    class DummyModel:
        def make_mll(self):
            return SimpleNamespace(beta=0.75)

    model_cls = enable_make_mll_beta(DummyModel)
    mll = model_cls().make_mll()

    assert mll.beta == 0.75


def test_make_mll_beta_adapter_rejects_unsupported_mll() -> None:
    class DummyModel:
        def make_mll(self):
            return SimpleNamespace()

    model_cls = enable_make_mll_beta(DummyModel)

    with pytest.raises(TypeError, match="does not support the beta parameter"):
        model_cls().make_mll(beta=0.01)
