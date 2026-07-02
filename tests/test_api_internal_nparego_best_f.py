from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import bochan.api.engine_defaults as engine_defaults
from bochan.acquisition.binary.bayesian_optimization import (
    qHeteroMultiOutputBinaryNParEGO,
    qMultiOutputBinaryNParEGO,
)
from bochan.acquisition.multiclass.bayesian_optimization import (
    qHeteroMultiOutputMulticlassNParEGO,
    qMultiOutputMulticlassNParEGO,
)
from bochan.acquisition.ordinal.bayesian_optimization import (
    qHeteroMultiOutputOrdinalNParEGO,
    qMultiOutputOrdinalNParEGO,
)
from bochan.acquisition.regression.bayesian_optimization import (
    qHeteroMultiOutputRegressionNParEGO,
    qMultiOutputRegressionNParEGO,
)
from bochan.api import AcquisitionConfig, DataContext
from bochan.api import _uses_internal_nparego_baseline


@pytest.mark.parametrize(
    "acqf_cls",
    [
        qMultiOutputRegressionNParEGO,
        qHeteroMultiOutputRegressionNParEGO,
        qMultiOutputBinaryNParEGO,
        qHeteroMultiOutputBinaryNParEGO,
        qMultiOutputOrdinalNParEGO,
        qHeteroMultiOutputOrdinalNParEGO,
        qMultiOutputMulticlassNParEGO,
        qHeteroMultiOutputMulticlassNParEGO,
    ],
)
def test_builtin_nparego_uses_internal_baseline(acqf_cls) -> None:
    config = AcquisitionConfig(name="nparego", acqf_cls=acqf_cls)

    assert _uses_internal_nparego_baseline(config)


def test_internal_ordinal_nparego_skips_automatic_best_f(monkeypatch) -> None:
    monkeypatch.setattr(
        engine_defaults,
        "compute_best_f",
        lambda *args, **kwargs: pytest.fail("internal NParEGO must not compute best_f"),
    )
    config = AcquisitionConfig(
        name="nparego",
        acqf_cls=qMultiOutputOrdinalNParEGO,
    )
    context = DataContext(best_f=torch.tensor(5.0, dtype=torch.double))

    resolved, resolved_context = engine_defaults._resolve_best_f_default(
        SimpleNamespace(),
        config,
        context,
    )

    assert resolved is config
    assert resolved_context.best_f is None
    assert "best_f" not in resolved.acqf_kwargs


def test_explicit_best_f_is_preserved_for_internal_nparego(monkeypatch) -> None:
    explicit = torch.tensor(9.0, dtype=torch.double)
    monkeypatch.setattr(
        engine_defaults,
        "compute_best_f",
        lambda *args, **kwargs: pytest.fail("explicit best_f must not be recomputed"),
    )
    config = AcquisitionConfig(
        name="nparego",
        acqf_cls=qMultiOutputOrdinalNParEGO,
        acqf_kwargs={"best_f": explicit},
    )

    resolved, context = engine_defaults._resolve_best_f_default(
        SimpleNamespace(),
        config,
        DataContext(best_f=torch.tensor(5.0, dtype=torch.double)),
    )

    assert resolved.acqf_kwargs["best_f"] is explicit
    assert context.best_f is None


class _ExternalNParEGO:
    def __init__(self, model, best_f, objective=None) -> None:
        self.model = model
        self.best_f = best_f
        self.objective = objective


def test_external_nparego_keeps_generic_best_f_inference(monkeypatch) -> None:
    expected = torch.tensor(1.25, dtype=torch.double)
    monkeypatch.setattr(
        engine_defaults,
        "compute_best_f",
        lambda *args, **kwargs: expected,
    )
    config = AcquisitionConfig(name="nparego", acqf_cls=_ExternalNParEGO)

    _, context = engine_defaults._resolve_best_f_default(
        SimpleNamespace(),
        config,
        DataContext(),
    )

    assert context.best_f is expected
