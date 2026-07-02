from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import bochan.api as api
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
def test_internal_nparego_skips_generic_scalarization_objective(acqf_cls) -> None:
    config = AcquisitionConfig(name="nparego", acqf_cls=acqf_cls)

    resolved = engine_defaults._resolve_default_nparego_objective(
        SimpleNamespace(),
        config,
        DataContext(),
    )

    assert resolved is config
    assert resolved.objective is None


def test_internal_nparego_preserves_explicit_preprocessing_objective() -> None:
    objective = object()
    config = AcquisitionConfig(
        name="nparego",
        acqf_cls=qMultiOutputMulticlassNParEGO,
        objective=objective,
    )

    resolved = engine_defaults._resolve_default_nparego_objective(
        SimpleNamespace(),
        config,
        DataContext(),
    )

    assert resolved is config
    assert resolved.objective is objective


class _ExternalNParEGO:
    pass


def test_external_nparego_keeps_generic_objective_resolution(monkeypatch) -> None:
    sentinel = object()

    def _fake_original(bundle, config, context):
        del bundle, context
        return replace(config, objective=sentinel)

    monkeypatch.setattr(
        api,
        "_original_resolve_default_nparego_objective",
        _fake_original,
    )
    config = AcquisitionConfig(name="nparego", acqf_cls=_ExternalNParEGO)

    resolved = engine_defaults._resolve_default_nparego_objective(
        SimpleNamespace(),
        config,
        DataContext(),
    )

    assert resolved.objective is sentinel
