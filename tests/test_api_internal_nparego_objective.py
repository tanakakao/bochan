from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.acquisition.multi_objective.parego import qLogNParEGO

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
        qLogNParEGO,
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
    values = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        dtype=torch.double,
    )
    monkeypatch.setattr(
        engine_defaults,
        "observed_multiobjective_values",
        lambda *args, **kwargs: values,
    )
    config = AcquisitionConfig(name="nparego", acqf_cls=_ExternalNParEGO)

    resolved = engine_defaults._resolve_default_nparego_objective(
        SimpleNamespace(),
        config,
        DataContext(),
    )

    assert resolved.objective is not None
