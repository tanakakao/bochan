from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.acquisition.multi_objective.parego import qLogNParEGO

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
from bochan.api import AcquisitionConfig, DataContext, MultiObjectiveConfig
from bochan.api.acquisition import defaults as engine_defaults


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


def test_log_nparego_routes_multiobjective_weights_to_botorch_keyword() -> None:
    weights = torch.tensor([0.3, 0.7], dtype=torch.double)
    config = AcquisitionConfig(name="lognparego", acqf_cls=qLogNParEGO)
    original_multi_objective = MultiObjectiveConfig(scalarization_weights=weights)
    context = DataContext(multi_objective=original_multi_objective)

    resolved, resolved_context = (
        engine_defaults._resolve_internal_nparego_scalarization_weights(
            config,
            context,
        )
    )

    assert resolved.acqf_kwargs["scalarization_weights"] is weights
    assert resolved_context.multi_objective is not original_multi_objective
    assert resolved_context.multi_objective.auto_scalarization is False
    assert original_multi_objective.auto_scalarization is True


def test_native_nparego_routes_multiobjective_weights_to_native_keyword() -> None:
    weights = torch.tensor([0.4, 0.6], dtype=torch.double)
    config = AcquisitionConfig(
        name="nparego",
        acqf_cls=qMultiOutputRegressionNParEGO,
    )
    context = DataContext(
        multi_objective=MultiObjectiveConfig(scalarization_weights=weights),
    )

    resolved, resolved_context = (
        engine_defaults._resolve_internal_nparego_scalarization_weights(
            config,
            context,
        )
    )

    assert resolved.acqf_kwargs["weights"] is weights
    assert resolved_context.multi_objective.auto_scalarization is False


def test_internal_nparego_explicit_weights_take_precedence() -> None:
    configured = torch.tensor([0.2, 0.8], dtype=torch.double)
    explicit = torch.tensor([0.9, 0.1], dtype=torch.double)
    config = AcquisitionConfig(
        name="nparego",
        acqf_cls=qMultiOutputRegressionNParEGO,
        acqf_kwargs={"weights": explicit},
    )
    context = DataContext(
        multi_objective=MultiObjectiveConfig(scalarization_weights=configured),
    )

    resolved, resolved_context = (
        engine_defaults._resolve_internal_nparego_scalarization_weights(
            config,
            context,
        )
    )

    assert resolved.acqf_kwargs["weights"] is explicit
    assert resolved_context.multi_objective.auto_scalarization is False


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
