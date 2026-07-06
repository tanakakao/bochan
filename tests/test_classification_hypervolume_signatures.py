from __future__ import annotations

import inspect

import pytest

from bochan.acquisition.multiclass.bayesian_optimization import (
    multi_output as multiclass_bo,
)
from bochan.acquisition.ordinal.bayesian_optimization import multi_output as ordinal_bo
from bochan.api import AcquisitionConfig
from bochan.api.engine import _filter_context_fields_for_acqf


@pytest.mark.parametrize(
    "acquisition_cls",
    [
        ordinal_bo.qMultiOutputOrdinalExpectedHypervolumeImprovement,
        multiclass_bo.qMultiOutputMulticlassExpectedHypervolumeImprovement,
    ],
)
def test_eta_patched_ehvi_signatures_preserve_required_context(
    acquisition_cls,
) -> None:
    parameters = inspect.signature(acquisition_cls).parameters

    for name in (
        "model",
        "ref_point",
        "partitioning",
        "constraints",
        "eta",
        "fat",
    ):
        assert name in parameters


@pytest.mark.parametrize(
    "acquisition_cls",
    [
        ordinal_bo.qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
        multiclass_bo.qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
    ],
)
def test_eta_patched_nehvi_signatures_keep_x_baseline(
    acquisition_cls,
) -> None:
    parameters = inspect.signature(acquisition_cls).parameters

    assert "model" in parameters
    assert "X_baseline" in parameters

    config = AcquisitionConfig(name="nehvi", acqf_cls=acquisition_cls)
    filtered = _filter_context_fields_for_acqf(config)
    assert "X_baseline" in filtered.context_fields


def test_multiclass_nehvi_keeps_full_explicit_constructor_signature() -> None:
    parameters = inspect.signature(
        multiclass_bo.qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement
    ).parameters

    for name in (
        "model",
        "ref_point",
        "X_baseline",
        "constraints",
        "eta",
        "fat",
    ):
        assert name in parameters
