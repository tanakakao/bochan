from __future__ import annotations

import inspect

import pytest

from bochan.acquisition.multiclass.bayesian_optimization import (
    qMulticlassExpectedHypervolumeImprovement,
    qMulticlassNoisyExpectedHypervolumeImprovement,
)
from bochan.acquisition.ordinal.bayesian_optimization import (
    qOrdinalExpectedHypervolumeImprovement,
    qOrdinalNoisyExpectedHypervolumeImprovement,
)
from bochan.api import AcquisitionConfig
from bochan.api.engine import _filter_context_fields_for_acqf


@pytest.mark.parametrize(
    "acquisition_cls",
    [
        qOrdinalExpectedHypervolumeImprovement,
        qMulticlassExpectedHypervolumeImprovement,
    ],
)
def test_ehvi_signatures_preserve_required_context(acquisition_cls) -> None:
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
        qOrdinalNoisyExpectedHypervolumeImprovement,
        qMulticlassNoisyExpectedHypervolumeImprovement,
    ],
)
def test_nehvi_signatures_keep_x_baseline(acquisition_cls) -> None:
    parameters = inspect.signature(acquisition_cls).parameters

    assert "model" in parameters
    assert "X_baseline" in parameters

    config = AcquisitionConfig(name="nehvi", acqf_cls=acquisition_cls)
    filtered = _filter_context_fields_for_acqf(config)
    assert "X_baseline" in filtered.context_fields


def test_multiclass_nehvi_keeps_full_explicit_constructor_signature() -> None:
    parameters = inspect.signature(
        qMulticlassNoisyExpectedHypervolumeImprovement
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
