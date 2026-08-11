from __future__ import annotations

import inspect

import pytest
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)

from bochan.acquisition.binary.bayesian_optimization import (
    qBinaryExpectedHypervolumeImprovement,
    qBinaryExpectedImprovement,
    qBinaryNParEGO,
    qBinaryProbabilityOfImprovement,
    qBinaryUpperConfidenceBound,
)
from bochan.acquisition.multiclass.bayesian_optimization import (
    qMulticlassExpectedHypervolumeImprovement,
    qMulticlassExpectedImprovement,
    qMulticlassNParEGO,
    qMulticlassProbabilityOfImprovement,
    qMulticlassUpperConfidenceBound,
)
from bochan.acquisition.ordinal.bayesian_optimization import (
    qOrdinalExpectedHypervolumeImprovement,
    qOrdinalExpectedImprovement,
    qOrdinalExpectedUtility,
    qOrdinalNParEGO,
    qOrdinalProbabilityOfImprovement,
    qOrdinalUpperConfidenceBound,
)
from bochan.acquisition.regression.bayesian_optimization import (
    qRegressionExpectedHypervolumeImprovement,
    qRegressionNParEGO,
)


@pytest.mark.parametrize(
    ("acquisition_cls", "botorch_base"),
    [
        (qBinaryExpectedImprovement, qExpectedImprovement),
        (qBinaryProbabilityOfImprovement, qProbabilityOfImprovement),
        (qBinaryUpperConfidenceBound, qUpperConfidenceBound),
        (qMulticlassExpectedImprovement, qExpectedImprovement),
        (qMulticlassProbabilityOfImprovement, qProbabilityOfImprovement),
        (qMulticlassUpperConfidenceBound, qUpperConfidenceBound),
        (qOrdinalExpectedUtility, qSimpleRegret),
        (qOrdinalExpectedImprovement, qExpectedImprovement),
        (qOrdinalProbabilityOfImprovement, qProbabilityOfImprovement),
        (qOrdinalUpperConfidenceBound, qUpperConfidenceBound),
    ],
)
def test_standard_bo_acquisitions_use_botorch_joint_q_bases(
    acquisition_cls,
    botorch_base,
) -> None:
    assert issubclass(acquisition_cls, botorch_base)


@pytest.mark.parametrize(
    "acquisition_cls",
    [
        qBinaryExpectedImprovement,
        qBinaryProbabilityOfImprovement,
        qBinaryUpperConfidenceBound,
        qMulticlassExpectedImprovement,
        qMulticlassProbabilityOfImprovement,
        qMulticlassUpperConfidenceBound,
        qOrdinalExpectedUtility,
        qOrdinalExpectedImprovement,
        qOrdinalProbabilityOfImprovement,
        qOrdinalUpperConfidenceBound,
    ],
)
def test_standard_bo_public_api_excludes_legacy_pointwise_and_penalty_controls(
    acquisition_cls,
) -> None:
    parameters = inspect.signature(acquisition_cls).parameters
    legacy_parameters = {
        "q_mode",
        "reduction",
        "num_samples",
        "X_observed",
        "X_baseline",
        "pending_penalty_weight",
        "pending_penalty_beta",
        "observed_penalty_weight",
        "observed_penalty_beta",
        "same_batch_penalty_weight",
        "same_batch_penalty_beta",
        "best_f_margin",
        "best_f_quantile",
    }
    assert legacy_parameters.isdisjoint(parameters)


@pytest.mark.parametrize(
    "acquisition_cls",
    [
        qBinaryExpectedImprovement,
        qBinaryProbabilityOfImprovement,
        qMulticlassExpectedImprovement,
        qMulticlassProbabilityOfImprovement,
        qOrdinalExpectedImprovement,
        qOrdinalProbabilityOfImprovement,
    ],
)
def test_improvement_acquisitions_require_explicit_best_f(acquisition_cls) -> None:
    best_f = inspect.signature(acquisition_cls).parameters["best_f"]
    assert best_f.default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "acquisition_cls",
    [
        qBinaryNParEGO,
        qMulticlassNParEGO,
        qOrdinalNParEGO,
        qRegressionNParEGO,
    ],
)
def test_nparego_exposes_baseline_and_reference_point(acquisition_cls) -> None:
    parameters = inspect.signature(acquisition_cls).parameters
    for name in ("model", "X_baseline", "ref_point"):
        assert name in parameters
        assert parameters[name].default is inspect.Parameter.empty


def test_multiobjective_public_names_are_domain_first() -> None:
    classes = [
        qBinaryExpectedHypervolumeImprovement,
        qBinaryNParEGO,
        qMulticlassExpectedHypervolumeImprovement,
        qMulticlassNParEGO,
        qOrdinalExpectedHypervolumeImprovement,
        qOrdinalNParEGO,
        qRegressionExpectedHypervolumeImprovement,
        qRegressionNParEGO,
    ]
    for acquisition_cls in classes:
        assert "MultiOutput" not in acquisition_cls.__name__
