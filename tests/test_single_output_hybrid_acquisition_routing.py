from __future__ import annotations

from types import SimpleNamespace

import pytest

from bochan.api import AcquisitionConfig
from bochan.api.engine import BayesianOptimizer


def _resolve_hybrid_acquisition(
    *,
    task_type: str,
    model_type: str,
    acquisition: str,
    n_outputs: int = 1,
) -> str:
    optimizer = BayesianOptimizer.__new__(BayesianOptimizer)
    sub_bundles = [SimpleNamespace(task_type=task_type, model_type=model_type) for _ in range(n_outputs)]
    optimizer.bundle = SimpleNamespace(
        task_type="hybrid",
        model_type=model_type,
        metadata={"multi_output": True, "sub_bundles": sub_bundles},
        model=SimpleNamespace(specs=[]),
    )
    optimizer.model = object()
    optimizer.acquisition_registry = None

    resolved = optimizer._resolve_acquisition_config(AcquisitionConfig(name=acquisition))
    assert resolved.acqf_cls is not None
    return resolved.acqf_cls.__name__


@pytest.mark.parametrize(
    ("acquisition", "expected"),
    [
        ("variance", "qRegressionPosteriorVariance"),
        ("predictive_entropy", "qRegressionPredictiveEntropy"),
        ("BALD", "qRegressionBALD"),
        ("NIPV", "qRegressionNegIntegratedPosteriorVariance"),
        ("straddle", "qRegressionStraddle"),
        ("ICU", "qRegressionICU"),
        ("boundary_variance", "qRegressionBoundaryVariance"),
    ],
)
def test_single_regression_hybrid_routes_to_single_output_acquisition(
    acquisition: str,
    expected: str,
) -> None:
    assert (
        _resolve_hybrid_acquisition(
            task_type="regression",
            model_type="base",
            acquisition=acquisition,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("task_type", "acquisition", "expected"),
    [
        ("binary", "BALD", "qBinaryBALD"),
        ("binary", "variance", "qBinaryProbabilityVariance"),
        # The public binary NIPV name intentionally resolves to its differentiable
        # single-output proxy for standard optimize_acqf compatibility.
        ("binary", "NIPV", "qBinaryIntegratedPosteriorVarianceProxy"),
        ("ordinal", "BALD", "qOrdinalBALD"),
        ("ordinal", "variance", "qOrdinalUtilityVariance"),
        ("ordinal", "NIPV", "qOrdinalFantasyNegIntegratedPosteriorVariance"),
        ("multiclass", "BALD", "qMulticlassBALD"),
        ("multiclass", "variance", "qMulticlassProbabilityVariance"),
        (
            "multiclass",
            "NIPV",
            "qMulticlassIntegratedPosteriorVarianceProxy",
        ),
    ],
)
def test_single_classification_hybrid_uses_underlying_task_family(
    task_type: str,
    acquisition: str,
    expected: str,
) -> None:
    assert (
        _resolve_hybrid_acquisition(
            task_type=task_type,
            model_type="base",
            acquisition=acquisition,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("model_type", "acquisition", "expected"),
    [
        ("hetero", "variance", "qHeteroRegressionPosteriorVariance"),
        (
            "hetero",
            "NIPV",
            "qHeteroRegressionNegIntegratedPosteriorVariance",
        ),
        ("gamma_base", "variance", "qNonGaussianResponseMeanVariance"),
        ("gamma_base", "BALD", "qNonGaussianBALDProxy"),
        (
            "gamma_base",
            "NIPV",
            "qNonGaussianNegIntegratedResponseMeanVariance",
        ),
    ],
)
def test_single_regression_hybrid_preserves_model_family(
    model_type: str,
    acquisition: str,
    expected: str,
) -> None:
    assert (
        _resolve_hybrid_acquisition(
            task_type="regression",
            model_type=model_type,
            acquisition=acquisition,
        )
        == expected
    )


def test_multi_output_hybrid_keeps_multi_output_routing() -> None:
    assert (
        _resolve_hybrid_acquisition(
            task_type="regression",
            model_type="base",
            acquisition="variance",
            n_outputs=2,
        )
        == "qMultiOutputRegressionPosteriorVariance"
    )


def test_single_output_hybrid_can_fall_back_to_model_spec() -> None:
    optimizer = BayesianOptimizer.__new__(BayesianOptimizer)
    optimizer.bundle = SimpleNamespace(
        task_type="hybrid",
        model_type="base",
        metadata={"multi_output": True, "sub_bundles": []},
        model=SimpleNamespace(specs=[SimpleNamespace(task_type="binary", model=object())]),
    )
    optimizer.model = object()
    optimizer.acquisition_registry = None

    resolved = optimizer._resolve_acquisition_config(AcquisitionConfig(name="BALD"))
    assert resolved.acqf_cls is not None
    assert resolved.acqf_cls.__name__ == "qBinaryBALD"
