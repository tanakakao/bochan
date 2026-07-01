from types import SimpleNamespace

import pytest

from bochan.api import AcquisitionConfig, BayesianOptimizer


@pytest.mark.parametrize(
    ("alias", "expected_name"),
    [
        ("bald", "qMultiOutputBinaryBALD"),
        ("entropy", "qMultiOutputBinaryPredictiveEntropy"),
        ("variance", "qMultiOutputBinaryProbabilityVariance"),
        ("margin", "qMultiOutputBinaryMarginUncertainty"),
        ("nipv", "qMultiOutputBinaryIntegratedPosteriorVarianceProxy"),
        ("straddle", "qMultiOutputBinaryLatentStraddleAcquisition"),
        ("jointstraddle", "qMultiOutputBinaryJointLatentStraddleAcquisition"),
        ("icu", "qMultiOutputBinaryICUAcquisition"),
        ("boundaryvariance", "qMultiOutputBinaryBoundaryVarianceAcquisition"),
        ("classentropy", "qMultiOutputBinaryClassEntropyAcquisition"),
        ("ehvi", "qMultiOutputBinaryExpectedHypervolumeImprovement"),
        ("nehvi", "qMultiOutputBinaryNoisyExpectedHypervolumeImprovement"),
        ("nparego", "qMultiOutputBinaryNParEGO"),
    ],
)
def test_kronecker_binary_aliases_use_multioutput_family(alias, expected_name):
    optimizer = object.__new__(BayesianOptimizer)
    optimizer.acquisition_registry = None
    optimizer.model = SimpleNamespace(num_outputs=2)
    optimizer.bundle = SimpleNamespace(
        model=optimizer.model,
        task_type="binary",
        model_type="kronecker",
        metadata={},
    )

    resolved = optimizer._resolve_acquisition_config(AcquisitionConfig(name=alias))

    assert resolved.acqf_cls.__name__ == expected_name


def test_single_output_binary_alias_stays_single_output():
    optimizer = object.__new__(BayesianOptimizer)
    optimizer.acquisition_registry = None
    optimizer.model = SimpleNamespace(num_outputs=1)
    optimizer.bundle = SimpleNamespace(
        model=optimizer.model,
        task_type="binary",
        model_type="single_task",
        metadata={},
    )

    resolved = optimizer._resolve_acquisition_config(
        AcquisitionConfig(name="entropy")
    )

    assert resolved.acqf_cls.__name__ == "qBinaryPredictiveEntropy"
