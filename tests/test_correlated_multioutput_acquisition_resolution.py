from types import SimpleNamespace

import pytest

from bochan.api import AcquisitionConfig, BayesianOptimizer


def _make_optimizer(task_type: str, *, num_outputs: int = 2) -> BayesianOptimizer:
    optimizer = object.__new__(BayesianOptimizer)
    optimizer.acquisition_registry = None
    optimizer.model = SimpleNamespace(num_outputs=num_outputs)
    optimizer.bundle = SimpleNamespace(
        model=optimizer.model,
        task_type=task_type,
        model_type="kronecker" if num_outputs > 1 else "single_task",
        metadata={},
    )
    return optimizer


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
    optimizer = _make_optimizer("binary")
    resolved = optimizer._resolve_acquisition_config(AcquisitionConfig(name=alias))
    assert resolved.acqf_cls.__name__ == expected_name


@pytest.mark.parametrize(
    ("alias", "expected_name"),
    [
        ("bald", "qMultiOutputOrdinalBALD"),
        ("entropy", "qMultiOutputOrdinalPredictiveEntropy"),
        ("variance", "qMultiOutputOrdinalUtilityVariance"),
        ("margin", "qMultiOutputOrdinalMarginUncertainty"),
        ("nipv", "qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance"),
        ("straddle", "qMultiOutputOrdinalLatentStraddleAcquisition"),
        ("jointstraddle", "qMultiOutputOrdinalJointLatentStraddleAcquisition"),
        ("icu", "qMultiOutputOrdinalICUAcquisition"),
        ("boundaryvariance", "qMultiOutputOrdinalBoundaryVarianceAcquisition"),
        ("classentropy", "qMultiOutputOrdinalClassEntropyAcquisition"),
        ("ehvi", "qMultiOutputOrdinalExpectedHypervolumeImprovement"),
        ("nehvi", "qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement"),
        ("nparego", "qMultiOutputOrdinalNParEGO"),
    ],
)
def test_kronecker_ordinal_aliases_use_multioutput_family(alias, expected_name):
    optimizer = _make_optimizer("ordinal")
    resolved = optimizer._resolve_acquisition_config(AcquisitionConfig(name=alias))
    assert resolved.acqf_cls.__name__ == expected_name


@pytest.mark.parametrize(
    ("alias", "expected_name"),
    [
        ("bald", "qMultiOutputMulticlassBALD"),
        ("jointbald", "qMultiOutputMulticlassJointBALD"),
        ("greedyjointbald", "qMultiOutputMulticlassGreedyJointBALD"),
        ("entropy", "qMultiOutputMulticlassPredictiveEntropy"),
        ("variance", "qMultiOutputMulticlassProbabilityVariance"),
        ("margin", "qMultiOutputMulticlassMarginUncertainty"),
        ("nipv", "qMultiOutputMulticlassIntegratedPosteriorVarianceProxy"),
        ("straddle", "qMultiOutputMulticlassLatentStraddleAcquisition"),
        ("jointstraddle", "qMultiOutputMulticlassJointLatentStraddleAcquisition"),
        ("icu", "qMultiOutputMulticlassICUAcquisition"),
        ("boundaryvariance", "qMultiOutputMulticlassBoundaryVarianceAcquisition"),
        ("classentropy", "qMultiOutputMulticlassClassEntropyAcquisition"),
        ("poe", "qMultiOutputMulticlassProbabilityOfExceedance"),
        ("levelset", "qMultiOutputMulticlassLevelSetUncertainty"),
        ("ei", "qMultiOutputMulticlassExpectedImprovement"),
        ("pi", "qMultiOutputMulticlassProbabilityOfImprovement"),
        ("ucb", "qMultiOutputMulticlassUpperConfidenceBound"),
        ("ehvi", "qMultiOutputMulticlassExpectedHypervolumeImprovement"),
        ("nehvi", "qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement"),
        ("nparego", "qMultiOutputMulticlassNParEGO"),
    ],
)
def test_kronecker_multiclass_aliases_use_multioutput_family(alias, expected_name):
    optimizer = _make_optimizer("multiclass")
    resolved = optimizer._resolve_acquisition_config(AcquisitionConfig(name=alias))
    assert resolved.acqf_cls.__name__ == expected_name


@pytest.mark.parametrize(
    ("task_type", "expected_name"),
    [
        ("binary", "qBinaryPredictiveEntropy"),
        ("ordinal", "qOrdinalPredictiveEntropy"),
        ("multiclass", "qMulticlassPredictiveEntropy"),
    ],
)
def test_single_output_alias_stays_single_output(task_type, expected_name):
    optimizer = _make_optimizer(task_type, num_outputs=1)
    resolved = optimizer._resolve_acquisition_config(
        AcquisitionConfig(name="entropy")
    )
    assert resolved.acqf_cls.__name__ == expected_name
