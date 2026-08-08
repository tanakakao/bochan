"""Acquisition function name resolver for the high-level bochan API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

AcqPath = tuple[str, str]


_ACQF_ALIASES: dict[str, AcqPath] = {
    "qei": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "qexpectedimprovement": (
        "botorch.acquisition.monte_carlo",
        "qExpectedImprovement",
    ),
    "ei": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "expectedimprovement": (
        "botorch.acquisition.monte_carlo",
        "qExpectedImprovement",
    ),
    "qlogei": ("botorch.acquisition.logei", "qLogExpectedImprovement"),
    "logei": ("botorch.acquisition.logei", "qLogExpectedImprovement"),
    "qlogexpectedimprovement": (
        "botorch.acquisition.logei",
        "qLogExpectedImprovement",
    ),
    "logexpectedimprovement": (
        "botorch.acquisition.logei",
        "qLogExpectedImprovement",
    ),
    "qnei": (
        "botorch.acquisition.monte_carlo",
        "qNoisyExpectedImprovement",
    ),
    "nei": (
        "botorch.acquisition.monte_carlo",
        "qNoisyExpectedImprovement",
    ),
    "qlognei": (
        "botorch.acquisition.logei",
        "qLogNoisyExpectedImprovement",
    ),
    "lognei": (
        "botorch.acquisition.logei",
        "qLogNoisyExpectedImprovement",
    ),
    "qlognoisyexpectedimprovement": (
        "botorch.acquisition.logei",
        "qLogNoisyExpectedImprovement",
    ),
    "lognoisyexpectedimprovement": (
        "botorch.acquisition.logei",
        "qLogNoisyExpectedImprovement",
    ),
    "qlogpof": (
        "botorch.acquisition.logei",
        "qLogProbabilityOfFeasibility",
    ),
    "logpof": (
        "botorch.acquisition.logei",
        "qLogProbabilityOfFeasibility",
    ),
    "qlogpf": (
        "botorch.acquisition.logei",
        "qLogProbabilityOfFeasibility",
    ),
    "logpf": (
        "botorch.acquisition.logei",
        "qLogProbabilityOfFeasibility",
    ),
    "qlogprobabilityoffeasibility": (
        "botorch.acquisition.logei",
        "qLogProbabilityOfFeasibility",
    ),
    "logprobabilityoffeasibility": (
        "botorch.acquisition.logei",
        "qLogProbabilityOfFeasibility",
    ),
    "qucb": ("botorch.acquisition.monte_carlo", "qUpperConfidenceBound"),
    "ucb": ("botorch.acquisition.monte_carlo", "qUpperConfidenceBound"),
    "qpi": (
        "botorch.acquisition.monte_carlo",
        "qProbabilityOfImprovement",
    ),
    "pi": (
        "botorch.acquisition.monte_carlo",
        "qProbabilityOfImprovement",
    ),
    "qkg": ("botorch.acquisition.knowledge_gradient", "qKnowledgeGradient"),
    "kg": ("botorch.acquisition.knowledge_gradient", "qKnowledgeGradient"),
    "qmultisteplookahead": (
        "botorch.acquisition.multi_step_lookahead",
        "qMultiStepLookahead",
    ),
    "lookahead": (
        "botorch.acquisition.multi_step_lookahead",
        "qMultiStepLookahead",
    ),
    "qehi": (
        "botorch.acquisition.multi_objective.monte_carlo",
        "qExpectedHypervolumeImprovement",
    ),
    "ehi": (
        "botorch.acquisition.multi_objective.monte_carlo",
        "qExpectedHypervolumeImprovement",
    ),
    "qehvi": (
        "botorch.acquisition.multi_objective.monte_carlo",
        "qExpectedHypervolumeImprovement",
    ),
    "ehvi": (
        "botorch.acquisition.multi_objective.monte_carlo",
        "qExpectedHypervolumeImprovement",
    ),
    "qlogehvi": (
        "botorch.acquisition.multi_objective.logei",
        "qLogExpectedHypervolumeImprovement",
    ),
    "logehvi": (
        "botorch.acquisition.multi_objective.logei",
        "qLogExpectedHypervolumeImprovement",
    ),
    "qlogexpectedhypervolumeimprovement": (
        "botorch.acquisition.multi_objective.logei",
        "qLogExpectedHypervolumeImprovement",
    ),
    "logexpectedhypervolumeimprovement": (
        "botorch.acquisition.multi_objective.logei",
        "qLogExpectedHypervolumeImprovement",
    ),
    "qnehvi": (
        "botorch.acquisition.multi_objective.monte_carlo",
        "qNoisyExpectedHypervolumeImprovement",
    ),
    "nehvi": (
        "botorch.acquisition.multi_objective.monte_carlo",
        "qNoisyExpectedHypervolumeImprovement",
    ),
    "qlognehvi": (
        "botorch.acquisition.multi_objective.logei",
        "qLogNoisyExpectedHypervolumeImprovement",
    ),
    "lognehvi": (
        "botorch.acquisition.multi_objective.logei",
        "qLogNoisyExpectedHypervolumeImprovement",
    ),
    "qlognoisyexpectedhypervolumeimprovement": (
        "botorch.acquisition.multi_objective.logei",
        "qLogNoisyExpectedHypervolumeImprovement",
    ),
    "lognoisyexpectedhypervolumeimprovement": (
        "botorch.acquisition.multi_objective.logei",
        "qLogNoisyExpectedHypervolumeImprovement",
    ),
    "qnparego": (
        "botorch.acquisition.monte_carlo",
        "qExpectedImprovement",
    ),
    "nparego": (
        "botorch.acquisition.monte_carlo",
        "qExpectedImprovement",
    ),
    "qlognparego": (
        "botorch.acquisition.multi_objective.parego",
        "qLogNParEGO",
    ),
    "lognparego": (
        "botorch.acquisition.multi_objective.parego",
        "qLogNParEGO",
    ),
}


def _normalize_acqf_name(name: str) -> str:
    """Normalize public acquisition names for lookup."""
    return (
        str(name)
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .lower()
    )


def _normalize_task_type(task_type: str | None) -> str | None:
    """Normalize task aliases while preserving the non-Gaussian API family."""
    if task_type is None:
        return None
    normalized = (
        str(task_type)
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .lower()
    )
    if normalized in {"classification", "binaryclassification", "binaryclass"}:
        return "binary"
    if normalized in {"ordinalregression"}:
        return "ordinal"
    if normalized in {"multiobjective", "multioutputregression", "hybrid"}:
        return "regression"
    if normalized in {"nongaussian", "nongp"}:
        return "nongaussian"
    if normalized in {
        "multiclassclassification",
        "multiclass",
        "multiclassclass",
    }:
        return "multiclass"
    return normalized


def _normalized_model_type(model_type: str | None) -> str:
    """Return a separator-free model type."""
    if model_type is None:
        return ""
    return (
        str(model_type)
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .lower()
    )


def _is_hetero(model_type: str | None) -> bool:
    """Return whether the model type denotes heteroscedastic modelling."""
    return "hetero" in _normalized_model_type(model_type)


def _is_non_gaussian_model_type(model_type: str | None) -> bool:
    """Infer non-Gaussian regression family from registered model type names."""
    normalized = _normalized_model_type(model_type)
    return normalized.startswith(
        ("beta", "gamma", "poisson", "negativebinomial")
    ) or any(
        token in normalized
        for token in (
            "betagp",
            "gammagp",
            "poissongp",
            "negativebinomialgp",
        )
    )


def _register(module_name: str, names: Sequence[str]) -> None:
    """Register canonical names from one module."""
    for name in names:
        _ACQF_ALIASES[_normalize_acqf_name(name)] = (module_name, name)


def _register_alias(alias: str, module_name: str, attr_name: str) -> None:
    """Register a compatibility alias."""
    _ACQF_ALIASES[_normalize_acqf_name(alias)] = (module_name, attr_name)


_register(
    "bochan.acquisition.regression.active_learning",
    [
        "qRegressionPredictiveEntropy",
        "qRegressionBALD",
        "qRegressionPosteriorVariance",
        "qRegressionNegIntegratedPosteriorVariance",
        "qRegressionIntegratedPosteriorVarianceProxy",
        "qMultiOutputRegressionPredictiveEntropy",
        "qMultiOutputRegressionBALD",
        "qMultiOutputRegressionPosteriorVariance",
        "qMultiOutputRegressionNegIntegratedPosteriorVariance",
        "qMultiOutputRegressionIntegratedPosteriorVarianceProxy",
        "qHeteroRegressionPredictiveEntropy",
        "qHeteroRegressionBALD",
        "qHeteroRegressionPosteriorVariance",
        "qHeteroRegressionNegIntegratedPosteriorVariance",
        "qHeteroRegressionIntegratedPosteriorVarianceProxy",
        "qHeteroMultiOutputRegressionPredictiveEntropy",
        "qHeteroMultiOutputRegressionBALD",
        "qHeteroMultiOutputRegressionPosteriorVariance",
        "qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy",
    ],
)
_register(
    "bochan.acquisition.regression.levelset_estimation",
    [
        "qRegressionStraddle",
        "qRegressionJointStraddle",
        "qRegressionICU",
        "qRegressionBoundaryVariance",
        "qRegressionProbabilityOfExceedance",
        "qMultiOutputRegressionStraddle",
        "qMultiOutputRegressionJointStraddle",
        "qMultiOutputRegressionICU",
        "qMultiOutputRegressionBoundaryVariance",
        "qMultiOutputRegressionProbabilityOfExceedance",
        "qHeteroRegressionStraddle",
        "qHeteroRegressionJointStraddle",
        "qHeteroRegressionICU",
        "qHeteroRegressionBoundaryVariance",
        "qHeteroRegressionProbabilityOfExceedance",
        "qHeteroMultiOutputRegressionStraddle",
        "qHeteroMultiOutputRegressionJointStraddle",
        "qHeteroMultiOutputRegressionICU",
        "qHeteroMultiOutputRegressionBoundaryVariance",
        "qHeteroMultiOutputRegressionProbabilityOfExceedance",
    ],
)
_register(
    "bochan.acquisition.regression.bayesian_optimization",
    [
        "qMultiOutputRegressionExpectedHypervolumeImprovement",
        "qMultiOutputRegressionNoisyExpectedHypervolumeImprovement",
        "qMultiOutputRegressionLogExpectedHypervolumeImprovement",
        "qMultiOutputRegressionLogNoisyExpectedHypervolumeImprovement",
        "qMultiOutputRegressionNParEGO",
        "qHeteroRegressionUpperConfidenceBound",
        "qHeteroRegressionExpectedImprovement",
        "qHeteroRegressionProbabilityOfImprovement",
        "qHeteroMultiOutputRegressionDecoupledExpectedHypervolumeImprovement",
        "qHeteroMultiOutputRegressionExpectedHypervolumeImprovement",
        "qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement",
        "qHeteroMultiOutputRegressionNParEGO",
    ],
)
_register(
    "bochan.acquisition.binary.active_learning",
    [
        "qBinaryPredictiveEntropy",
        "qBinaryBALD",
        "qBinaryJointBALD",
        "qBinaryGreedyJointBALD",
        "qBinaryProbabilityVariance",
        "qBinaryMarginUncertainty",
        "qBinaryFantasyNegIntegratedPosteriorVariance",
        "qMultiOutputBinaryPredictiveEntropy",
        "qMultiOutputBinaryProbabilityVariance",
        "qMultiOutputBinaryMarginUncertainty",
        "qMultiOutputBinaryBALD",
        "qMultiOutputBinaryIntegratedPosteriorVarianceProxy",
        "qHeteroBinaryPredictiveEntropy",
        "qHeteroBinaryBALD",
        "qHeteroBinaryProbabilityVariance",
        "qHeteroBinaryMarginUncertainty",
        "qHeteroBinaryIntegratedPosteriorVariance",
        "qHeteroMultiOutputBinaryPredictiveEntropy",
        "qHeteroMultiOutputBinaryProbabilityVariance",
        "qHeteroMultiOutputBinaryMarginUncertainty",
        "qHeteroMultiOutputBinaryBALD",
        "qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy",
    ],
)
_register(
    "bochan.acquisition.binary.levelset_estimation",
    [
        "qBinaryLatentStraddleAcquisition",
        "qBinaryJointLatentStraddleAcquisition",
        "qBinaryICUAcquisition",
        "qBinaryBoundaryVarianceAcquisition",
        "qBinaryClassEntropyAcquisition",
        "qMultiOutputBinaryLatentStraddleAcquisition",
        "qMultiOutputBinaryJointLatentStraddleAcquisition",
        "qMultiOutputBinaryClassEntropyAcquisition",
        "qMultiOutputBinaryICUAcquisition",
        "qMultiOutputBinaryBoundaryVarianceAcquisition",
        "qHeteroBinaryLatentStraddleAcquisition",
        "qHeteroBinaryICUAcquisition",
        "qHeteroBinaryBoundaryVarianceAcquisition",
        "qHeteroBinaryClassEntropyAcquisition",
        "qHeteroMultiOutputBinaryClassEntropyAcquisition",
        "qHeteroMultiOutputBinaryICUAcquisition",
        "qHeteroMultiOutputBinaryBoundaryVarianceAcquisition",
        "qHeteroMultiOutputBinaryLatentStraddleAcquisition",
        "qHeteroMultiOutputBinaryJointLatentStraddleAcquisition",
    ],
)
_register(
    "bochan.acquisition.binary.bayesian_optimization",
    [
        "qBinaryProbabilityOfFeasibility",
        "qBinaryExpectedImprovement",
        "qBinaryProbabilityOfImprovement",
        "qBinaryUpperConfidenceBound",
        "qMultiOutputBinaryProbabilityOfFeasibility",
        "qMultiOutputBinaryExpectedHypervolumeImprovement",
        "qMultiOutputBinaryNoisyExpectedHypervolumeImprovement",
        "qMultiOutputBinaryNParEGO",
        "qHeteroBinaryUpperConfidenceBound",
        "qHeteroBinaryExpectedImprovement",
        "qHeteroBinaryProbabilityOfImprovement",
        "qHeteroMultiOutputBinaryExpectedHypervolumeImprovement",
        "qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement",
        "qHeteroMultiOutputBinaryNParEGO",
    ],
)
_register(
    "bochan.acquisition.ordinal.active_learning",
    [
        "qOrdinalPredictiveEntropy",
        "qOrdinalBALD",
        "qOrdinalUtilityVariance",
        "qOrdinalMarginUncertainty",
        "qOrdinalFantasyNegIntegratedPosteriorVariance",
        "qMultiOutputOrdinalPredictiveEntropy",
        "qMultiOutputOrdinalBALD",
        "qMultiOutputOrdinalUtilityVariance",
        "qMultiOutputOrdinalMarginUncertainty",
        "qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance",
        "qMultiOutputOrdinalIntegratedPosteriorVarianceProxy",
        "qHeteroOrdinalPredictiveEntropy",
        "qHeteroOrdinalUtilityVariance",
        "qHeteroOrdinalMarginUncertainty",
        "qHeteroOrdinalBALD",
        "qHeteroOrdinalIntegratedPosteriorVariance",
        "qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy",
        "qHeteroMultiOutputOrdinalPredictiveEntropy",
        "qHeteroMultiOutputOrdinalUtilityVariance",
        "qHeteroMultiOutputOrdinalMarginUncertainty",
        "qHeteroMultiOutputOrdinalBALD",
    ],
)
_register(
    "bochan.acquisition.ordinal.levelset_estimation",
    [
        "qOrdinalLatentStraddleAcquisition",
        "qOrdinalJointLatentStraddleAcquisition",
        "qOrdinalICUAcquisition",
        "qOrdinalBoundaryVarianceAcquisition",
        "qOrdinalClassEntropyAcquisition",
        "qMultiOutputOrdinalLatentStraddleAcquisition",
        "qMultiOutputOrdinalJointLatentStraddleAcquisition",
        "qMultiOutputOrdinalICUAcquisition",
        "qMultiOutputOrdinalBoundaryVarianceAcquisition",
        "qMultiOutputOrdinalClassEntropyAcquisition",
        "qHeteroOrdinalLatentStraddleAcquisition",
        "qHeteroOrdinalICUAcquisition",
        "qHeteroOrdinalBoundaryVarianceAcquisition",
        "qHeteroOrdinalClassEntropyAcquisition",
        "qHeteroMultiOutputOrdinalProbabilityOfExceedance",
        "qHeteroMultiOutputOrdinalLevelSetUncertainty",
        "qHeteroMultiOutputOrdinalStraddle",
        "qHeteroMultiOutputOrdinalBoundaryVariance",
    ],
)
_register(
    "bochan.acquisition.ordinal.bayesian_optimization",
    [
        "qOrdinalExpectedImprovement",
        "qOrdinalProbabilityOfImprovement",
        "qOrdinalUpperConfidenceBound",
        "qOrdinalProbabilityOfFeasibility",
        "qMultiOutputOrdinalExpectedHypervolumeImprovement",
        "qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement",
        "qMultiOutputOrdinalNParEGO",
        "qHeteroOrdinalExpectedUtility",
        "qHeteroOrdinalExpectedImprovement",
        "qHeteroOrdinalProbabilityOfImprovement",
        "qHeteroOrdinalExpectedUtilityUpperConfidenceBound",
        "qHeteroMultiOutputOrdinalExpectedUtility",
        "qHeteroMultiOutputOrdinalProbabilityOfImprovement",
        "qHeteroMultiOutputOrdinalExpectedImprovement",
        "qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement",
        "qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement",
        "qHeteroMultiOutputOrdinalNParEGO",
    ],
)
_register(
    "bochan.acquisition.multiclass.active_learning",
    [
        "qMulticlassPredictiveEntropy",
        "qMulticlassBALD",
        "qMulticlassJointBALD",
        "qMulticlassGreedyJointBALD",
        "qMulticlassProbabilityVariance",
        "qMulticlassMarginUncertainty",
        "qMulticlassIntegratedPosteriorVarianceProxy",
        "qMultiOutputMulticlassPredictiveEntropy",
        "qMultiOutputMulticlassBALD",
        "qMultiOutputMulticlassJointBALD",
        "qMultiOutputMulticlassGreedyJointBALD",
        "qMultiOutputMulticlassProbabilityVariance",
        "qMultiOutputMulticlassMarginUncertainty",
        "qMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
        "qHeteroMulticlassPredictiveEntropy",
        "qHeteroMulticlassBALD",
        "qHeteroMulticlassProbabilityVariance",
        "qHeteroMulticlassMarginUncertainty",
        "qHeteroMulticlassIntegratedPosteriorVarianceProxy",
        "qHeteroMultiOutputMulticlassPredictiveEntropy",
        "qHeteroMultiOutputMulticlassBALD",
        "qHeteroMultiOutputMulticlassJointBALD",
        "qHeteroMultiOutputMulticlassGreedyJointBALD",
        "qHeteroMultiOutputMulticlassProbabilityVariance",
        "qHeteroMultiOutputMulticlassMarginUncertainty",
        "qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
    ],
)
_register(
    "bochan.acquisition.multiclass.levelset_estimation",
    [
        "qMulticlassLatentStraddleAcquisition",
        "qMulticlassJointLatentStraddleAcquisition",
        "qMulticlassICUAcquisition",
        "qMulticlassBoundaryVarianceAcquisition",
        "qMulticlassClassEntropyAcquisition",
        "qMulticlassProbabilityOfExceedance",
        "qMulticlassLevelSetUncertainty",
        "qMultiOutputMulticlassLatentStraddleAcquisition",
        "qMultiOutputMulticlassJointLatentStraddleAcquisition",
        "qMultiOutputMulticlassICUAcquisition",
        "qMultiOutputMulticlassBoundaryVarianceAcquisition",
        "qMultiOutputMulticlassClassEntropyAcquisition",
        "qMultiOutputMulticlassProbabilityOfExceedance",
        "qMultiOutputMulticlassLevelSetUncertainty",
        "qHeteroMulticlassLatentStraddleAcquisition",
        "qHeteroMulticlassJointLatentStraddleAcquisition",
        "qHeteroMulticlassICUAcquisition",
        "qHeteroMulticlassBoundaryVarianceAcquisition",
        "qHeteroMulticlassClassEntropyAcquisition",
        "qHeteroMulticlassProbabilityOfExceedance",
        "qHeteroMulticlassLevelSetUncertainty",
        "qHeteroMultiOutputMulticlassLatentStraddleAcquisition",
        "qHeteroMultiOutputMulticlassJointLatentStraddleAcquisition",
        "qHeteroMultiOutputMulticlassICUAcquisition",
        "qHeteroMultiOutputMulticlassBoundaryVarianceAcquisition",
        "qHeteroMultiOutputMulticlassClassEntropyAcquisition",
        "qHeteroMultiOutputMulticlassProbabilityOfExceedance",
        "qHeteroMultiOutputMulticlassLevelSetUncertainty",
    ],
)
_register(
    "bochan.acquisition.multiclass.bayesian_optimization",
    [
        "qMulticlassProbabilityOfFeasibility",
        "qMulticlassExpectedImprovement",
        "qMulticlassProbabilityOfImprovement",
        "qMulticlassUpperConfidenceBound",
        "qMultiOutputMulticlassProbabilityOfFeasibility",
        "qMultiOutputMulticlassExpectedHypervolumeImprovement",
        "qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement",
        "qMultiOutputMulticlassNParEGO",
        "qMultiOutputMulticlassExpectedImprovement",
        "qMultiOutputMulticlassProbabilityOfImprovement",
        "qMultiOutputMulticlassUpperConfidenceBound",
        "qHeteroMulticlassProbabilityOfFeasibility",
        "qHeteroMulticlassExpectedImprovement",
        "qHeteroMulticlassProbabilityOfImprovement",
        "qHeteroMulticlassUpperConfidenceBound",
        "qHeteroMultiOutputMulticlassProbabilityOfFeasibility",
        "qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement",
        "qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement",
        "qHeteroMultiOutputMulticlassNParEGO",
        "qHeteroMultiOutputMulticlassExpectedImprovement",
        "qHeteroMultiOutputMulticlassProbabilityOfImprovement",
        "qHeteroMultiOutputMulticlassUpperConfidenceBound",
    ],
)

_NON_GAUSSIAN_AL_NAMES = [
    "ResponseMeanVariance",
    "ExpectedObservationVariance",
    "TotalObservationVariance",
    "ExpectedObservationEntropy",
    "PredictiveEntropyProxy",
    "BALDProxy",
    "IntegratedResponseMeanVarianceProxy",
    "NegIntegratedResponseMeanVariance",
    "NegIntegratedPosteriorVariance",
    "NIPV",
    "JointBALDProxy",
    "GreedyJointBALDProxy",
]
_register(
    "bochan.acquisition.non_gaussian.active_learning",
    [
        f"{prefix}{suffix}"
        for prefix in (
            "qNonGaussian",
            "qMultiOutputNonGaussian",
            "qHeteroNonGaussian",
            "qHeteroMultiOutputNonGaussian",
        )
        for suffix in _NON_GAUSSIAN_AL_NAMES
    ]
    + ["qNonGaussianPosteriorVariance", "qNonGaussianVariance"],
)
_register(
    "bochan.acquisition.non_gaussian.levelset_estimation",
    [
        "qNonGaussianStraddle",
        "qNonGaussianJointStraddle",
        "qNonGaussianBoundaryVariance",
        "qNonGaussianICUProxy",
        "qNonGaussianProbabilityOfExceedanceProxy",
        "qNonGaussianObservationProbabilityOfExceedance",
        "qNonGaussianLevelSetUncertainty",
        "qMultiOutputNonGaussianStraddle",
        "qMultiOutputNonGaussianJointStraddle",
        "qMultiOutputNonGaussianBoundaryVariance",
        "qMultiOutputNonGaussianICUProxy",
        "qMultiOutputNonGaussianProbabilityOfExceedanceProxy",
        "qMultiOutputNonGaussianObservationProbabilityOfExceedance",
        "qMultiOutputNonGaussianLevelSetUncertainty",
        "qHeteroNonGaussianStraddle",
        "qHeteroNonGaussianJointStraddle",
        "qHeteroNonGaussianBoundaryVariance",
        "qHeteroNonGaussianICUProxy",
        "qHeteroNonGaussianProbabilityOfExceedanceProxy",
        "qHeteroNonGaussianObservationProbabilityOfExceedance",
        "qHeteroNonGaussianLevelSetUncertainty",
        "qHeteroMultiOutputNonGaussianStraddle",
        "qHeteroMultiOutputNonGaussianJointStraddle",
        "qHeteroMultiOutputNonGaussianBoundaryVariance",
        "qHeteroMultiOutputNonGaussianICUProxy",
        "qHeteroMultiOutputNonGaussianProbabilityOfExceedanceProxy",
        "qHeteroMultiOutputNonGaussianObservationProbabilityOfExceedance",
        "qHeteroMultiOutputNonGaussianLevelSetUncertainty",
        "qNonGaussianICU",
        "qNonGaussianProbabilityOfExceedance",
    ],
)

_register_alias(
    "qRegressionVariance",
    "bochan.acquisition.regression.active_learning",
    "qRegressionPosteriorVariance",
)
_register_alias(
    "qBinaryVariance",
    "bochan.acquisition.binary.active_learning",
    "qBinaryProbabilityVariance",
)
_register_alias(
    "qOrdinalVariance",
    "bochan.acquisition.ordinal.active_learning",
    "qOrdinalUtilityVariance",
)
_register_alias(
    "qMulticlassVariance",
    "bochan.acquisition.multiclass.active_learning",
    "qMulticlassProbabilityVariance",
)
_register_alias(
    "qMultiOutputMulticlassVariance",
    "bochan.acquisition.multiclass.active_learning",
    "qMultiOutputMulticlassProbabilityVariance",
)
_register_alias(
    "qHeteroMulticlassVariance",
    "bochan.acquisition.multiclass.active_learning",
    "qHeteroMulticlassProbabilityVariance",
)
_register_alias(
    "qHeteroMultiOutputMulticlassVariance",
    "bochan.acquisition.multiclass.active_learning",
    "qHeteroMultiOutputMulticlassProbabilityVariance",
)
_register_alias(
    "qNonGaussianVariance",
    "bochan.acquisition.non_gaussian.active_learning",
    "qNonGaussianResponseMeanVariance",
)

_NIPV_SHORT_NAMES = {
    "nipv",
    "qnipv",
    "negintegratedposteriorvariance",
    "qnegintegratedposteriorvariance",
    "negativeintegratedposteriorvariance",
    "qnegativeintegratedposteriorvariance",
    "negintegratedresponsemeanvariance",
    "qnegintegratedresponsemeanvariance",
}
_LOG_NEI_SHORT_NAMES = {"lognei", "qlognei"}
_LOG_POF_SHORT_NAMES = {"logpof", "qlogpof", "logpf", "qlogpf"}
_LOG_EHVI_SHORT_NAMES = {"logehvi", "qlogehvi"}
_LOG_NEHVI_SHORT_NAMES = {"lognehvi", "qlognehvi"}
_LOG_NPAREGO_SHORT_NAMES = {"lognparego", "qlognparego"}
_CONTEXTUAL_SHORT_NAMES = {
    "bald",
    "jointbald",
    "greedyjointbald",
    "predictiveentropy",
    "entropy",
    "variance",
    "posteriorvariance",
    *_NIPV_SHORT_NAMES,
    "margin",
    "marginuncertainty",
    "straddle",
    "jointstraddle",
    "icu",
    "boundaryvariance",
    "classentropy",
    "probabilityofexceedance",
    "poe",
    "levelsetuncertainty",
    "levelset",
    "ei",
    "qei",
    "expectedimprovement",
    "qexpectedimprovement",
    "pi",
    "qpi",
    "probabilityofimprovement",
    "qprobabilityofimprovement",
    "ucb",
    "qucb",
    "upperconfidencebound",
    "qupperconfidencebound",
    "pof",
    "probabilityoffeasibility",
    *_LOG_NEI_SHORT_NAMES,
    *_LOG_POF_SHORT_NAMES,
    "ehi",
    "qehi",
    "ehvi",
    "qehvi",
    "expectedhypervolumeimprovement",
    "qexpectedhypervolumeimprovement",
    *_LOG_EHVI_SHORT_NAMES,
    "nehvi",
    "qnehvi",
    "noisyexpectedhypervolumeimprovement",
    "qnoisyexpectedhypervolumeimprovement",
    *_LOG_NEHVI_SHORT_NAMES,
    "nparego",
    "qnparego",
    *_LOG_NPAREGO_SHORT_NAMES,
    "kg",
    "qkg",
    "knowledgegradient",
    "qknowledgegradient",
    "multisteplookahead",
    "qmultisteplookahead",
    "lookahead",
}


def _family_prefix(
    task_type: str,
    *,
    multi_output: bool,
    hetero: bool,
) -> str:
    """Return the canonical acquisition family prefix."""
    if task_type == "regression":
        if hetero and multi_output:
            return "qHeteroMultiOutputRegression"
        if hetero:
            return "qHeteroRegression"
        if multi_output:
            return "qMultiOutputRegression"
        return "qRegression"
    if task_type == "binary":
        if hetero and multi_output:
            return "qHeteroMultiOutputBinary"
        if hetero:
            return "qHeteroBinary"
        if multi_output:
            return "qMultiOutputBinary"
        return "qBinary"
    if task_type == "ordinal":
        if hetero and multi_output:
            return "qHeteroMultiOutputOrdinal"
        if hetero:
            return "qHeteroOrdinal"
        if multi_output:
            return "qMultiOutputOrdinal"
        return "qOrdinal"
    if task_type == "multiclass":
        if hetero and multi_output:
            return "qHeteroMultiOutputMulticlass"
        if hetero:
            return "qHeteroMulticlass"
        if multi_output:
            return "qMultiOutputMulticlass"
        return "qMulticlass"
    if task_type == "nongaussian":
        if hetero and multi_output:
            return "qHeteroMultiOutputNonGaussian"
        if hetero:
            return "qHeteroNonGaussian"
        if multi_output:
            return "qMultiOutputNonGaussian"
        return "qNonGaussian"
    raise ValueError(
        "Short acquisition alias is not supported for "
        f"task_type={task_type!r}."
    )


def _fallback_builtin_path(normalized_name: str) -> AcqPath | None:
    return _ACQF_ALIASES.get(normalized_name)


def _raise_regression_only(name: str, task: str) -> None:
    raise ValueError(
        f"Acquisition alias {name!r} is available only for regression / "
        f"hybrid models. Current task_type={task!r}."
    )


def _raise_multi_output_only(name: str, task: str) -> None:
    raise ValueError(
        f"Acquisition alias {name!r} requires a multi-output model. "
        f"Current task_type={task!r}."
    )


def _resolve_contextual_nipv_path(*, task: str, prefix: str) -> AcqPath:
    """Resolve NIPV to a task-appropriate true or proxy implementation."""
    canonical_by_prefix = {
        "qRegression": "qRegressionNegIntegratedPosteriorVariance",
        "qMultiOutputRegression": "qMultiOutputRegressionNegIntegratedPosteriorVariance",
        "qHeteroRegression": "qHeteroRegressionNegIntegratedPosteriorVariance",
        "qHeteroMultiOutputRegression": (
            "qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy"
        ),
        "qBinary": "qBinaryFantasyNegIntegratedPosteriorVariance",
        "qMultiOutputBinary": "qMultiOutputBinaryIntegratedPosteriorVarianceProxy",
        "qHeteroBinary": "qHeteroBinaryIntegratedPosteriorVariance",
        "qHeteroMultiOutputBinary": (
            "qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy"
        ),
        "qOrdinal": "qOrdinalFantasyNegIntegratedPosteriorVariance",
        "qMultiOutputOrdinal": (
            "qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance"
        ),
        "qHeteroOrdinal": "qHeteroOrdinalIntegratedPosteriorVariance",
        "qHeteroMultiOutputOrdinal": (
            "qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy"
        ),
        "qMulticlass": "qMulticlassIntegratedPosteriorVarianceProxy",
        "qMultiOutputMulticlass": (
            "qMultiOutputMulticlassIntegratedPosteriorVarianceProxy"
        ),
        "qHeteroMulticlass": (
            "qHeteroMulticlassIntegratedPosteriorVarianceProxy"
        ),
        "qHeteroMultiOutputMulticlass": (
            "qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy"
        ),
        "qNonGaussian": "qNonGaussianNegIntegratedResponseMeanVariance",
        "qMultiOutputNonGaussian": (
            "qMultiOutputNonGaussianNegIntegratedResponseMeanVariance"
        ),
        "qHeteroNonGaussian": (
            "qHeteroNonGaussianNegIntegratedResponseMeanVariance"
        ),
        "qHeteroMultiOutputNonGaussian": (
            "qHeteroMultiOutputNonGaussianNegIntegratedResponseMeanVariance"
        ),
    }
    canonical_name = canonical_by_prefix.get(prefix)
    if canonical_name is None:
        raise ValueError(
            f"NIPV is not registered for task_type={task!r} and "
            f"prefix={prefix!r}."
        )
    path = _ACQF_ALIASES.get(_normalize_acqf_name(canonical_name))
    if path is None:
        raise ValueError(
            f"NIPV resolved to {canonical_name!r}, but that acquisition is "
            "not registered."
        )
    return path


def _resolve_contextual_bo_path(
    normalized_name: str,
    *,
    task: str,
    prefix: str,
    multi_output: bool,
) -> AcqPath | None:
    """Resolve BO / integrated-variance names before AL / LSE names."""
    if normalized_name in _NIPV_SHORT_NAMES:
        return _resolve_contextual_nipv_path(task=task, prefix=prefix)
    if normalized_name in _LOG_NEI_SHORT_NAMES:
        if task != "regression":
            _raise_regression_only(normalized_name, task)
        return _fallback_builtin_path("qlognei")
    if normalized_name in _LOG_POF_SHORT_NAMES:
        if task != "regression":
            _raise_regression_only(normalized_name, task)
        return _fallback_builtin_path("qlogpof")
    if normalized_name in _LOG_EHVI_SHORT_NAMES:
        if task != "regression":
            _raise_regression_only(normalized_name, task)
        if not multi_output:
            _raise_multi_output_only(normalized_name, task)
        return _fallback_builtin_path("qlogehvi")
    if normalized_name in _LOG_NEHVI_SHORT_NAMES:
        if task != "regression":
            _raise_regression_only(normalized_name, task)
        if not multi_output:
            _raise_multi_output_only(normalized_name, task)
        return _fallback_builtin_path("qlognehvi")
    if normalized_name in _LOG_NPAREGO_SHORT_NAMES:
        if task != "regression":
            _raise_regression_only(normalized_name, task)
        if not multi_output:
            _raise_multi_output_only(normalized_name, task)
        return _fallback_builtin_path("qlognparego")
    if normalized_name in {
        "kg",
        "qkg",
        "knowledgegradient",
        "qknowledgegradient",
    }:
        if task != "regression":
            _raise_regression_only(normalized_name, task)
        return _fallback_builtin_path("qkg")
    if normalized_name in {
        "multisteplookahead",
        "qmultisteplookahead",
        "lookahead",
    }:
        if task != "regression":
            _raise_regression_only(normalized_name, task)
        return _fallback_builtin_path("qmultisteplookahead")
    if normalized_name in {
        "ei",
        "qei",
        "expectedimprovement",
        "qexpectedimprovement",
    }:
        if task in {"binary", "ordinal", "multiclass"} or prefix.startswith(
            "qHeteroRegression"
        ):
            return _ACQF_ALIASES.get(
                _normalize_acqf_name(f"{prefix}ExpectedImprovement")
            )
        return _fallback_builtin_path("qei")
    if normalized_name in {
        "pi",
        "qpi",
        "probabilityofimprovement",
        "qprobabilityofimprovement",
    }:
        if task in {"binary", "ordinal", "multiclass"} or prefix.startswith(
            "qHeteroRegression"
        ):
            return _ACQF_ALIASES.get(
                _normalize_acqf_name(f"{prefix}ProbabilityOfImprovement")
            )
        return _fallback_builtin_path("qpi")
    if normalized_name in {
        "ucb",
        "qucb",
        "upperconfidencebound",
        "qupperconfidencebound",
    }:
        if task in {"binary", "ordinal", "multiclass"} or prefix.startswith(
            "qHeteroRegression"
        ):
            return _ACQF_ALIASES.get(
                _normalize_acqf_name(f"{prefix}UpperConfidenceBound")
            )
        return _fallback_builtin_path("qucb")
    if normalized_name in {"pof", "probabilityoffeasibility"}:
        if task in {"binary", "ordinal", "multiclass"}:
            return _ACQF_ALIASES.get(
                _normalize_acqf_name(f"{prefix}ProbabilityOfFeasibility")
            )
        return None
    if normalized_name in {
        "ehi",
        "qehi",
        "ehvi",
        "qehvi",
        "expectedhypervolumeimprovement",
        "qexpectedhypervolumeimprovement",
    }:
        if task in {"binary", "ordinal", "multiclass"} or prefix.startswith(
            "qHeteroMultiOutputRegression"
        ):
            if not multi_output:
                return None
            return _ACQF_ALIASES.get(
                _normalize_acqf_name(
                    f"{prefix}ExpectedHypervolumeImprovement"
                )
            )
        return _fallback_builtin_path("qehvi")
    if normalized_name in {
        "nehvi",
        "qnehvi",
        "noisyexpectedhypervolumeimprovement",
        "qnoisyexpectedhypervolumeimprovement",
    }:
        if task in {"binary", "ordinal", "multiclass"} or prefix.startswith(
            "qHeteroMultiOutputRegression"
        ):
            if not multi_output:
                return None
            return _ACQF_ALIASES.get(
                _normalize_acqf_name(
                    f"{prefix}NoisyExpectedHypervolumeImprovement"
                )
            )
        return _fallback_builtin_path("qnehvi")
    if normalized_name in {"nparego", "qnparego"}:
        if not multi_output:
            return None
        if task in {"binary", "ordinal", "multiclass"} or prefix.startswith(
            "qHeteroMultiOutputRegression"
        ):
            return _ACQF_ALIASES.get(
                _normalize_acqf_name(f"{prefix}NParEGO")
            )
        if task == "regression":
            return _ACQF_ALIASES.get(
                _normalize_acqf_name("qMultiOutputRegressionNParEGO")
            )
        return _fallback_builtin_path("qnparego")
    return None


def _resolve_contextual_acqf_path(
    normalized_name: str,
    *,
    task_type: str | None = None,
    model_type: str | None = None,
    multi_output: bool = False,
) -> AcqPath | None:
    """Resolve a short name using task, posterior family, and output shape."""
    if normalized_name not in _CONTEXTUAL_SHORT_NAMES:
        return None
    task = _normalize_task_type(task_type)
    if task is None:
        raise ValueError(
            f"Acquisition name {normalized_name!r} is task-dependent. Use a "
            "canonical name or call through BayesianOptimizer after fit()."
        )
    if task == "regression" and _is_non_gaussian_model_type(model_type):
        task = "nongaussian"
    hetero = _is_hetero(model_type)
    prefix = _family_prefix(
        task,
        multi_output=multi_output,
        hetero=hetero,
    )
    bo_path = _resolve_contextual_bo_path(
        normalized_name,
        task=task,
        prefix=prefix,
        multi_output=multi_output,
    )
    if bo_path is not None:
        return bo_path
    if normalized_name == "bald":
        suffix = "BALD" if task != "nongaussian" else "BALDProxy"
    elif normalized_name == "jointbald":
        suffix = "JointBALDProxy" if task == "nongaussian" else "JointBALD"
    elif normalized_name == "greedyjointbald":
        suffix = (
            "GreedyJointBALDProxy"
            if task == "nongaussian"
            else "GreedyJointBALD"
        )
    elif normalized_name in {"predictiveentropy", "entropy"}:
        suffix = (
            "PredictiveEntropy"
            if task != "nongaussian"
            else "PredictiveEntropyProxy"
        )
    elif normalized_name in {"variance", "posteriorvariance"}:
        if task in {"binary", "multiclass"}:
            suffix = "ProbabilityVariance"
        elif task == "ordinal":
            suffix = "UtilityVariance"
        elif task == "nongaussian":
            suffix = "ResponseMeanVariance"
        else:
            suffix = "PosteriorVariance"
    elif normalized_name in {"margin", "marginuncertainty"}:
        if task not in {"binary", "ordinal", "multiclass"}:
            raise ValueError(
                "Margin uncertainty is currently task-dependent for binary / "
                "ordinal / multiclass models only."
            )
        suffix = "MarginUncertainty"
    elif normalized_name == "straddle":
        suffix = (
            "LatentStraddleAcquisition"
            if task in {"binary", "ordinal", "multiclass"}
            else "Straddle"
        )
    elif normalized_name == "jointstraddle":
        suffix = (
            "JointLatentStraddleAcquisition"
            if task in {"binary", "ordinal", "multiclass"}
            else "JointStraddle"
        )
    elif normalized_name == "icu":
        suffix = (
            "ICUAcquisition"
            if task in {"binary", "ordinal", "multiclass"}
            else ("ICUProxy" if task == "nongaussian" else "ICU")
        )
    elif normalized_name == "boundaryvariance":
        suffix = (
            "BoundaryVarianceAcquisition"
            if task in {"binary", "ordinal", "multiclass"}
            else "BoundaryVariance"
        )
    elif normalized_name == "classentropy":
        if task not in {"binary", "ordinal", "multiclass"}:
            raise ValueError(
                "Class entropy is only available for binary / ordinal / "
                "multiclass acquisitions."
            )
        suffix = "ClassEntropyAcquisition"
    elif normalized_name in {"probabilityofexceedance", "poe"}:
        suffix = (
            "ProbabilityOfExceedanceProxy"
            if task == "nongaussian"
            else "ProbabilityOfExceedance"
        )
    elif normalized_name in {"levelsetuncertainty", "levelset"}:
        suffix = "LevelSetUncertainty"
    else:
        return None
    canonical_name = f"{prefix}{suffix}"
    path = _ACQF_ALIASES.get(_normalize_acqf_name(canonical_name))
    if path is None:
        raise ValueError(
            f"Short acquisition alias {normalized_name!r} resolved to "
            f"{canonical_name!r}, but that acquisition is not registered."
        )
    return path


def _import_from_path(module_name: str, attr_name: str) -> Any:
    """Import one registered acquisition lazily."""
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def resolve_acqf_cls(
    name: str,
    acquisition_registry: Mapping[str, Any] | None = None,
    *,
    task_type: str | None = None,
    model_type: str | None = None,
    multi_output: bool = False,
) -> type | Callable[..., Any]:
    """Resolve a canonical or contextual acquisition name."""
    normalized = _normalize_acqf_name(name)
    if acquisition_registry is not None:
        if name in acquisition_registry:
            value = acquisition_registry[name]
        elif normalized in acquisition_registry:
            value = acquisition_registry[normalized]
        else:
            value = None
        if value is not None:
            if isinstance(value, tuple) and len(value) == 2:
                return _import_from_path(value[0], value[1])
            return value
    contextual_path = _resolve_contextual_acqf_path(
        normalized,
        task_type=task_type,
        model_type=model_type,
        multi_output=multi_output,
    )
    if contextual_path is not None:
        return _import_from_path(*contextual_path)
    if normalized not in _ACQF_ALIASES:
        available = sorted(_ACQF_ALIASES)
        raise ValueError(
            f"Unknown acquisition function name: {name!r}. Available built-in "
            f"aliases include: {available}. For custom acquisitions, pass "
            "acquisition_registry."
        )
    return _import_from_path(*_ACQF_ALIASES[normalized])


def available_acqf_names() -> list[str]:
    """Return canonical and contextual acquisition names."""
    return sorted(
        _ACQF_ALIASES
        | {name: ("", "") for name in _CONTEXTUAL_SHORT_NAMES}
    )


__all__ = ["available_acqf_names", "resolve_acqf_cls"]
