"""Acquisition function name resolver for the high-level bochan API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

AcqPath = tuple[str, str]


_ACQF_ALIASES: dict[str, AcqPath] = {
    "qei": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "qexpectedimprovement": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "ei": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "expectedimprovement": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "qlogei": ("botorch.acquisition.logei", "qLogExpectedImprovement"),
    "logei": ("botorch.acquisition.logei", "qLogExpectedImprovement"),
    "qnei": ("botorch.acquisition.monte_carlo", "qNoisyExpectedImprovement"),
    "nei": ("botorch.acquisition.monte_carlo", "qNoisyExpectedImprovement"),
    "qucb": ("botorch.acquisition.monte_carlo", "qUpperConfidenceBound"),
    "ucb": ("botorch.acquisition.monte_carlo", "qUpperConfidenceBound"),
    "qpi": ("botorch.acquisition.monte_carlo", "qProbabilityOfImprovement"),
    "pi": ("botorch.acquisition.monte_carlo", "qProbabilityOfImprovement"),
    "qkg": ("botorch.acquisition.knowledge_gradient", "qKnowledgeGradient"),
    "kg": ("botorch.acquisition.knowledge_gradient", "qKnowledgeGradient"),
    "qmultisteplookahead": ("botorch.acquisition.multi_step_lookahead", "qMultiStepLookahead"),
    "lookahead": ("botorch.acquisition.multi_step_lookahead", "qMultiStepLookahead"),
    "qehi": ("botorch.acquisition.multi_objective.monte_carlo", "qExpectedHypervolumeImprovement"),
    "ehi": ("botorch.acquisition.multi_objective.monte_carlo", "qExpectedHypervolumeImprovement"),
    "qehvi": ("botorch.acquisition.multi_objective.monte_carlo", "qExpectedHypervolumeImprovement"),
    "ehvi": ("botorch.acquisition.multi_objective.monte_carlo", "qExpectedHypervolumeImprovement"),
    "qnehvi": ("botorch.acquisition.multi_objective.monte_carlo", "qNoisyExpectedHypervolumeImprovement"),
    "nehvi": ("botorch.acquisition.multi_objective.monte_carlo", "qNoisyExpectedHypervolumeImprovement"),
    "qnparego": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "nparego": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
}


def _normalize_acqf_name(name: str) -> str:
    return str(name).replace("_", "").replace("-", "").replace(" ", "").lower()


def _normalize_task_type(task_type: str | None) -> str | None:
    if task_type is None:
        return None
    normalized = str(task_type).replace("_", "").replace("-", "").replace(" ", "").lower()
    if normalized in {"classification", "binaryclassification", "binaryclass"}:
        return "binary"
    if normalized in {"ordinalregression"}:
        return "ordinal"
    if normalized in {"multiobjective", "multioutputregression", "hybrid"}:
        return "regression"
    if normalized in {"nongaussian", "nongp"}:
        return "nongaussian"
    if normalized in {"multiclassclassification", "multiclass", "multiclassclass"}:
        return "multiclass"
    return normalized


def _is_hetero(model_type: str | None) -> bool:
    return model_type is not None and "hetero" in str(model_type).replace("_", "").replace("-", "").lower()


def _register(module_name: str, names: Sequence[str]) -> None:
    for name in names:
        _ACQF_ALIASES[_normalize_acqf_name(name)] = (module_name, name)


def _register_alias(alias: str, module_name: str, attr_name: str) -> None:
    _ACQF_ALIASES[_normalize_acqf_name(alias)] = (module_name, attr_name)


_register("bochan.acquisition.regression.active_learning", [
    "qRegressionPredictiveEntropy", "qRegressionBALD", "qRegressionPosteriorVariance",
    "qRegressionNegIntegratedPosteriorVariance", "qRegressionIntegratedPosteriorVarianceProxy",
    "qMultiOutputRegressionPredictiveEntropy", "qMultiOutputRegressionBALD", "qMultiOutputRegressionPosteriorVariance",
    "qMultiOutputRegressionNegIntegratedPosteriorVariance", "qMultiOutputRegressionIntegratedPosteriorVarianceProxy",
    "qHeteroRegressionPredictiveEntropy", "qHeteroRegressionBALD", "qHeteroRegressionPosteriorVariance",
    "qHeteroRegressionNegIntegratedPosteriorVariance", "qHeteroRegressionIntegratedPosteriorVarianceProxy",
    "qHeteroMultiOutputRegressionPredictiveEntropy", "qHeteroMultiOutputRegressionBALD",
    "qHeteroMultiOutputRegressionPosteriorVariance", "qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy",
])
_register("bochan.acquisition.regression.levelset_estimation", [
    "qRegressionStraddle", "qRegressionJointStraddle", "qRegressionICU", "qRegressionBoundaryVariance",
    "qRegressionProbabilityOfExceedance", "qMultiOutputRegressionStraddle", "qMultiOutputRegressionJointStraddle",
    "qMultiOutputRegressionICU", "qMultiOutputRegressionBoundaryVariance", "qMultiOutputRegressionProbabilityOfExceedance",
    "qHeteroRegressionStraddle", "qHeteroRegressionJointStraddle", "qHeteroRegressionICU", "qHeteroRegressionBoundaryVariance",
    "qHeteroRegressionProbabilityOfExceedance", "qHeteroMultiOutputRegressionStraddle", "qHeteroMultiOutputRegressionJointStraddle",
    "qHeteroMultiOutputRegressionICU", "qHeteroMultiOutputRegressionBoundaryVariance", "qHeteroMultiOutputRegressionProbabilityOfExceedance",
])
_register("bochan.acquisition.regression.bayesian_optimization", [
    "qHeteroRegressionUpperConfidenceBound", "qHeteroRegressionExpectedImprovement", "qHeteroRegressionProbabilityOfImprovement",
    "qHeteroMultiOutputRegressionDecoupledExpectedHypervolumeImprovement", "qHeteroMultiOutputRegressionExpectedHypervolumeImprovement",
    "qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement", "qHeteroMultiOutputRegressionNParEGO",
])
_register("bochan.acquisition.binary.active_learning", [
    "qBinaryPredictiveEntropy", "qBinaryBALD", "qBinaryJointBALD", "qBinaryGreedyJointBALD", "qBinaryProbabilityVariance",
    "qBinaryMarginUncertainty", "qBinaryFantasyNegIntegratedPosteriorVariance", "qMultiOutputBinaryPredictiveEntropy",
    "qMultiOutputBinaryProbabilityVariance", "qMultiOutputBinaryMarginUncertainty", "qMultiOutputBinaryBALD",
    "qMultiOutputBinaryIntegratedPosteriorVarianceProxy", "qHeteroBinaryPredictiveEntropy", "qHeteroBinaryBALD",
    "qHeteroBinaryProbabilityVariance", "qHeteroBinaryMarginUncertainty", "qHeteroBinaryIntegratedPosteriorVariance",
    "qHeteroMultiOutputBinaryPredictiveEntropy", "qHeteroMultiOutputBinaryProbabilityVariance",
    "qHeteroMultiOutputBinaryMarginUncertainty", "qHeteroMultiOutputBinaryBALD", "qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy",
])
_register("bochan.acquisition.binary.levelset_estimation", [
    "qBinaryLatentStraddleAcquisition", "qBinaryJointLatentStraddleAcquisition", "qBinaryICUAcquisition",
    "qBinaryBoundaryVarianceAcquisition", "qBinaryClassEntropyAcquisition", "qMultiOutputBinaryLatentStraddleAcquisition",
    "qMultiOutputBinaryJointLatentStraddleAcquisition", "qMultiOutputBinaryClassEntropyAcquisition",
    "qMultiOutputBinaryICUAcquisition", "qMultiOutputBinaryBoundaryVarianceAcquisition", "qHeteroBinaryLatentStraddleAcquisition",
    "qHeteroBinaryICUAcquisition", "qHeteroBinaryBoundaryVarianceAcquisition", "qHeteroBinaryClassEntropyAcquisition",
    "qHeteroMultiOutputBinaryClassEntropyAcquisition", "qHeteroMultiOutputBinaryICUAcquisition",
    "qHeteroMultiOutputBinaryBoundaryVarianceAcquisition", "qHeteroMultiOutputBinaryLatentStraddleAcquisition",
    "qHeteroMultiOutputBinaryJointLatentStraddleAcquisition",
])
_register("bochan.acquisition.binary.bayesian_optimization", [
    "qBinaryProbabilityOfFeasibility", "qBinaryExpectedImprovement", "qBinaryProbabilityOfImprovement", "qBinaryUpperConfidenceBound",
    "qMultiOutputBinaryProbabilityOfFeasibility", "qMultiOutputBinaryExpectedHypervolumeImprovement",
    "qMultiOutputBinaryNoisyExpectedHypervolumeImprovement", "qMultiOutputBinaryNParEGO", "qHeteroBinaryUpperConfidenceBound",
    "qHeteroBinaryExpectedImprovement", "qHeteroBinaryProbabilityOfImprovement", "qHeteroMultiOutputBinaryExpectedHypervolumeImprovement",
    "qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement", "qHeteroMultiOutputBinaryNParEGO",
])
_register("bochan.acquisition.ordinal.active_learning", [
    "qOrdinalPredictiveEntropy", "qOrdinalBALD", "qOrdinalUtilityVariance", "qOrdinalMarginUncertainty",
    "qOrdinalFantasyNegIntegratedPosteriorVariance", "qMultiOutputOrdinalPredictiveEntropy", "qMultiOutputOrdinalBALD",
    "qMultiOutputOrdinalUtilityVariance", "qMultiOutputOrdinalMarginUncertainty", "qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance",
    "qMultiOutputOrdinalIntegratedPosteriorVarianceProxy", "qHeteroOrdinalPredictiveEntropy", "qHeteroOrdinalUtilityVariance",
    "qHeteroOrdinalMarginUncertainty", "qHeteroOrdinalBALD", "qHeteroOrdinalIntegratedPosteriorVariance",
    "qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy", "qHeteroMultiOutputOrdinalPredictiveEntropy",
    "qHeteroMultiOutputOrdinalUtilityVariance", "qHeteroMultiOutputOrdinalMarginUncertainty", "qHeteroMultiOutputOrdinalBALD",
])
_register("bochan.acquisition.ordinal.levelset_estimation", [
    "qOrdinalLatentStraddleAcquisition", "qOrdinalJointLatentStraddleAcquisition", "qOrdinalICUAcquisition",
    "qOrdinalBoundaryVarianceAcquisition", "qOrdinalClassEntropyAcquisition", "qMultiOutputOrdinalLatentStraddleAcquisition",
    "qMultiOutputOrdinalJointLatentStraddleAcquisition", "qMultiOutputOrdinalICUAcquisition", "qMultiOutputOrdinalBoundaryVarianceAcquisition",
    "qMultiOutputOrdinalClassEntropyAcquisition", "qHeteroOrdinalLatentStraddleAcquisition", "qHeteroOrdinalICUAcquisition",
    "qHeteroOrdinalBoundaryVarianceAcquisition", "qHeteroOrdinalClassEntropyAcquisition", "qHeteroMultiOutputOrdinalProbabilityOfExceedance",
    "qHeteroMultiOutputOrdinalLevelSetUncertainty", "qHeteroMultiOutputOrdinalStraddle", "qHeteroMultiOutputOrdinalBoundaryVariance",
])
_register("bochan.acquisition.ordinal.bayesian_optimization", [
    "qOrdinalExpectedImprovement", "qOrdinalProbabilityOfImprovement", "qOrdinalUpperConfidenceBound", "qOrdinalProbabilityOfFeasibility",
    "qMultiOutputOrdinalExpectedHypervolumeImprovement", "qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement", "qMultiOutputOrdinalNParEGO",
    "qHeteroOrdinalExpectedUtility", "qHeteroOrdinalExpectedImprovement", "qHeteroOrdinalProbabilityOfImprovement",
    "qHeteroOrdinalExpectedUtilityUpperConfidenceBound", "qHeteroMultiOutputOrdinalExpectedUtility", "qHeteroMultiOutputOrdinalProbabilityOfImprovement",
    "qHeteroMultiOutputOrdinalExpectedImprovement", "qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement",
    "qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement", "qHeteroMultiOutputOrdinalNParEGO",
])
_register("bochan.acquisition.multiclass.active_learning", [
    "qMulticlassPredictiveEntropy", "qMulticlassBALD", "qMulticlassJointBALD", "qMulticlassGreedyJointBALD",
    "qMulticlassProbabilityVariance", "qMulticlassMarginUncertainty", "qMulticlassIntegratedPosteriorVarianceProxy",
    "qMultiOutputMulticlassPredictiveEntropy", "qMultiOutputMulticlassBALD", "qMultiOutputMulticlassProbabilityVariance",
    "qMultiOutputMulticlassMarginUncertainty", "qMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
    "qHeteroMulticlassPredictiveEntropy", "qHeteroMulticlassBALD", "qHeteroMulticlassProbabilityVariance",
    "qHeteroMulticlassMarginUncertainty", "qHeteroMulticlassIntegratedPosteriorVarianceProxy",
    "qHeteroMultiOutputMulticlassPredictiveEntropy", "qHeteroMultiOutputMulticlassBALD",
    "qHeteroMultiOutputMulticlassProbabilityVariance", "qHeteroMultiOutputMulticlassMarginUncertainty",
    "qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
])
_register("bochan.acquisition.multiclass.levelset_estimation", [
    "qMulticlassLatentStraddleAcquisition", "qMulticlassJointLatentStraddleAcquisition", "qMulticlassICUAcquisition",
    "qMulticlassBoundaryVarianceAcquisition", "qMulticlassClassEntropyAcquisition", "qMulticlassProbabilityOfExceedance",
    "qMulticlassLevelSetUncertainty", "qMultiOutputMulticlassLatentStraddleAcquisition",
    "qMultiOutputMulticlassJointLatentStraddleAcquisition", "qMultiOutputMulticlassICUAcquisition",
    "qMultiOutputMulticlassBoundaryVarianceAcquisition", "qMultiOutputMulticlassClassEntropyAcquisition",
    "qMultiOutputMulticlassProbabilityOfExceedance", "qMultiOutputMulticlassLevelSetUncertainty",
    "qHeteroMulticlassLatentStraddleAcquisition", "qHeteroMulticlassJointLatentStraddleAcquisition",
    "qHeteroMulticlassICUAcquisition", "qHeteroMulticlassBoundaryVarianceAcquisition",
    "qHeteroMulticlassClassEntropyAcquisition", "qHeteroMulticlassProbabilityOfExceedance",
    "qHeteroMulticlassLevelSetUncertainty", "qHeteroMultiOutputMulticlassLatentStraddleAcquisition",
    "qHeteroMultiOutputMulticlassJointLatentStraddleAcquisition", "qHeteroMultiOutputMulticlassICUAcquisition",
    "qHeteroMultiOutputMulticlassBoundaryVarianceAcquisition", "qHeteroMultiOutputMulticlassClassEntropyAcquisition",
    "qHeteroMultiOutputMulticlassProbabilityOfExceedance", "qHeteroMultiOutputMulticlassLevelSetUncertainty",
])
_register("bochan.acquisition.multiclass.bayesian_optimization", [
    "qMulticlassProbabilityOfFeasibility", "qMulticlassExpectedImprovement", "qMulticlassProbabilityOfImprovement", "qMulticlassUpperConfidenceBound",
    "qMultiOutputMulticlassProbabilityOfFeasibility", "qMultiOutputMulticlassExpectedImprovement",
    "qMultiOutputMulticlassProbabilityOfImprovement", "qMultiOutputMulticlassUpperConfidenceBound",
    "qHeteroMulticlassProbabilityOfFeasibility", "qHeteroMulticlassExpectedImprovement",
    "qHeteroMulticlassProbabilityOfImprovement", "qHeteroMulticlassUpperConfidenceBound",
    "qHeteroMultiOutputMulticlassProbabilityOfFeasibility", "qHeteroMultiOutputMulticlassExpectedImprovement",
    "qHeteroMultiOutputMulticlassProbabilityOfImprovement", "qHeteroMultiOutputMulticlassUpperConfidenceBound",
])
_register("bochan.acquisition.non_gaussian.active_learning", [
    "qNonGaussianResponseMeanVariance", "qNonGaussianPosteriorVariance", "qNonGaussianExpectedObservationVariance",
    "qNonGaussianTotalObservationVariance", "qNonGaussianExpectedObservationEntropy", "qNonGaussianPredictiveEntropyProxy", "qNonGaussianBALDProxy",
])
_register("bochan.acquisition.non_gaussian.levelset_estimation", [
    "qNonGaussianStraddle", "qNonGaussianBoundaryVariance", "qNonGaussianICU", "qNonGaussianProbabilityOfExceedance",
])

_register_alias("qRegressionVariance", "bochan.acquisition.regression.active_learning", "qRegressionPosteriorVariance")
_register_alias("qBinaryVariance", "bochan.acquisition.binary.active_learning", "qBinaryProbabilityVariance")
_register_alias("qOrdinalVariance", "bochan.acquisition.ordinal.active_learning", "qOrdinalUtilityVariance")
_register_alias("qMulticlassVariance", "bochan.acquisition.multiclass.active_learning", "qMulticlassProbabilityVariance")
_register_alias("qMultiOutputMulticlassVariance", "bochan.acquisition.multiclass.active_learning", "qMultiOutputMulticlassProbabilityVariance")
_register_alias("qHeteroMulticlassVariance", "bochan.acquisition.multiclass.active_learning", "qHeteroMulticlassProbabilityVariance")
_register_alias("qHeteroMultiOutputMulticlassVariance", "bochan.acquisition.multiclass.active_learning", "qHeteroMultiOutputMulticlassProbabilityVariance")
_register_alias("qNonGaussianVariance", "bochan.acquisition.non_gaussian.active_learning", "qNonGaussianResponseMeanVariance")

_NIPV_SHORT_NAMES = {"nipv", "qnipv", "negintegratedposteriorvariance", "qnegintegratedposteriorvariance", "negativeintegratedposteriorvariance", "qnegativeintegratedposteriorvariance"}
_CONTEXTUAL_SHORT_NAMES = {
    "bald", "predictiveentropy", "entropy", "variance", "posteriorvariance", *_NIPV_SHORT_NAMES,
    "margin", "marginuncertainty", "straddle", "jointstraddle", "icu", "boundaryvariance",
    "classentropy", "probabilityofexceedance", "poe", "levelsetuncertainty", "levelset",
    "ei", "qei", "expectedimprovement", "qexpectedimprovement", "pi", "qpi", "probabilityofimprovement",
    "qprobabilityofimprovement", "ucb", "qucb", "upperconfidencebound", "qupperconfidencebound", "pof",
    "probabilityoffeasibility", "ehi", "qehi", "ehvi", "qehvi", "expectedhypervolumeimprovement",
    "qexpectedhypervolumeimprovement", "nehvi", "qnehvi", "noisyexpectedhypervolumeimprovement",
    "qnoisyexpectedhypervolumeimprovement", "nparego", "qnparego", "kg", "qkg", "knowledgegradient",
    "qknowledgegradient", "multisteplookahead", "qmultisteplookahead", "lookahead",
}


def _family_prefix(task_type: str, *, multi_output: bool, hetero: bool) -> str:
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
        return "qNonGaussian"
    raise ValueError(f"Short acquisition alias is not supported for task_type={task_type!r}.")


def _fallback_builtin_path(normalized_name: str) -> AcqPath | None:
    return _ACQF_ALIASES.get(normalized_name)


def _raise_regression_only(name: str, task: str) -> None:
    raise ValueError(f"Acquisition alias {name!r} is available only for regression / hybrid models. Current task_type={task!r}.")


def _resolve_contextual_nipv_path(*, task: str, prefix: str) -> AcqPath:
    canonical_by_prefix = {
        "qRegression": "qRegressionNegIntegratedPosteriorVariance",
        "qMultiOutputRegression": "qMultiOutputRegressionNegIntegratedPosteriorVariance",
        "qHeteroRegression": "qHeteroRegressionNegIntegratedPosteriorVariance",
        "qHeteroMultiOutputRegression": "qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy",
        "qBinary": "qBinaryFantasyNegIntegratedPosteriorVariance",
        "qMultiOutputBinary": "qMultiOutputBinaryIntegratedPosteriorVarianceProxy",
        "qHeteroBinary": "qHeteroBinaryIntegratedPosteriorVariance",
        "qHeteroMultiOutputBinary": "qHeteroMultiOutputBinaryIntegratedPosteriorVarianceProxy",
        "qOrdinal": "qOrdinalFantasyNegIntegratedPosteriorVariance",
        "qMultiOutputOrdinal": "qMultiOutputOrdinalFantasyNegIntegratedPosteriorVariance",
        "qHeteroOrdinal": "qHeteroOrdinalIntegratedPosteriorVariance",
        "qHeteroMultiOutputOrdinal": "qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy",
        "qMulticlass": "qMulticlassIntegratedPosteriorVarianceProxy",
        "qMultiOutputMulticlass": "qMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
        "qHeteroMulticlass": "qHeteroMulticlassIntegratedPosteriorVarianceProxy",
        "qHeteroMultiOutputMulticlass": "qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
    }
    canonical_name = canonical_by_prefix.get(prefix)
    if canonical_name is None:
        raise ValueError(f"NIPV is not registered for task_type={task!r} and prefix={prefix!r}.")
    path = _ACQF_ALIASES.get(_normalize_acqf_name(canonical_name))
    if path is None:
        raise ValueError(f"NIPV resolved to {canonical_name!r}, but that acquisition is not registered.")
    return path


def _resolve_contextual_bo_path(normalized_name: str, *, task: str, prefix: str, multi_output: bool) -> AcqPath | None:
    if normalized_name in _NIPV_SHORT_NAMES:
        return _resolve_contextual_nipv_path(task=task, prefix=prefix)
    if normalized_name in {"kg", "qkg", "knowledgegradient", "qknowledgegradient"}:
        if task != "regression":
            _raise_regression_only(normalized_name, task)
        return _fallback_builtin_path("qkg")
    if normalized_name in {"multisteplookahead", "qmultisteplookahead", "lookahead"}:
        if task != "regression":
            _raise_regression_only(normalized_name, task)
        return _fallback_builtin_path("qmultisteplookahead")
    if normalized_name in {"ei", "qei", "expectedimprovement", "qexpectedimprovement"}:
        if task in {"binary", "ordinal", "multiclass"} or prefix.startswith("qHeteroRegression"):
            return _ACQF_ALIASES.get(_normalize_acqf_name(f"{prefix}ExpectedImprovement"))
        return _fallback_builtin_path("qei")
    if normalized_name in {"pi", "qpi", "probabilityofimprovement", "qprobabilityofimprovement"}:
        if task in {"binary", "ordinal", "multiclass"} or prefix.startswith("qHeteroRegression"):
            return _ACQF_ALIASES.get(_normalize_acqf_name(f"{prefix}ProbabilityOfImprovement"))
        return _fallback_builtin_path("qpi")
    if normalized_name in {"ucb", "qucb", "upperconfidencebound", "qupperconfidencebound"}:
        if task in {"binary", "ordinal", "multiclass"} or prefix.startswith("qHeteroRegression"):
            return _ACQF_ALIASES.get(_normalize_acqf_name(f"{prefix}UpperConfidenceBound"))
        return _fallback_builtin_path("qucb")
    if normalized_name in {"pof", "probabilityoffeasibility"}:
        if task in {"binary", "ordinal", "multiclass"}:
            return _ACQF_ALIASES.get(_normalize_acqf_name(f"{prefix}ProbabilityOfFeasibility"))
        return None
    if normalized_name in {"ehi", "qehi", "ehvi", "qehvi", "expectedhypervolumeimprovement", "qexpectedhypervolumeimprovement"}:
        if task in {"binary", "ordinal"} or prefix.startswith("qHeteroMultiOutputRegression"):
            if not multi_output:
                return None
            return _ACQF_ALIASES.get(_normalize_acqf_name(f"{prefix}ExpectedHypervolumeImprovement"))
        if task == "multiclass":
            return None
        return _fallback_builtin_path("qehvi")
    if normalized_name in {"nehvi", "qnehvi", "noisyexpectedhypervolumeimprovement", "qnoisyexpectedhypervolumeimprovement"}:
        if task in {"binary", "ordinal"} or prefix.startswith("qHeteroMultiOutputRegression"):
            if not multi_output:
                return None
            return _ACQF_ALIASES.get(_normalize_acqf_name(f"{prefix}NoisyExpectedHypervolumeImprovement"))
        if task == "multiclass":
            return None
        return _fallback_builtin_path("qnehvi")
    if normalized_name in {"nparego", "qnparego"}:
        if task in {"binary", "ordinal"} or prefix.startswith("qHeteroMultiOutputRegression"):
            if not multi_output:
                return None
            return _ACQF_ALIASES.get(_normalize_acqf_name(f"{prefix}NParEGO"))
        if task == "multiclass":
            return None
        return _fallback_builtin_path("qnparego")
    return None


def _resolve_contextual_acqf_path(normalized_name: str, *, task_type: str | None = None, model_type: str | None = None, multi_output: bool = False) -> AcqPath | None:
    if normalized_name not in _CONTEXTUAL_SHORT_NAMES:
        return None
    task = _normalize_task_type(task_type)
    if task is None:
        raise ValueError(f"Acquisition name {normalized_name!r} is task-dependent. Use a canonical name or call through BayesianOptimizer after fit().")
    hetero = _is_hetero(model_type)
    prefix = _family_prefix(task, multi_output=multi_output, hetero=hetero)
    bo_path = _resolve_contextual_bo_path(normalized_name, task=task, prefix=prefix, multi_output=multi_output)
    if bo_path is not None:
        return bo_path
    if normalized_name == "bald":
        suffix = "BALD" if task != "nongaussian" else "BALDProxy"
    elif normalized_name in {"predictiveentropy", "entropy"}:
        suffix = "PredictiveEntropy" if task != "nongaussian" else "PredictiveEntropyProxy"
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
            raise ValueError("Margin uncertainty is currently task-dependent for binary / ordinal / multiclass models only.")
        suffix = "MarginUncertainty"
    elif normalized_name == "straddle":
        suffix = "LatentStraddleAcquisition" if task in {"binary", "ordinal", "multiclass"} else "Straddle"
    elif normalized_name == "jointstraddle":
        suffix = "JointLatentStraddleAcquisition" if task in {"binary", "ordinal", "multiclass"} else "JointStraddle"
    elif normalized_name == "icu":
        suffix = "ICUAcquisition" if task in {"binary", "ordinal", "multiclass"} else "ICU"
    elif normalized_name == "boundaryvariance":
        suffix = "BoundaryVarianceAcquisition" if task in {"binary", "ordinal", "multiclass"} else "BoundaryVariance"
    elif normalized_name == "classentropy":
        if task not in {"binary", "ordinal", "multiclass"}:
            raise ValueError("Class entropy is only available for binary / ordinal / multiclass acquisitions.")
        suffix = "ClassEntropyAcquisition"
    elif normalized_name in {"probabilityofexceedance", "poe"}:
        suffix = "ProbabilityOfExceedance"
    elif normalized_name in {"levelsetuncertainty", "levelset"}:
        suffix = "LevelSetUncertainty"
    else:
        return None
    canonical_name = f"{prefix}{suffix}"
    path = _ACQF_ALIASES.get(_normalize_acqf_name(canonical_name))
    if path is None:
        raise ValueError(f"Short acquisition alias {normalized_name!r} resolved to {canonical_name!r}, but that acquisition is not registered.")
    return path


def _import_from_path(module_name: str, attr_name: str) -> Any:
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def resolve_acqf_cls(name: str, acquisition_registry: Mapping[str, Any] | None = None, *, task_type: str | None = None, model_type: str | None = None, multi_output: bool = False) -> type | Callable[..., Any]:
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
    contextual_path = _resolve_contextual_acqf_path(normalized, task_type=task_type, model_type=model_type, multi_output=multi_output)
    if contextual_path is not None:
        return _import_from_path(*contextual_path)
    if normalized not in _ACQF_ALIASES:
        available = sorted(_ACQF_ALIASES)
        raise ValueError(f"Unknown acquisition function name: {name!r}. Available built-in aliases include: {available}. For custom acquisitions, pass acquisition_registry.")
    return _import_from_path(*_ACQF_ALIASES[normalized])


def available_acqf_names() -> list[str]:
    return sorted(_ACQF_ALIASES | {name: ("", "") for name in _CONTEXTUAL_SHORT_NAMES})


__all__ = ["available_acqf_names", "resolve_acqf_cls"]
