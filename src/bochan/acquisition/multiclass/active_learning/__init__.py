from .alignment import apply_active_learning_alignment
from .hetero_alignment import apply_hetero_noise_alignment
from .hetero_multi_output import (
    qHeteroMultiOutputMulticlassBALD,
    qHeteroMultiOutputMulticlassGreedyJointBALD,
    qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy,
    qHeteroMultiOutputMulticlassJointBALD,
    qHeteroMultiOutputMulticlassMarginUncertainty,
    qHeteroMultiOutputMulticlassPredictiveEntropy,
    qHeteroMultiOutputMulticlassProbabilityVariance,
)
from .hetero_single_output import (
    NoiseCombineType,
    NoiseQAggregateType,
    NoiseWeightMode,
    qHeteroMulticlassBALD,
    qHeteroMulticlassGreedyJointBALD,
    qHeteroMulticlassIntegratedPosteriorVarianceProxy,
    qHeteroMulticlassJointBALD,
    qHeteroMulticlassMarginUncertainty,
    qHeteroMulticlassPredictiveEntropy,
    qHeteroMulticlassProbabilityVariance,
)
from .multi_output import (
    OutputReductionType,
    qMultiOutputMulticlassBALD,
    qMultiOutputMulticlassGreedyJointBALD,
    qMultiOutputMulticlassIntegratedPosteriorVarianceProxy,
    qMultiOutputMulticlassJointBALD,
    qMultiOutputMulticlassMarginUncertainty,
    qMultiOutputMulticlassPredictiveEntropy,
    qMultiOutputMulticlassProbabilityVariance,
)
from .single_output import (
    qMulticlassBALD,
    qMulticlassGreedyJointBALD,
    qMulticlassIntegratedPosteriorVarianceProxy,
    qMulticlassJointBALD,
    qMulticlassMarginUncertainty,
    qMulticlassPredictiveEntropy,
    qMulticlassProbabilityVariance,
)

# DeepGP などで posterior sample / latent 軸が片側だけに残る場合の align 互換 patch。
apply_active_learning_alignment()
# Hetero multi-output の score/noise weight 軸ずれを補正する patch。
apply_hetero_noise_alignment()

__all__ = [
    "NoiseCombineType",
    "NoiseQAggregateType",
    "NoiseWeightMode",
    "OutputReductionType",
    "apply_active_learning_alignment",
    "apply_hetero_noise_alignment",
    "qMulticlassPredictiveEntropy",
    "qMulticlassProbabilityVariance",
    "qMulticlassMarginUncertainty",
    "qMulticlassBALD",
    "qMulticlassJointBALD",
    "qMulticlassGreedyJointBALD",
    "qMulticlassIntegratedPosteriorVarianceProxy",
    "qMultiOutputMulticlassPredictiveEntropy",
    "qMultiOutputMulticlassProbabilityVariance",
    "qMultiOutputMulticlassMarginUncertainty",
    "qMultiOutputMulticlassBALD",
    "qMultiOutputMulticlassJointBALD",
    "qMultiOutputMulticlassGreedyJointBALD",
    "qMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
    "qHeteroMulticlassPredictiveEntropy",
    "qHeteroMulticlassProbabilityVariance",
    "qHeteroMulticlassMarginUncertainty",
    "qHeteroMulticlassBALD",
    "qHeteroMulticlassJointBALD",
    "qHeteroMulticlassGreedyJointBALD",
    "qHeteroMulticlassIntegratedPosteriorVarianceProxy",
    "qHeteroMultiOutputMulticlassPredictiveEntropy",
    "qHeteroMultiOutputMulticlassProbabilityVariance",
    "qHeteroMultiOutputMulticlassMarginUncertainty",
    "qHeteroMultiOutputMulticlassBALD",
    "qHeteroMultiOutputMulticlassJointBALD",
    "qHeteroMultiOutputMulticlassGreedyJointBALD",
    "qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
]
