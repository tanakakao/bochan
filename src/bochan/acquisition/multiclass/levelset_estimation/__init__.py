from .sample_compat import apply_levelset_q_like_compat
from .hetero_multi_output import (
    qHeteroMultiOutputMulticlassBoundaryVarianceAcquisition,
    qHeteroMultiOutputMulticlassClassEntropyAcquisition,
    qHeteroMultiOutputMulticlassICUAcquisition,
    qHeteroMultiOutputMulticlassJointLatentStraddleAcquisition,
    qHeteroMultiOutputMulticlassLatentStraddleAcquisition,
    qHeteroMultiOutputMulticlassLevelSetUncertainty,
    qHeteroMultiOutputMulticlassProbabilityOfExceedance,
)
from .hetero_single_output import (
    NoiseCombineType,
    NoiseWeightMode,
    qHeteroMulticlassBoundaryVarianceAcquisition,
    qHeteroMulticlassClassEntropyAcquisition,
    qHeteroMulticlassICUAcquisition,
    qHeteroMulticlassJointLatentStraddleAcquisition,
    qHeteroMulticlassLatentStraddleAcquisition,
    qHeteroMulticlassLevelSetUncertainty,
    qHeteroMulticlassProbabilityOfExceedance,
)
from .multi_output import (
    OutputReductionType,
    qMultiOutputMulticlassBoundaryVarianceAcquisition,
    qMultiOutputMulticlassClassEntropyAcquisition,
    qMultiOutputMulticlassICUAcquisition,
    qMultiOutputMulticlassJointLatentStraddleAcquisition,
    qMultiOutputMulticlassLatentStraddleAcquisition,
    qMultiOutputMulticlassLevelSetUncertainty,
    qMultiOutputMulticlassProbabilityOfExceedance,
)
from .single_output import (
    qMulticlassBoundaryVarianceAcquisition,
    qMulticlassClassEntropyAcquisition,
    qMulticlassICUAcquisition,
    qMulticlassJointLatentStraddleAcquisition,
    qMulticlassLatentStraddleAcquisition,
    qMulticlassLevelSetUncertainty,
    qMulticlassProbabilityOfExceedance,
)

# Heteroscedastic / wrapper multiclass models may return q_like=1 for
# pending+candidate q-batches. Apply a joint level-set q_like alignment patch.
apply_levelset_q_like_compat()

__all__ = [
    "NoiseCombineType",
    "NoiseWeightMode",
    "OutputReductionType",
    "apply_levelset_q_like_compat",
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
]
