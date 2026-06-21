from bochan.acquisition._nehvi_cache_root import patch_nehvi_cache_root_init

from . import multi_output as _multi_output
from .hetero_multi_output import (
    qHeteroMultiOutputBinaryExpectedHypervolumeImprovement,
    qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
    qHeteroMultiOutputBinaryNParEGO,
)

from .hetero_single_output import (
    qHeteroBinaryUpperConfidenceBound,
    qHeteroBinaryExpectedImprovement,
    qHeteroBinaryProbabilityOfImprovement,
)

# Apply the same model-aware qNEHVI default used by ordinal models. This keeps
# Kronecker binary models out of BoTorch's incompatible cached-Cholesky path.
patch_nehvi_cache_root_init(
    _multi_output.qMultiOutputBinaryNoisyExpectedHypervolumeImprovement
)

from .multi_output import (
    qMultiOutputBinaryProbabilityOfFeasibility,
    qMultiOutputBinaryExpectedHypervolumeImprovement,
    qMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
    qMultiOutputBinaryNParEGO,
)

from .single_output import (
    QBatchMode,
    qBinaryProbabilityOfFeasibility,
    qBinaryExpectedImprovement,
    qBinaryProbabilityOfImprovement,
    qBinaryUpperConfidenceBound,
)
from ._utils import (
    compute_binary_best_f,
    compute_hetero_binary_classification_best_f,
)

__all__ = [
    "QBatchMode",
    "qHeteroMultiOutputBinaryExpectedHypervolumeImprovement",
    "qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement",
    "qHeteroMultiOutputBinaryNParEGO",
    "qHeteroBinaryUpperConfidenceBound",
    "qHeteroBinaryExpectedImprovement",
    "qHeteroBinaryProbabilityOfImprovement",
    "qMultiOutputBinaryProbabilityOfFeasibility",
    "qMultiOutputBinaryExpectedHypervolumeImprovement",
    "qMultiOutputBinaryNoisyExpectedHypervolumeImprovement",
    "qMultiOutputBinaryNParEGO",
    "qBinaryProbabilityOfFeasibility",
    "qBinaryExpectedImprovement",
    "qBinaryProbabilityOfImprovement",
    "qBinaryUpperConfidenceBound",
    "compute_binary_best_f",
    "compute_hetero_binary_classification_best_f",
]
