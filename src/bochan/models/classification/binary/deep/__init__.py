from .deepgp import DeepBinaryClassificationGPModel, DeepBinaryClassificationMixedGPModel
from .deepkernel_configurable import DeepKernelBinaryClassificationGPModel, DeepKernelBinaryClassificationMixedGPModel
from .deepkerneldeepgp import DeepKernelDeepBinaryClassificationGPModel, DeepKernelDeepBinaryClassificationMixedGPModel

__all__ = [
    "DeepBinaryClassificationGPModel",
    "DeepBinaryClassificationMixedGPModel",
    "DeepKernelBinaryClassificationGPModel",
    "DeepKernelBinaryClassificationMixedGPModel",
    "DeepKernelDeepBinaryClassificationGPModel",
    "DeepKernelDeepBinaryClassificationMixedGPModel"
]
