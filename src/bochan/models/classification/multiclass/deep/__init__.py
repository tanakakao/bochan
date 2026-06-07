from .deepgp import MulticlassDeepGPModel, MulticlassMixedDeepGPModel
from .deepkernel import (
    DeepKernelMulticlassClassificationGPModel,
    DeepKernelMulticlassClassificationMixedGPModel,
)

__all__ = [
    "MulticlassDeepGPModel",
    "MulticlassMixedDeepGPModel",
    "DeepKernelMulticlassClassificationGPModel",
    "DeepKernelMulticlassClassificationMixedGPModel",
]
