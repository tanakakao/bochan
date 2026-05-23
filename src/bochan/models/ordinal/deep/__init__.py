from .deepgp import OrdinalDeepGPModel, OrdinalMixedDeepGPModel
from .deepkernel import DeepKernelOrdinalGPModel, DeepKernelOrdinalMixedGPModel
from .deepkerneldeepgp import DeepKernelOrdinalDeepGPModel, DeepKernelOrdinalMixedDeepGPModel

__all__ = [
    "OrdinalDeepGPModel", "OrdinalMixedDeepGPModel",
    "DeepKernelOrdinalGPModel", "DeepKernelOrdinalMixedGPModel",
    "DeepKernelOrdinalDeepGPModel", "DeepKernelOrdinalMixedDeepGPModel"
]