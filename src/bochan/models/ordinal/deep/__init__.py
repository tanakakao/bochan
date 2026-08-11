from .deepgp import DeepOrdinalGPModel, DeepOrdinalMixedGPModel
from .deepkernel_configurable import DeepKernelOrdinalGPModel, DeepKernelOrdinalMixedGPModel
from .deepkerneldeepgp import DeepKernelDeepOrdinalGPModel, DeepKernelDeepOrdinalMixedGPModel

__all__ = [
    "DeepOrdinalGPModel",
    "DeepOrdinalMixedGPModel",
    "DeepKernelOrdinalGPModel",
    "DeepKernelOrdinalMixedGPModel",
    "DeepKernelDeepOrdinalGPModel",
    "DeepKernelDeepOrdinalMixedGPModel",
]
