from .hidden_layers import (
    DeepGPHiddenLayer, DeepKernelDeepGPHiddenLayer,
    DeepKernelDeepMixedGPHiddenLayer, DeepMixedGPHiddenLayer,
    SkipDeepGPHiddenLayer, SkipDeepMixedGPHiddenLayer,
    SkipDeepKernelDeepGPHiddenLayer, SkipDeepKernelDeepMixedGPHiddenLayer
)
from .kernel_layers import (
    DeepKernel,
    DeepKernelMixed,
    _PartialObservationMultitaskKernel,
)

# Material FastAPI metadata historically exposes the semantic GPyTorch kernel
# name. Keep that public contract stable while the Phase 6 subclass supplies
# masked exact prediction internally.
_PartialObservationMultitaskKernel.__name__ = "MultitaskKernel"

__all__ = [
    "DeepGPHiddenLayer", "DeepKernelDeepGPHiddenLayer",
    "DeepKernelDeepMixedGPHiddenLayer", "DeepMixedGPHiddenLayer",
    "DeepKernel", "DeepKernelMixed",
    "SkipDeepGPHiddenLayer", "SkipDeepMixedGPHiddenLayer",
    "SkipDeepKernelDeepGPHiddenLayer", "SkipDeepKernelDeepMixedGPHiddenLayer"
]