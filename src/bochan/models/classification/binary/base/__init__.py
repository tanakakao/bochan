from __future__ import annotations

from .kronecker_multitask import KroneckerMultiTaskBinaryClassificationGPModel
from .kronecker_multitask_mixed import (
    KroneckerMultiTaskBinaryClassificationMixedGPModel,
)
from .models import (
    BinaryClassificationGPModel,
    BinaryClassificationMixedGPModel,
)
from .multifidelity import (
    WideMixedMultiFidelityBinaryClassificationGPModel,
    WideMultiFidelityBinaryClassificationGPModel,
    WideMultiFidelityBinaryClassificationMixedGPModel,
)
from bochan.models.multioutput.binary import MultiOutputBinaryClassificationModel
from .multitask import MultiTaskBinaryClassificationGPModel
from .multitask_mixed import MultiTaskBinaryClassificationMixedGPModel


__all__ = [
    "BinaryClassificationGPModel",
    "BinaryClassificationMixedGPModel",
    "KroneckerMultiTaskBinaryClassificationGPModel",
    "KroneckerMultiTaskBinaryClassificationMixedGPModel",
    "MultiOutputBinaryClassificationModel",
    "MultiTaskBinaryClassificationGPModel",
    "MultiTaskBinaryClassificationMixedGPModel",
    "WideMixedMultiFidelityBinaryClassificationGPModel",
    "WideMultiFidelityBinaryClassificationGPModel",
    "WideMultiFidelityBinaryClassificationMixedGPModel",
]
