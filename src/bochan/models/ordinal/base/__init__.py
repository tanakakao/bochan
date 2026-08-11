from .kronecker_multitask import KroneckerMultiTaskOrdinalGPModel
from .kronecker_multitask_mixed import KroneckerMultiTaskOrdinalMixedGPModel
from .models import OrdinalGPModel, OrdinalMixedGPModel
from bochan.models.multioutput.ordinal import MultiOutputOrdinalModel
from .multitask import MultiTaskOrdinalGPModel
from .multitask_mixed import MultiTaskOrdinalMixedGPModel

__all__ = [
    "KroneckerMultiTaskOrdinalGPModel",
    "KroneckerMultiTaskOrdinalMixedGPModel",
    "OrdinalGPModel",
    "OrdinalMixedGPModel",
    "MultiOutputOrdinalModel",
    "MultiTaskOrdinalGPModel",
    "MultiTaskOrdinalMixedGPModel",
]
