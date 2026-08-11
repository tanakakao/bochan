from . import posterior
from .kronecker_multitask import (
    BlockDesignMulticlassLikelihood,
    KroneckerMultiTaskMulticlassClassificationGPModel,
    KroneckerMultiTaskMulticlassProbsPosterior,
)
from .kronecker_multitask_mixed import (
    KroneckerMultiTaskMulticlassClassificationMixedGPModel,
)
from .models import (
    MulticlassClassificationGPModel,
    MulticlassClassificationMixedGPModel,
    build_mixed_multiclass_kernel,
)
from .multioutput import (
    MultiOutputMulticlassClassificationGPModel,
    MultiOutputMulticlassClassificationModel,
    MultiOutputMulticlassModel,
    MultiOutputMulticlassProbsPosterior,
)
from .multitask import (
    MultiTaskMulticlassClassificationGPModel,
    MultiTaskMulticlassClassificationMixedGPModel,
)

__all__ = [
    "posterior",
    "BlockDesignMulticlassLikelihood",
    "KroneckerMultiTaskMulticlassClassificationGPModel",
    "KroneckerMultiTaskMulticlassClassificationMixedGPModel",
    "KroneckerMultiTaskMulticlassProbsPosterior",
    "MulticlassClassificationGPModel",
    "MulticlassClassificationMixedGPModel",
    "build_mixed_multiclass_kernel",
    "MultiOutputMulticlassProbsPosterior",
    "MultiOutputMulticlassClassificationModel",
    "MultiOutputMulticlassModel",
    "MultiOutputMulticlassClassificationGPModel",
    "MultiTaskMulticlassClassificationGPModel",
    "MultiTaskMulticlassClassificationMixedGPModel",
]
