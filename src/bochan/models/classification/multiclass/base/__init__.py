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
from .posteriors import apply_multiclass_posteriors

apply_multiclass_posteriors()

__all__ = [
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
    "apply_multiclass_posteriors",
]
