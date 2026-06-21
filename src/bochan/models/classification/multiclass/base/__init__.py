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
from .posterior_compat import apply_multiclass_posterior_compat

apply_multiclass_posterior_compat()

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
    "apply_multiclass_posterior_compat",
]
