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

# BoTorch qEHVI / qNEHVI compatibility for multiclass probability posteriors.
apply_multiclass_posterior_compat()

__all__ = [
    "MulticlassClassificationGPModel",
    "MulticlassClassificationMixedGPModel",
    "build_mixed_multiclass_kernel",
    "MultiOutputMulticlassProbsPosterior",
    "MultiOutputMulticlassClassificationModel",
    "MultiOutputMulticlassModel",
    "MultiOutputMulticlassClassificationGPModel",
    "apply_multiclass_posterior_compat",
]
