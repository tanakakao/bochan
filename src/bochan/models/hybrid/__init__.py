from __future__ import annotations

from .class_probability_shapes import apply_hybrid_class_probability_shapes
from .multi_output import HybridMultiOutputModel
from .posterior import HybridPosterior
from .prediction import attach_prediction_methods
from .specs import OutputSpec, PosteriorMode, TaskType


def _hybrid_set_transformed_inputs(self: HybridMultiOutputModel) -> None:
    """Skip BoTorch wrapper-level transformed-input caching.

    HybridMultiOutputModel delegates input transforms to its submodels. The
    wrapper may still expose a shared ``input_transform`` for distance-space
    calculations, but it should not rewrite its synthetic train inputs during
    ``Model.eval()``. Submodels own their transformed-input caches.
    """

    return None


def _hybrid_eval(self: HybridMultiOutputModel) -> HybridMultiOutputModel:
    """Put the hybrid wrapper and every submodel into eval mode safely.

    This keeps a shared wrapper ``input_transform`` available when one exists,
    while avoiding BoTorch's base wrapper-level transformed-input preprocessing
    path. That path assumes the wrapper itself owns a concrete train-data
    transform, which is not true for heterogeneous hybrid wrappers.
    """

    self.training = False
    for model in self.models:
        if hasattr(model, "eval"):
            model.eval()
    return self


# Install the methods on the class so all import paths see the same behavior,
# including direct imports from ``bochan.models.hybrid.multi_output`` after this
# package has been initialized.
HybridMultiOutputModel._set_transformed_inputs = _hybrid_set_transformed_inputs
HybridMultiOutputModel.eval = _hybrid_eval
apply_hybrid_class_probability_shapes(HybridMultiOutputModel)

attach_prediction_methods(HybridMultiOutputModel)

__all__ = [
    "HybridMultiOutputModel",
    "HybridPosterior",
    "OutputSpec",
    "PosteriorMode",
    "TaskType",
]
