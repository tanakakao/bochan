from .multi_output import HybridMultiOutputModel
from .posterior import HybridPosterior
from .prediction import attach_prediction_methods
from .specs import OutputSpec, PosteriorMode, TaskType

attach_prediction_methods(HybridMultiOutputModel)

__all__ = [
    "HybridMultiOutputModel",
    "HybridPosterior",
    "OutputSpec",
    "PosteriorMode",
    "TaskType",
]
