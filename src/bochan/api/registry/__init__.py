"""Model and acquisition registries for the high-level bochan API."""

from .acquisition import available_acqf_names, resolve_acqf_cls
from .model import DEFAULT_MODEL_REGISTRY, MODEL_REGISTRY, LazyModelRegistry
from .multifidelity import register_multifidelity_gp_model_types

__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "LazyModelRegistry",
    "MODEL_REGISTRY",
    "available_acqf_names",
    "register_multifidelity_gp_model_types",
    "resolve_acqf_cls",
]
