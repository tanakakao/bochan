"""Model and acquisition registries for the high-level bochan API."""

from .acquisition import available_acqf_names, resolve_acqf_cls
from .model import DEFAULT_MODEL_REGISTRY, MODEL_REGISTRY, LazyModelRegistry


def _register_multifidelity_gp() -> None:
    """Register the long-format Gaussian multi-fidelity model type.

    The historical ``model_type='multifidelity'`` entries remain untouched and
    continue to resolve to wide-format models. ``multifidelity_gp`` is the new
    long-format fidelity-feature contract introduced in phases 43-47.
    """

    tree = MODEL_REGISTRY.raw()
    adapter = (
        "bochan.models.multifidelity.configured",
        "create_configured_fidelity_surrogate",
    )
    tree["normal"]["regression"]["multifidelity_gp"] = adapter
    tree["mixed"]["regression"]["multifidelity_gp"] = adapter


_register_multifidelity_gp()


__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "LazyModelRegistry",
    "MODEL_REGISTRY",
    "available_acqf_names",
    "resolve_acqf_cls",
]
