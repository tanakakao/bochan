"""Central model capability metadata used by API and serving clients."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapability:
    """Describe the supported workflows and distribution contract of a model."""

    model_type: str
    task_types: tuple[str, ...]
    input_types: tuple[str, ...]
    output_domain: str
    likelihood_family: str
    supports_single_output: bool = True
    supports_independent_multi_output: bool = True
    supports_native_multi_output: bool = False
    supports_hybrid_output: bool = True
    supports_feature_importance: bool = True
    supports_noise_importance: bool = False


BETA_MODEL_TYPES = (
    "beta_base",
    "beta_deepgp",
    "beta_deepkernel",
    "beta_saas",
    "beta_pca",
    "beta_rembo",
    "beta_rrp",
    "beta_hetero",
)

MODEL_CAPABILITIES = {
    model_type: ModelCapability(
        model_type=model_type,
        task_types=("regression",),
        input_types=("normal", "mixed"),
        output_domain="unit_interval",
        likelihood_family="beta",
        supports_noise_importance=model_type == "beta_hetero",
    )
    for model_type in BETA_MODEL_TYPES
}


def model_capability(model_type: str) -> ModelCapability | None:
    """Return capability metadata for ``model_type`` when it is catalogued.

    Args:
        model_type: Public model registry key.

    Returns:
        Immutable model capability metadata, or ``None`` for an uncatalogued key.
    """
    return MODEL_CAPABILITIES.get(str(model_type))


def model_likelihood_family(model_type: str) -> str:
    """Resolve a model's likelihood family from the central catalog.

    Args:
        model_type: Public model registry key.

    Returns:
        Catalogued likelihood family, falling back to ``"gaussian"``.
    """
    capability = model_capability(model_type)
    return capability.likelihood_family if capability is not None else "gaussian"


__all__ = [
    "BETA_MODEL_TYPES",
    "MODEL_CAPABILITIES",
    "ModelCapability",
    "model_capability",
    "model_likelihood_family",
]
