"""Lazy registry for material-aware Gaussian model families.

The registry stores stable metadata and import paths only. Importing this module
must not import optional material backends such as MACE, CHGNet, ALIGNN, or
M3GNet. Concrete model classes are resolved only when explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Literal

from .pretrained import (
    MaterialDomain,
    PretrainedMaterialCapabilities,
    PretrainedMaterialSpec,
)

MaterialModelVariant = Literal[
    "gp",
    "dkl",
    "mixed_gp",
    "mixed_dkl",
    "multitask_gp",
    "multitask_dkl",
    "mixed_multitask_gp",
    "mixed_multitask_dkl",
    "residual_gp",
]


@dataclass(frozen=True)
class MaterialFamilyRegistration:
    """Describe one material-model family without importing its backend."""

    family: str
    domain: MaterialDomain
    model_paths: dict[MaterialModelVariant, str]
    pretrained: PretrainedMaterialSpec

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("family must be a non-empty string.")
        if self.domain not in {"composition", "structure"}:
            raise ValueError("domain must be 'composition' or 'structure'.")
        if self.pretrained.family != self.family:
            raise ValueError("pretrained.family must match registration.family.")
        if self.pretrained.domain != self.domain:
            raise ValueError("pretrained.domain must match registration.domain.")
        if not self.model_paths:
            raise ValueError("model_paths must contain at least one model variant.")
        for variant, path in self.model_paths.items():
            if not isinstance(path, str) or ":" not in path:
                raise ValueError(
                    f"Model path for {variant!r} must use 'module:attribute' syntax."
                )

    @property
    def variants(self) -> frozenset[MaterialModelVariant]:
        """Return all model variants currently implemented by Bochan."""

        return frozenset(self.model_paths)

    def supports(self, variant: MaterialModelVariant) -> bool:
        """Return whether the family exposes one Bochan model variant."""

        return variant in self.model_paths

    def model_path(self, variant: MaterialModelVariant) -> str:
        """Return the lazy import path for one model variant."""

        try:
            return self.model_paths[variant]
        except KeyError as error:
            raise ValueError(
                f"Material family {self.family!r} does not support variant {variant!r}."
            ) from error

    def resolve_model_class(self, variant: MaterialModelVariant) -> type:
        """Import and return one concrete model class on demand."""

        path = self.model_path(variant)
        module_name, attribute = path.split(":", 1)
        module = import_module(module_name)
        resolved = getattr(module, attribute, None)
        if not isinstance(resolved, type):
            raise RuntimeError(f"Registered model path {path!r} did not resolve to a class.")
        return resolved


def _capabilities(
    *,
    loading_modes: frozenset[str],
    fine_tuning: bool = True,
    direct_prediction: bool = False,
    residual_gp: bool = False,
) -> PretrainedMaterialCapabilities:
    """Build capability metadata for one currently supported material adapter."""

    return PretrainedMaterialCapabilities(
        representation=True,
        direct_prediction=direct_prediction,
        loading_modes=loading_modes,  # type: ignore[arg-type]
        device_aware=True,
        dtype_aware=True,
        fine_tuning=fine_tuning,
        residual_gp=residual_gp,
    )


def _models(
    namespace: str,
    family: str,
    *,
    full_matrix: bool,
    residual_gp: bool = False,
) -> dict[MaterialModelVariant, str]:
    prefix = family.upper() if family != "mace" else "MACE"
    if family == "m3gnet":
        prefix = "M3GNet"
    elif family == "chgnet":
        prefix = "CHGNet"
    elif family == "alignn":
        prefix = "ALIGNN"
    elif family == "crabnet":
        prefix = "CrabNet"
    elif family == "roost":
        prefix = "Roost"

    paths: dict[MaterialModelVariant, str] = {
        "gp": f"{namespace}:{prefix}GPModel",
        "dkl": f"{namespace}:{prefix}DKLModel",
    }
    if full_matrix:
        paths.update(
            {
                "mixed_gp": f"{namespace}:{prefix}MixedGPModel",
                "mixed_dkl": f"{namespace}:{prefix}MixedDKLModel",
                "multitask_gp": f"{namespace}:{prefix}MultiTaskGPModel",
                "multitask_dkl": f"{namespace}:{prefix}MultiTaskDKLModel",
                "mixed_multitask_gp": f"{namespace}:{prefix}MixedMultiTaskGPModel",
                "mixed_multitask_dkl": f"{namespace}:{prefix}MixedMultiTaskDKLModel",
            }
        )
    if residual_gp:
        paths["residual_gp"] = f"{namespace}:{prefix}ResidualGPModel"
    return paths


MATERIAL_FAMILY_REGISTRY: dict[str, MaterialFamilyRegistration] = {
    "crabnet": MaterialFamilyRegistration(
        family="crabnet",
        domain="composition",
        model_paths=_models(
            "bochan.models.regression.gaussian.materials.composition",
            "crabnet",
            full_matrix=True,
        ),
        pretrained=PretrainedMaterialSpec(
            family="crabnet",
            domain="composition",
            capabilities=_capabilities(
                loading_modes=frozenset({"checkpoint", "injected"})
            ),
        ),
    ),
    "roost": MaterialFamilyRegistration(
        family="roost",
        domain="composition",
        model_paths=_models(
            "bochan.models.regression.gaussian.materials.composition",
            "roost",
            full_matrix=False,
        ),
        pretrained=PretrainedMaterialSpec(
            family="roost",
            domain="composition",
            capabilities=_capabilities(
                loading_modes=frozenset({"checkpoint", "injected"})
            ),
        ),
    ),
    "alignn": MaterialFamilyRegistration(
        family="alignn",
        domain="structure",
        model_paths=_models(
            "bochan.models.regression.gaussian.materials.structure",
            "alignn",
            full_matrix=True,
        ),
        pretrained=PretrainedMaterialSpec(
            family="alignn",
            domain="structure",
            capabilities=_capabilities(
                loading_modes=frozenset({"checkpoint", "injected"})
            ),
        ),
    ),
    "chgnet": MaterialFamilyRegistration(
        family="chgnet",
        domain="structure",
        model_paths=_models(
            "bochan.models.regression.gaussian.materials.structure",
            "chgnet",
            full_matrix=True,
            residual_gp=True,
        ),
        pretrained=PretrainedMaterialSpec(
            family="chgnet",
            domain="structure",
            capabilities=_capabilities(
                loading_modes=frozenset({"checkpoint", "model_name", "injected"}),
                direct_prediction=True,
                residual_gp=True,
            ),
            default_model_name="0.3.0",
        ),
    ),
    "m3gnet": MaterialFamilyRegistration(
        family="m3gnet",
        domain="structure",
        model_paths=_models(
            "bochan.models.regression.gaussian.materials.structure",
            "m3gnet",
            full_matrix=True,
        ),
        pretrained=PretrainedMaterialSpec(
            family="m3gnet",
            domain="structure",
            capabilities=_capabilities(
                loading_modes=frozenset({"model_name", "injected"})
            ),
            default_model_name="M3GNet-PES-MatPES-PBE-2025.2",
        ),
    ),
    "mace": MaterialFamilyRegistration(
        family="mace",
        domain="structure",
        model_paths=_models(
            "bochan.models.regression.gaussian.materials.structure",
            "mace",
            full_matrix=True,
        ),
        pretrained=PretrainedMaterialSpec(
            family="mace",
            domain="structure",
            capabilities=_capabilities(
                loading_modes=frozenset({"model_name", "injected"})
            ),
            default_model_name="medium-mpa-0",
        ),
    ),
}


def get_material_family(family: str) -> MaterialFamilyRegistration:
    """Return one registered family by stable lowercase identifier."""

    if not isinstance(family, str) or not family.strip():
        raise ValueError("family must be a non-empty string.")
    key = family.strip().lower()
    try:
        return MATERIAL_FAMILY_REGISTRY[key]
    except KeyError as error:
        raise KeyError(
            f"Unknown material family {family!r}; choose one of {sorted(MATERIAL_FAMILY_REGISTRY)!r}."
        ) from error


def list_material_families(*, domain: MaterialDomain | None = None) -> tuple[str, ...]:
    """List registered family names, optionally restricted by domain."""

    if domain is not None and domain not in {"composition", "structure"}:
        raise ValueError("domain must be 'composition', 'structure', or None.")
    return tuple(
        family
        for family, registration in MATERIAL_FAMILY_REGISTRY.items()
        if domain is None or registration.domain == domain
    )


__all__ = [
    "MATERIAL_FAMILY_REGISTRY",
    "MaterialFamilyRegistration",
    "MaterialModelVariant",
    "get_material_family",
    "list_material_families",
]
