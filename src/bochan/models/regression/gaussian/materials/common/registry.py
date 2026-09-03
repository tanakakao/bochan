"""Lazy registry for material-aware Gaussian model families."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Literal

from .pretrained import MaterialDomain, PretrainedMaterialCapabilities, PretrainedMaterialSpec

MaterialModelVariant = Literal[
    "gp", "dkl", "mixed_gp", "mixed_dkl", "multitask_gp", "multitask_dkl",
    "mixed_multitask_gp", "mixed_multitask_dkl", "residual_gp", "mixed_residual_gp",
]


@dataclass(frozen=True)
class MaterialFamilyRegistration:
    family: str
    domain: MaterialDomain
    model_paths: dict[MaterialModelVariant, str]
    pretrained: PretrainedMaterialSpec

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("family must be a non-empty string.")
        if self.domain not in {"composition", "structure"}:
            raise ValueError("domain must be 'composition' or 'structure'.")
        if self.pretrained.family != self.family or self.pretrained.domain != self.domain:
            raise ValueError("pretrained metadata must match registration family/domain.")
        if not self.model_paths:
            raise ValueError("model_paths must contain at least one model variant.")
        for variant, path in self.model_paths.items():
            if not isinstance(path, str) or ":" not in path:
                raise ValueError(f"Model path for {variant!r} must use 'module:attribute' syntax.")

    @property
    def variants(self) -> frozenset[MaterialModelVariant]:
        return frozenset(self.model_paths)

    def supports(self, variant: MaterialModelVariant) -> bool:
        return variant in self.model_paths

    def model_path(self, variant: MaterialModelVariant) -> str:
        try:
            return self.model_paths[variant]
        except KeyError as error:
            raise ValueError(
                f"Material family {self.family!r} does not support variant {variant!r}."
            ) from error

    def resolve_model_class(self, variant: MaterialModelVariant) -> type:
        module_name, attribute = self.model_path(variant).split(":", 1)
        resolved = getattr(import_module(module_name), attribute, None)
        if not isinstance(resolved, type):
            raise RuntimeError("Registered model path did not resolve to a class.")
        return resolved


def _capabilities(*, loading_modes: frozenset[str], fine_tuning: bool = True,
                  direct_prediction: bool = False, residual_gp: bool = False) -> PretrainedMaterialCapabilities:
    return PretrainedMaterialCapabilities(
        representation=True, direct_prediction=direct_prediction,
        loading_modes=loading_modes,  # type: ignore[arg-type]
        device_aware=True, dtype_aware=True, fine_tuning=fine_tuning, residual_gp=residual_gp,
    )


def _models(namespace: str, family: str, *, full_matrix: bool,
            residual_gp: bool = False, mixed_residual_gp: bool = False) -> dict[MaterialModelVariant, str]:
    prefixes = {"mace": "MACE", "m3gnet": "M3GNet", "chgnet": "CHGNet", "alignn": "ALIGNN", "crabnet": "CrabNet", "roost": "Roost"}
    prefix = prefixes[family]
    paths: dict[MaterialModelVariant, str] = {"gp": f"{namespace}:{prefix}GPModel", "dkl": f"{namespace}:{prefix}DKLModel"}
    if full_matrix:
        paths.update({
            "mixed_gp": f"{namespace}:{prefix}MixedGPModel",
            "mixed_dkl": f"{namespace}:{prefix}MixedDKLModel",
            "multitask_gp": f"{namespace}:{prefix}MultiTaskGPModel",
            "multitask_dkl": f"{namespace}:{prefix}MultiTaskDKLModel",
            "mixed_multitask_gp": f"{namespace}:{prefix}MixedMultiTaskGPModel",
            "mixed_multitask_dkl": f"{namespace}:{prefix}MixedMultiTaskDKLModel",
        })
    if residual_gp:
        paths["residual_gp"] = f"{namespace}:{prefix}ResidualGPModel"
    if mixed_residual_gp:
        paths["mixed_residual_gp"] = f"{namespace}:{prefix}MixedResidualGPModel"
    return paths


_structure_namespace = "bochan.models.regression.gaussian.materials.structure"
_composition_namespace = "bochan.models.regression.gaussian.materials.composition"

MATERIAL_FAMILY_REGISTRY: dict[str, MaterialFamilyRegistration] = {
    "crabnet": MaterialFamilyRegistration("crabnet", "composition", _models(_composition_namespace, "crabnet", full_matrix=True), PretrainedMaterialSpec("crabnet", "composition", _capabilities(loading_modes=frozenset({"checkpoint", "injected"})))),
    "roost": MaterialFamilyRegistration("roost", "composition", _models(_composition_namespace, "roost", full_matrix=False), PretrainedMaterialSpec("roost", "composition", _capabilities(loading_modes=frozenset({"checkpoint", "injected"})))),
    "alignn": MaterialFamilyRegistration("alignn", "structure", _models(_structure_namespace, "alignn", full_matrix=True), PretrainedMaterialSpec("alignn", "structure", _capabilities(loading_modes=frozenset({"checkpoint", "injected"})))),
    "chgnet": MaterialFamilyRegistration("chgnet", "structure", _models(_structure_namespace, "chgnet", full_matrix=True, residual_gp=True, mixed_residual_gp=True), PretrainedMaterialSpec("chgnet", "structure", _capabilities(loading_modes=frozenset({"checkpoint", "model_name", "injected"}), direct_prediction=True, residual_gp=True), default_model_name="0.3.0")),
    "m3gnet": MaterialFamilyRegistration("m3gnet", "structure", _models(_structure_namespace, "m3gnet", full_matrix=True, residual_gp=True, mixed_residual_gp=True), PretrainedMaterialSpec("m3gnet", "structure", _capabilities(loading_modes=frozenset({"model_name", "injected"}), direct_prediction=True, residual_gp=True), default_model_name="M3GNet-PES-MatPES-PBE-2025.2")),
    "mace": MaterialFamilyRegistration("mace", "structure", _models(_structure_namespace, "mace", full_matrix=True, residual_gp=True, mixed_residual_gp=True), PretrainedMaterialSpec("mace", "structure", _capabilities(loading_modes=frozenset({"model_name", "injected"}), direct_prediction=True, residual_gp=True), default_model_name="medium-mpa-0")),
}


def get_material_family(family: str) -> MaterialFamilyRegistration:
    if not isinstance(family, str) or not family.strip():
        raise ValueError("family must be a non-empty string.")
    key = family.strip().lower()
    try:
        return MATERIAL_FAMILY_REGISTRY[key]
    except KeyError as error:
        raise KeyError(f"Unknown material family {family!r}; choose one of {sorted(MATERIAL_FAMILY_REGISTRY)!r}.") from error


def list_material_families(*, domain: MaterialDomain | None = None) -> tuple[str, ...]:
    if domain is not None and domain not in {"composition", "structure"}:
        raise ValueError("domain must be 'composition', 'structure', or None.")
    return tuple(family for family, registration in MATERIAL_FAMILY_REGISTRY.items() if domain is None or registration.domain == domain)


__all__ = ["MATERIAL_FAMILY_REGISTRY", "MaterialFamilyRegistration", "MaterialModelVariant", "get_material_family", "list_material_families"]
