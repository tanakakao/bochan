"""Compatibility metadata for historical material-model import paths.

Material model implementation classes still live under ``gaussian.deep`` so
older pickle payloads can resolve their original ``__module__`` paths.  The
canonical ``gaussian.materials`` namespaces re-export those exact class objects.

This module records that boundary explicitly without importing concrete model
modules eagerly.  It is intentionally metadata-only: no deprecation warnings
are emitted while historical module paths remain part of the serialization
contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialCompatibilityPath:
    """Map one historical import path to its canonical material namespace path."""

    legacy: str
    canonical: str
    serialization_protected: bool = True

    def __post_init__(self) -> None:
        for name, value in (("legacy", self.legacy), ("canonical", self.canonical)):
            if not isinstance(value, str) or ":" not in value:
                raise ValueError(f"{name} must use 'module:attribute' syntax.")
        if self.legacy == self.canonical:
            raise ValueError("legacy and canonical paths must differ.")


LEGACY_MATERIAL_MODEL_PATHS: tuple[MaterialCompatibilityPath, ...] = (
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.crabnet:CrabNetGPModel",
        "bochan.models.regression.gaussian.materials.composition.crabnet:CrabNetGPModel",
    ),
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.crabnet:CrabNetDKLModel",
        "bochan.models.regression.gaussian.materials.composition.crabnet:CrabNetDKLModel",
    ),
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.roost:RoostGPModel",
        "bochan.models.regression.gaussian.materials.composition.roost:RoostGPModel",
    ),
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.roost:RoostDKLModel",
        "bochan.models.regression.gaussian.materials.composition.roost:RoostDKLModel",
    ),
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.alignn:ALIGNNGPModel",
        "bochan.models.regression.gaussian.materials.structure.alignn:ALIGNNGPModel",
    ),
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.alignn:ALIGNNDKLModel",
        "bochan.models.regression.gaussian.materials.structure.alignn:ALIGNNDKLModel",
    ),
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.chgnet:CHGNetGPModel",
        "bochan.models.regression.gaussian.materials.structure.chgnet:CHGNetGPModel",
    ),
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.chgnet:CHGNetDKLModel",
        "bochan.models.regression.gaussian.materials.structure.chgnet:CHGNetDKLModel",
    ),
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.m3gnet:M3GNetGPModel",
        "bochan.models.regression.gaussian.materials.structure.m3gnet:M3GNetGPModel",
    ),
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.m3gnet:M3GNetDKLModel",
        "bochan.models.regression.gaussian.materials.structure.m3gnet:M3GNetDKLModel",
    ),
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.mace:MACEGPModel",
        "bochan.models.regression.gaussian.materials.structure.mace:MACEGPModel",
    ),
    MaterialCompatibilityPath(
        "bochan.models.regression.gaussian.deep.mace:MACEDKLModel",
        "bochan.models.regression.gaussian.materials.structure.mace:MACEDKLModel",
    ),
)


def legacy_material_model_paths() -> tuple[str, ...]:
    """Return historical model paths protected for import/serialization compatibility."""

    return tuple(item.legacy for item in LEGACY_MATERIAL_MODEL_PATHS)


def canonical_material_model_paths() -> tuple[str, ...]:
    """Return canonical model paths corresponding to protected historical paths."""

    return tuple(item.canonical for item in LEGACY_MATERIAL_MODEL_PATHS)


__all__ = [
    "LEGACY_MATERIAL_MODEL_PATHS",
    "MaterialCompatibilityPath",
    "canonical_material_model_paths",
    "legacy_material_model_paths",
]
