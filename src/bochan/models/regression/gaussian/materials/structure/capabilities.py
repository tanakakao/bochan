"""Capability introspection for material MLIP backends and workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .factory import (
    SUPPORTED_MLIP_BACKENDS,
    MaterialMLIPBackend,
    normalize_material_backend,
)
from .model_factory import SUPPORTED_MATERIAL_MODEL_MODES, MaterialModelMode
from .property_factory import SUPPORTED_MATERIAL_QUANTITIES, MaterialQuantity
from .workflow_factory import SUPPORTED_MATERIAL_WORKFLOW_MODES, MaterialWorkflowMode


@dataclass(frozen=True, slots=True)
class MaterialBackendCapabilities:
    """Serializable capabilities and input constraints for one MLIP backend."""

    backend: MaterialMLIPBackend
    direct_quantities: tuple[MaterialQuantity, ...]
    residual_quantities: tuple[MaterialQuantity, ...]
    workflow_modes: tuple[MaterialWorkflowMode, ...]
    supports_relaxation: bool
    supports_relax_rank: bool
    supports_relax_acquisition: bool
    force_fixed_topology: bool
    stress_components: int
    residual_requires_structure_graphs: bool = False

    @property
    def model_modes(self) -> tuple[MaterialModelMode, ...]:
        """Return model modes exposed by this backend."""

        modes: list[MaterialModelMode] = []
        if self.direct_quantities:
            modes.append("direct")
        if self.residual_quantities:
            modes.append("residual_gp")
        return tuple(modes)

    def supports(
        self,
        *,
        quantity: str | None = None,
        model_mode: str | None = None,
        workflow_mode: str | None = None,
    ) -> bool:
        """Return whether a normalized combination is supported by this backend."""

        from .model_factory import normalize_material_model_mode
        from .property_factory import normalize_material_quantity
        from .workflow_factory import normalize_material_workflow_mode

        resolved_quantity = (
            None if quantity is None else normalize_material_quantity(quantity)
        )
        resolved_model_mode = (
            None if model_mode is None else normalize_material_model_mode(model_mode)
        )
        resolved_workflow_mode = (
            None
            if workflow_mode is None
            else normalize_material_workflow_mode(workflow_mode)
        )

        if resolved_model_mode == "direct":
            if resolved_quantity is not None and resolved_quantity not in self.direct_quantities:
                return False
        elif resolved_model_mode == "residual_gp":
            if resolved_quantity is not None and resolved_quantity not in self.residual_quantities:
                return False
        elif (
            resolved_quantity is not None
            and resolved_quantity not in self.direct_quantities
            and resolved_quantity not in self.residual_quantities
        ):
            return False

        return not (
            resolved_workflow_mode is not None
            and resolved_workflow_mode not in self.workflow_modes
        )

    def requirements(
        self,
        *,
        quantity: str,
        model_mode: str,
    ) -> tuple[str, ...]:
        """Return extra input requirements for one model configuration."""

        from .model_factory import normalize_material_model_mode
        from .property_factory import normalize_material_quantity

        resolved_quantity = normalize_material_quantity(quantity)
        resolved_model_mode = normalize_material_model_mode(model_mode)
        if not self.supports(
            quantity=resolved_quantity,
            model_mode=resolved_model_mode,
        ):
            raise ValueError(
                f"Unsupported material configuration for {self.backend}: "
                f"{resolved_quantity}/{resolved_model_mode}."
            )

        requirements: list[str] = ["structures"]
        if resolved_model_mode == "residual_gp":
            requirements.extend(("train_X", "train_Y"))
            if self.residual_requires_structure_graphs:
                requirements.append("structure_graphs")
        if resolved_quantity == "force" and self.force_fixed_topology:
            requirements.append("fixed_atom_count")
        return tuple(requirements)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable capability metadata for UI/API discovery."""

        return {
            "backend": self.backend,
            "direct_quantities": list(self.direct_quantities),
            "residual_quantities": list(self.residual_quantities),
            "model_modes": list(self.model_modes),
            "workflow_modes": list(self.workflow_modes),
            "supports_relaxation": self.supports_relaxation,
            "supports_relax_rank": self.supports_relax_rank,
            "supports_relax_acquisition": self.supports_relax_acquisition,
            "force_fixed_topology": self.force_fixed_topology,
            "stress_components": self.stress_components,
            "residual_requires_structure_graphs": self.residual_requires_structure_graphs,
        }


_COMMON = dict(
    direct_quantities=SUPPORTED_MATERIAL_QUANTITIES,
    residual_quantities=SUPPORTED_MATERIAL_QUANTITIES,
    workflow_modes=SUPPORTED_MATERIAL_WORKFLOW_MODES,
    supports_relaxation=True,
    supports_relax_rank=True,
    supports_relax_acquisition=True,
    force_fixed_topology=True,
    stress_components=9,
)

_CAPABILITIES: dict[MaterialMLIPBackend, MaterialBackendCapabilities] = {
    "mace": MaterialBackendCapabilities(backend="mace", **_COMMON),
    "chgnet": MaterialBackendCapabilities(backend="chgnet", **_COMMON),
    "m3gnet": MaterialBackendCapabilities(backend="m3gnet", **_COMMON),
    "alignn-ff": MaterialBackendCapabilities(
        backend="alignn-ff",
        residual_requires_structure_graphs=True,
        **_COMMON,
    ),
}


def get_material_backend_capabilities(backend: str) -> MaterialBackendCapabilities:
    """Return immutable capability metadata for one normalized backend."""

    return _CAPABILITIES[normalize_material_backend(backend)]


def list_material_backend_capabilities() -> tuple[MaterialBackendCapabilities, ...]:
    """Return capabilities for all public backends in canonical order."""

    return tuple(_CAPABILITIES[backend] for backend in SUPPORTED_MLIP_BACKENDS)


def get_material_capability_catalog() -> dict[str, Any]:
    """Return a JSON-ready discovery catalog for UI/API option generation."""

    return {
        "backends": [
            capability.as_dict() for capability in list_material_backend_capabilities()
        ],
        "quantities": list(SUPPORTED_MATERIAL_QUANTITIES),
        "model_modes": list(SUPPORTED_MATERIAL_MODEL_MODES),
        "workflow_modes": list(SUPPORTED_MATERIAL_WORKFLOW_MODES),
    }


__all__ = [
    "MaterialBackendCapabilities",
    "get_material_backend_capabilities",
    "get_material_capability_catalog",
    "list_material_backend_capabilities",
]
