"""High-level factory for backend-neutral MLIP model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from torch import Tensor

from .factory import MaterialMLIPBackend, normalize_material_backend
from .property_factory import (
    MaterialQuantity,
    create_direct_material_predictor,
    create_material_residual_gp,
    normalize_material_quantity,
)

MaterialModelMode = Literal["direct", "residual_gp"]
SUPPORTED_MATERIAL_MODEL_MODES: tuple[MaterialModelMode, ...] = (
    "direct",
    "residual_gp",
)

_MODE_ALIASES = {
    "direct": "direct",
    "pretrained": "direct",
    "baseline": "direct",
    "residual_gp": "residual_gp",
    "residual-gp": "residual_gp",
    "residualgp": "residual_gp",
}


def normalize_material_model_mode(mode: str) -> MaterialModelMode:
    """Normalize one supported high-level material model mode."""

    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("mode must be a non-empty string.")
    normalized = mode.strip().lower()
    resolved = _MODE_ALIASES.get(normalized)
    if resolved is None:
        supported = ", ".join(SUPPORTED_MATERIAL_MODEL_MODES)
        raise ValueError(
            f"Unsupported material model mode {mode!r}. Supported modes: {supported}."
        )
    return cast(MaterialModelMode, resolved)


@dataclass(frozen=True, slots=True)
class MaterialModelSpec:
    """Serializable normalized identity for one material model configuration.

    Backend-specific constructor options are intentionally kept outside this
    identity object because their schemas differ materially across MLIP families.
    The spec is therefore safe to persist in UI/API state while runtime options
    remain explicit at construction time.
    """

    backend: MaterialMLIPBackend | str
    quantity: MaterialQuantity | str
    mode: MaterialModelMode | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", normalize_material_backend(self.backend))
        object.__setattr__(self, "quantity", normalize_material_quantity(self.quantity))
        object.__setattr__(self, "mode", normalize_material_model_mode(self.mode))

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable canonical representation."""

        return {
            "backend": self.backend,
            "quantity": self.quantity,
            "mode": self.mode,
        }


def create_material_model(
    backend: str,
    quantity: str,
    mode: str,
    /,
    *,
    structures: Any,
    train_X: Tensor | None = None,
    train_Y: Tensor | None = None,
    train_Yvar: Tensor | None = None,
    **backend_kwargs: Any,
) -> Any:
    """Construct a direct MLIP predictor or residual GP through one public API.

    ``direct`` mode requires only a structure bank and backend-specific runtime
    options. ``residual_gp`` additionally requires ``train_X`` and ``train_Y``;
    ``train_Yvar`` remains optional. Backend-specific arguments are delegated to
    the Phase 21 property factories unchanged.
    """

    spec = MaterialModelSpec(backend=backend, quantity=quantity, mode=mode)

    if spec.mode == "direct":
        supplied_training = [
            name
            for name, value in (
                ("train_X", train_X),
                ("train_Y", train_Y),
                ("train_Yvar", train_Yvar),
            )
            if value is not None
        ]
        if supplied_training:
            names = ", ".join(supplied_training)
            raise ValueError(
                f"direct material models do not accept training tensors: {names}."
            )
        return create_direct_material_predictor(
            spec.backend,
            spec.quantity,
            structures=structures,
            **backend_kwargs,
        )

    missing = [
        name
        for name, value in (("train_X", train_X), ("train_Y", train_Y))
        if value is None
    ]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"residual_gp material models require {names}.")

    return create_material_residual_gp(
        spec.backend,
        spec.quantity,
        cast(Tensor, train_X),
        cast(Tensor, train_Y),
        train_Yvar,
        structures=structures,
        **backend_kwargs,
    )


__all__ = [
    "MaterialModelMode",
    "MaterialModelSpec",
    "SUPPORTED_MATERIAL_MODEL_MODES",
    "create_material_model",
    "normalize_material_model_mode",
]
