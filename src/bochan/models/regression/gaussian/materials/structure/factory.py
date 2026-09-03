"""Backend-neutral factories for MLIP relaxation workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Literal, cast

from ..common.relaxation import MaterialStructureRelaxer
from .relax_acquisition import MaterialRelaxationAcquisitionSelector
from .relax_rank import MaterialRelaxationRanker

MaterialMLIPBackend = Literal["mace", "chgnet", "m3gnet", "alignn-ff"]
SUPPORTED_MLIP_BACKENDS: tuple[MaterialMLIPBackend, ...] = (
    "mace",
    "chgnet",
    "m3gnet",
    "alignn-ff",
)

_BACKEND_ALIASES = {
    "mace": "mace",
    "chgnet": "chgnet",
    "m3gnet": "m3gnet",
    "alignn-ff": "alignn-ff",
    "alignn_ff": "alignn-ff",
    "alignnff": "alignn-ff",
}

_RELAXER_IMPORTS: dict[MaterialMLIPBackend, tuple[str, str]] = {
    "mace": (".mace_relaxation", "MACEStructureRelaxer"),
    "chgnet": (".chgnet_relaxation", "CHGNetStructureRelaxer"),
    "m3gnet": (".m3gnet_relaxation", "M3GNetStructureRelaxer"),
    "alignn-ff": (".alignn_ff_relaxation", "ALIGNNFFStructureRelaxer"),
}


def normalize_material_backend(backend: str) -> MaterialMLIPBackend:
    """Normalize one supported MLIP backend identifier.

    ``alignn_ff`` and ``alignnff`` are accepted as spelling aliases for the
    canonical public identifier ``alignn-ff``. No semantic aliases such as
    ``matgl`` are accepted because one library may expose multiple potentials.
    """

    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("backend must be a non-empty string.")
    normalized = backend.strip().lower()
    resolved = _BACKEND_ALIASES.get(normalized)
    if resolved is None:
        supported = ", ".join(SUPPORTED_MLIP_BACKENDS)
        raise ValueError(f"Unsupported material backend {backend!r}. Supported backends: {supported}.")
    return cast(MaterialMLIPBackend, resolved)


def _resolve_relaxer_class(backend: MaterialMLIPBackend) -> type[Any]:
    module_name, class_name = _RELAXER_IMPORTS[backend]
    module = import_module(module_name, package=__package__)
    relaxer_class = getattr(module, class_name, None)
    if not isinstance(relaxer_class, type):
        raise RuntimeError(
            f"Material backend {backend!r} does not expose expected relaxer class {class_name}."
        )
    return relaxer_class


def create_structure_relaxer(
    backend: str,
    /,
    **backend_kwargs: Any,
) -> MaterialStructureRelaxer:
    """Construct a backend-specific structure relaxer through one public API.

    Backend-specific constructor arguments are intentionally forwarded without
    reinterpretation. This preserves capabilities such as MACE ``device``,
    CHGNet checkpoints/encoders, M3GNet potential injection, and ALIGNN-FF
    calculator injection while keeping backend selection uniform.
    """

    resolved_backend = normalize_material_backend(backend)
    relaxer_class = _resolve_relaxer_class(resolved_backend)
    return cast(MaterialStructureRelaxer, relaxer_class(**backend_kwargs))


def create_relaxation_ranker(
    backend: str,
    /,
    *,
    relaxer: MaterialStructureRelaxer | None = None,
    **backend_kwargs: Any,
) -> MaterialRelaxationRanker:
    """Construct a backend-neutral relax-and-rank workflow.

    Pass an already-created ``relaxer`` for dependency injection, or supply
    backend-specific relaxer keyword arguments. Mixing both is rejected to avoid
    silently ignored configuration.
    """

    if relaxer is not None and backend_kwargs:
        raise ValueError("Pass either relaxer or backend keyword arguments, not both.")
    resolved = create_structure_relaxer(backend, **backend_kwargs) if relaxer is None else relaxer
    return MaterialRelaxationRanker(relaxer=resolved)


def create_relaxation_acquisition_selector(
    backend: str,
    /,
    *,
    relaxer: MaterialStructureRelaxer | None = None,
    **backend_kwargs: Any,
) -> MaterialRelaxationAcquisitionSelector:
    """Construct a backend-neutral relax-then-select BO/AL workflow."""

    if relaxer is not None and backend_kwargs:
        raise ValueError("Pass either relaxer or backend keyword arguments, not both.")
    resolved = create_structure_relaxer(backend, **backend_kwargs) if relaxer is None else relaxer
    return MaterialRelaxationAcquisitionSelector(relaxer=resolved)


__all__ = [
    "MaterialMLIPBackend",
    "SUPPORTED_MLIP_BACKENDS",
    "create_relaxation_acquisition_selector",
    "create_relaxation_ranker",
    "create_structure_relaxer",
    "normalize_material_backend",
]
