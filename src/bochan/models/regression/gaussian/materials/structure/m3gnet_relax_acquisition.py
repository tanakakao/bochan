"""M3GNet wrapper around the backend-neutral relaxation selector."""

from __future__ import annotations

from typing import Any

from .m3gnet_relaxation import M3GNetStructureRelaxer
from .relax_acquisition import (
    BundleFactory,
    MaterialRelaxationAcquisitionSelector,
    RelaxedStructureAcquisitionCandidate,
    RelaxedStructureAcquisitionResult,
)


class M3GNetRelaxationAcquisitionSelector(MaterialRelaxationAcquisitionSelector):
    """Relax with M3GNet and select through the common BO/AL relaxation layer."""

    def __init__(self, *, relaxer: M3GNetStructureRelaxer | None = None, **relaxer_kwargs: Any) -> None:
        if relaxer is not None and relaxer_kwargs:
            raise ValueError("Pass either relaxer or relaxer keyword arguments, not both.")
        resolved = M3GNetStructureRelaxer(**relaxer_kwargs) if relaxer is None else relaxer
        super().__init__(relaxer=resolved)


__all__ = [
    "BundleFactory",
    "M3GNetRelaxationAcquisitionSelector",
    "RelaxedStructureAcquisitionCandidate",
    "RelaxedStructureAcquisitionResult",
]
