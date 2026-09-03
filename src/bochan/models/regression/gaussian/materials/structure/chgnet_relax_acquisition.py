"""CHGNet wrapper around the backend-neutral relaxation selector."""

from __future__ import annotations

from typing import Any

from .chgnet_relaxation import CHGNetStructureRelaxer
from .relax_acquisition import (
    BundleFactory,
    MaterialRelaxationAcquisitionSelector,
    RelaxedStructureAcquisitionCandidate,
    RelaxedStructureAcquisitionResult,
)


class CHGNetRelaxationAcquisitionSelector(MaterialRelaxationAcquisitionSelector):
    """Relax with CHGNet and select through the common BO/AL relaxation layer."""

    def __init__(self, *, relaxer: CHGNetStructureRelaxer | None = None, **relaxer_kwargs: Any) -> None:
        if relaxer is not None and relaxer_kwargs:
            raise ValueError("Pass either relaxer or relaxer keyword arguments, not both.")
        resolved = CHGNetStructureRelaxer(**relaxer_kwargs) if relaxer is None else relaxer
        super().__init__(relaxer=resolved)


__all__ = [
    "BundleFactory",
    "CHGNetRelaxationAcquisitionSelector",
    "RelaxedStructureAcquisitionCandidate",
    "RelaxedStructureAcquisitionResult",
]
