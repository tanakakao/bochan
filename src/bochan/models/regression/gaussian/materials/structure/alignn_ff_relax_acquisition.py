"""ALIGNN-FF wrapper around the backend-neutral relaxation selector."""

from __future__ import annotations

from typing import Any

from .alignn_ff_relaxation import ALIGNNFFStructureRelaxer
from .relax_acquisition import (
    BundleFactory,
    MaterialRelaxationAcquisitionSelector,
    RelaxedStructureAcquisitionCandidate,
    RelaxedStructureAcquisitionResult,
)


class ALIGNNFFRelaxationAcquisitionSelector(MaterialRelaxationAcquisitionSelector):
    """Relax with ALIGNN-FF and select through the common BO/AL layer."""

    def __init__(self, *, relaxer: ALIGNNFFStructureRelaxer | None = None, **relaxer_kwargs: Any) -> None:
        if relaxer is not None and relaxer_kwargs:
            raise ValueError("Pass either relaxer or relaxer keyword arguments, not both.")
        resolved = ALIGNNFFStructureRelaxer(**relaxer_kwargs) if relaxer is None else relaxer
        super().__init__(relaxer=resolved)


__all__ = [
    "ALIGNNFFRelaxationAcquisitionSelector",
    "BundleFactory",
    "RelaxedStructureAcquisitionCandidate",
    "RelaxedStructureAcquisitionResult",
]
