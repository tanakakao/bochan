"""InputPerturbation shape support for projected binary classifiers."""

from __future__ import annotations

from bochan.models.projected_input_perturbation import (
    configure_projected_model_classes,
)

_PATCHED = False


def configure_projected_binary_perturbation() -> None:
    """Patch non-mixed PCA and REMBO binary classifiers once."""

    global _PATCHED
    if _PATCHED:
        return

    from .decomposition import (
        PCABinaryClassificationGPModel,
        REMBOBinaryClassificationGPModel,
    )

    configure_projected_model_classes(
        [
            PCABinaryClassificationGPModel,
            REMBOBinaryClassificationGPModel,
        ]
    )
    _PATCHED = True


__all__ = [
    "configure_projected_binary_perturbation",
]
