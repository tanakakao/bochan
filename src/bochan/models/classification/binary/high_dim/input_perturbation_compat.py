"""InputPerturbation shape compatibility for projected binary classifiers."""

from __future__ import annotations

from bochan.models.projected_input_perturbation_compat import (
    patch_projected_model_classes,
)


_PATCHED = False


def apply_projected_binary_perturbation_compat() -> None:
    """Patch non-mixed PCA and REMBO binary classifiers once."""

    global _PATCHED
    if _PATCHED:
        return

    from .decomposition import (
        PCABinaryClassificationGPModel,
        REMBOBinaryClassificationGPModel,
    )

    patch_projected_model_classes(
        [
            PCABinaryClassificationGPModel,
            REMBOBinaryClassificationGPModel,
        ]
    )
    _PATCHED = True


__all__ = [
    "apply_projected_binary_perturbation_compat",
]
