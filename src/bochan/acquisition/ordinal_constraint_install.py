"""Install objective-space constraints on ordinal acquisition classes."""

from __future__ import annotations


_APPLIED = False


def apply_ordinal_constraint_compat() -> None:
    """Patch the internal ordinal EHVI classes behind public factory functions."""

    global _APPLIED
    if _APPLIED:
        return

    from bochan.acquisition.classification_constraint_compat import (
        _patch_hypervolume_constraints,
    )
    from bochan.acquisition.ordinal.bayesian_optimization import multi_output

    _patch_hypervolume_constraints(
        multi_output.qMultiOutputOrdinalExpectedHypervolumeImprovement
    )
    _patch_hypervolume_constraints(
        multi_output.qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement
    )
    _APPLIED = True


__all__ = ["apply_ordinal_constraint_compat"]
