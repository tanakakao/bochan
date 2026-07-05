"""Install objective-space constraints on ordinal acquisition classes."""

import torch

_APPLIED = False


def _patch_eta_buffer_assignment(acquisition_cls: type) -> None:
    """Pass eta as a Tensor when an ordinal wrapper reassigns the buffer."""

    if getattr(acquisition_cls, "_bochan_eta_buffer_patched", False):
        return

    original_init = acquisition_cls.__init__

    def compatible_init(self, *args, eta=1e-3, **kwargs) -> None:
        eta_tensor = eta if torch.is_tensor(eta) else torch.as_tensor(float(eta))
        original_init(self, *args, eta=eta_tensor, **kwargs)

    acquisition_cls.__init__ = compatible_init
    acquisition_cls._bochan_eta_buffer_patched = True


def apply_ordinal_constraint_compat() -> None:
    """Patch internal ordinal EHVI classes behind public factory functions."""

    global _APPLIED
    if _APPLIED:
        return

    from bochan.acquisition.classification_constraint_compat import (
        _patch_hypervolume_constraints,
    )
    from bochan.acquisition.ordinal.bayesian_optimization import multi_output

    for acquisition_cls in (
        multi_output.qMultiOutputOrdinalExpectedHypervolumeImprovement,
        multi_output.qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
    ):
        _patch_eta_buffer_assignment(acquisition_cls)
        _patch_hypervolume_constraints(acquisition_cls)

    _APPLIED = True


__all__ = ["apply_ordinal_constraint_compat"]
