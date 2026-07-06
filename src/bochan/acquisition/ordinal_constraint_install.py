"""Install objective-space constraints on ordinal acquisition classes."""

import inspect
from functools import wraps

import torch

_APPLIED = False


def _own_init_signature(acquisition_cls: type) -> inspect.Signature | None:
    """Return the class's own initializer signature without inherited metadata."""

    initializer = acquisition_cls.__dict__.get("__init__")
    if initializer is None:
        return None
    initializer = inspect.unwrap(initializer)
    try:
        signature = inspect.signature(initializer, follow_wrapped=False)
    except (TypeError, ValueError):
        return None
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]
    return signature.replace(parameters=parameters)


def _patch_eta_buffer_assignment(acquisition_cls: type) -> None:
    """Pass eta with the shape required by BoTorch's registered buffer."""

    if acquisition_cls.__dict__.get("_bochan_eta_buffer_patched", False):
        return

    from bochan.acquisition.classification_constraint_compat import (
        _install_init_patch,
    )

    original_init = acquisition_cls.__init__
    public_signature = _own_init_signature(acquisition_cls)

    @wraps(original_init)
    def compatible_init(self, *args, eta=1e-3, **kwargs) -> None:
        constraints = kwargs.get("constraints")
        num_constraints = 0 if constraints is None else len(constraints)
        if torch.is_tensor(eta):
            eta_tensor = eta
            if eta_tensor.ndim == 0 and num_constraints:
                eta_tensor = eta_tensor.expand(num_constraints).clone()
        elif num_constraints:
            eta_tensor = torch.full((num_constraints,), float(eta))
        else:
            eta_tensor = torch.as_tensor(float(eta))
        original_init(self, *args, eta=eta_tensor, **kwargs)

    _install_init_patch(acquisition_cls, compatible_init, public_signature)
    acquisition_cls._bochan_eta_buffer_patched = True
    acquisition_cls._bochan_original_init_before_eta_buffer = original_init


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
