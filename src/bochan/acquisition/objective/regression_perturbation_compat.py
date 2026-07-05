"""Compatibility for nested regression input-perturbation objectives."""

from __future__ import annotations

from typing import Any

from torch import Tensor


_APPLIED = False


def apply_regression_perturbation_objective_compat() -> None:
    """Let the outer objective aggregate ``q * n_w`` before q validation.

    BoTorch's ``MCMultiOutputObjective.__call__`` verifies that an objective's q
    axis matches ``X.shape[-2]``. For an input-perturbation wrapper, the inner
    objective intentionally returns ``q * n_w`` and the outer objective reduces
    that axis back to q. Temporarily disabling only the inner verification keeps
    the outer, final shape validation intact.
    """

    global _APPLIED
    if _APPLIED:
        return

    from .regression import MultiOutputRegressionInputPerturbationObjective

    acquisition_cls = MultiOutputRegressionInputPerturbationObjective
    if getattr(acquisition_cls, "_bochan_inner_q_validation_patched", False):
        _APPLIED = True
        return

    original_forward = acquisition_cls.forward

    def compatible_forward(
        self,
        samples: Tensor,
        X: Tensor | None = None,
    ) -> Tensor:
        inner_objective: Any = self.inner_objective
        had_verify_flag = hasattr(inner_objective, "_verify_output_shape")
        original_verify = getattr(inner_objective, "_verify_output_shape", None)
        if had_verify_flag:
            inner_objective._verify_output_shape = False
        try:
            return original_forward(self, samples=samples, X=X)
        finally:
            if had_verify_flag:
                inner_objective._verify_output_shape = original_verify

    acquisition_cls.forward = compatible_forward
    acquisition_cls._bochan_inner_q_validation_patched = True
    _APPLIED = True


__all__ = ["apply_regression_perturbation_objective_compat"]
