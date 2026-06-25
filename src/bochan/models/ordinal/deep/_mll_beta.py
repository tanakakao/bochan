from __future__ import annotations

from functools import wraps
from typing import TypeVar


T = TypeVar("T", bound=type)


def enable_make_mll_beta(model_cls: T) -> T:
    """Allow a model's ``make_mll`` method to accept an optional ``beta``.

    The wrapped implementation first creates the model-specific variational MLL,
    then updates its KL-divergence weight when ``beta`` is explicitly supplied.
    This preserves the existing PredictiveLogLikelihood / VariationalELBO choice.
    """

    if getattr(model_cls, "_make_mll_beta_enabled", False):
        return model_cls

    original_make_mll = model_cls.make_mll

    @wraps(original_make_mll)
    def wrapped_make_mll(self, beta: float | None = None):
        mll = original_make_mll(self)
        if beta is None:
            return mll
        if not hasattr(mll, "beta"):
            raise TypeError(
                f"{type(mll).__name__} does not support the beta parameter."
            )
        mll.beta = float(beta)
        return mll

    model_cls.make_mll = wrapped_make_mll
    model_cls._make_mll_beta_enabled = True
    return model_cls


__all__ = ["enable_make_mll_beta"]
