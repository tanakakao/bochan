from __future__ import annotations

from typing import Optional, Sequence

from .rrp_binary import (
    fit_rrp_binary_classifier_mll as _fit_rrp_binary_classifier_mll,
    fit_rrp_binary_classifier_mll_optimizer,
)
from .rrp_multiclass import (
    fit_rrp_multiclass_classifier_mll_optimizer,
    fit_rrp_multiclass_mll as _fit_rrp_multiclass_mll,
    fit_rrp_multiclass_mll_optimizer,
)
from .rrp_ordinal import (
    fit_rrp_ordinal_mll as _fit_rrp_ordinal_mll,
    fit_rrp_ordinal_mll_optimizer,
)


DEFAULT_RRP_SPARSITY_LEVELS = (0, 1, 2, 3)


def fit_rrp_binary_classifier_mll(
    mll,
    *,
    method: str = "backward",
    sparsity_levels: Optional[Sequence[int]] = DEFAULT_RRP_SPARSITY_LEVELS,
    initial_support: Optional[list[int]] = None,
    reset_parameters: bool = True,
    reset_dense_parameters: bool = False,
    record_model_trace: Optional[bool] = None,
    return_all: bool = False,
    optimizer=fit_rrp_binary_classifier_mll_optimizer,
    optimizer_kwargs: Optional[dict] = None,
    closure=None,
    closure_kwargs: Optional[dict] = None,
):
    """Fit binary RRP with a lightweight default sparsity search."""

    return _fit_rrp_binary_classifier_mll(
        mll,
        method=method,
        sparsity_levels=sparsity_levels,
        initial_support=initial_support,
        reset_parameters=reset_parameters,
        reset_dense_parameters=reset_dense_parameters,
        record_model_trace=record_model_trace,
        return_all=return_all,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        closure=closure,
        closure_kwargs=closure_kwargs,
    )


def fit_rrp_ordinal_mll(
    mll,
    *,
    fit_model=None,
    method: str = "backward",
    sparsity_levels: Optional[Sequence[int]] = DEFAULT_RRP_SPARSITY_LEVELS,
    initial_support: Optional[list[int]] = None,
    reset_parameters: bool = True,
    reset_dense_parameters: bool = False,
    record_model_trace: Optional[bool] = None,
    return_all: bool = False,
    optimizer=fit_rrp_ordinal_mll_optimizer,
    optimizer_kwargs: Optional[dict] = None,
    closure=None,
    closure_kwargs: Optional[dict] = None,
):
    """Fit ordinal RRP with a lightweight default sparsity search."""

    return _fit_rrp_ordinal_mll(
        mll,
        fit_model=fit_model,
        method=method,
        sparsity_levels=sparsity_levels,
        initial_support=initial_support,
        reset_parameters=reset_parameters,
        reset_dense_parameters=reset_dense_parameters,
        record_model_trace=record_model_trace,
        return_all=return_all,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        closure=closure,
        closure_kwargs=closure_kwargs,
    )


def fit_rrp_multiclass_mll(
    mll,
    *,
    fit_model=None,
    method: str = "backward",
    sparsity_levels: Optional[Sequence[int]] = DEFAULT_RRP_SPARSITY_LEVELS,
    initial_support: Optional[list[int]] = None,
    reset_parameters: bool = True,
    reset_dense_parameters: bool = False,
    record_model_trace: Optional[bool] = None,
    return_all: bool = False,
    optimizer=fit_rrp_multiclass_mll_optimizer,
    optimizer_kwargs: Optional[dict] = None,
    closure=None,
    closure_kwargs: Optional[dict] = None,
):
    """Fit multiclass RRP with a lightweight default sparsity search."""

    return _fit_rrp_multiclass_mll(
        mll,
        fit_model=fit_model,
        method=method,
        sparsity_levels=sparsity_levels,
        initial_support=initial_support,
        reset_parameters=reset_parameters,
        reset_dense_parameters=reset_dense_parameters,
        record_model_trace=record_model_trace,
        return_all=return_all,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        closure=closure,
        closure_kwargs=closure_kwargs,
    )


fit_rrp_multiclass_classifier_mll = fit_rrp_multiclass_mll


__all__ = [
    "DEFAULT_RRP_SPARSITY_LEVELS",
    "fit_rrp_binary_classifier_mll",
    "fit_rrp_binary_classifier_mll_optimizer",
    "fit_rrp_multiclass_mll",
    "fit_rrp_multiclass_mll_optimizer",
    "fit_rrp_multiclass_classifier_mll",
    "fit_rrp_multiclass_classifier_mll_optimizer",
    "fit_rrp_ordinal_mll",
    "fit_rrp_ordinal_mll_optimizer",
]
