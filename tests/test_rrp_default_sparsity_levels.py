from __future__ import annotations

import inspect

import pytest

import bochan.fit.robust as robust_fit


RRP_FIT_FUNCTIONS = [
    robust_fit.fit_rrp_binary_classifier_mll,
    robust_fit.fit_rrp_ordinal_mll,
    robust_fit.fit_rrp_multiclass_mll,
]


@pytest.mark.parametrize("fit_func", RRP_FIT_FUNCTIONS)
def test_rrp_fit_uses_lightweight_default_sparsity_levels(fit_func) -> None:
    parameter = inspect.signature(fit_func).parameters["sparsity_levels"]

    assert parameter.default == (0, 1, 2, 3)


@pytest.mark.parametrize("fit_func", RRP_FIT_FUNCTIONS)
def test_rrp_sparsity_levels_remain_keyword_overridable(fit_func) -> None:
    parameter = inspect.signature(fit_func).parameters["sparsity_levels"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_binary_rrp_can_restore_automatic_sparsity_search(monkeypatch) -> None:
    received = {}

    def fake_fit(mll, **kwargs):
        received.update(kwargs)
        return mll

    monkeypatch.setattr(robust_fit, "_fit_rrp_binary_classifier_mll", fake_fit)
    sentinel = object()

    result = robust_fit.fit_rrp_binary_classifier_mll(
        sentinel,
        sparsity_levels=None,
    )

    assert result is sentinel
    assert received["sparsity_levels"] is None


def test_multiclass_classifier_alias_uses_same_default() -> None:
    parameter = inspect.signature(
        robust_fit.fit_rrp_multiclass_classifier_mll
    ).parameters["sparsity_levels"]

    assert parameter.default == robust_fit.DEFAULT_RRP_SPARSITY_LEVELS
