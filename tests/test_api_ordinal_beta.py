from __future__ import annotations

from types import SimpleNamespace

import bochan.fit as fit_api


def test_make_ordinal_mll_applies_beta(monkeypatch) -> None:
    mll = SimpleNamespace(beta=1.0)

    def fake_make_ordinal_mll(model, **kwargs):
        assert model == "model"
        assert kwargs == {"num_data": 10}
        return mll

    monkeypatch.setattr(fit_api, "_make_ordinal_mll", fake_make_ordinal_mll)

    result = fit_api.make_ordinal_mll(
        "model",
        beta=0.01,
        num_data=10,
    )

    assert result is mll
    assert result.beta == 0.01


def test_make_ordinal_mll_without_beta_preserves_default(monkeypatch) -> None:
    mll = SimpleNamespace(beta=1.0)
    monkeypatch.setattr(
        fit_api,
        "_make_ordinal_mll",
        lambda model, **kwargs: mll,
    )

    result = fit_api.make_ordinal_mll("model")

    assert result.beta == 1.0


def test_make_ordinal_mll_rejects_beta_for_unsupported_mll(monkeypatch) -> None:
    monkeypatch.setattr(
        fit_api,
        "_make_ordinal_mll",
        lambda model, **kwargs: SimpleNamespace(),
    )

    try:
        fit_api.make_ordinal_mll("model", beta=0.01)
    except TypeError as exc:
        assert "does not support the beta parameter" in str(exc)
    else:
        raise AssertionError("Expected TypeError for an MLL without beta support.")
