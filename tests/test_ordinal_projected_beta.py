from __future__ import annotations

from types import SimpleNamespace

from bochan.models.ordinal.high_dim.decomposition import _BaseProjectedOrdinalGP


def test_projected_ordinal_make_mll_forwards_beta_to_base_model() -> None:
    calls = {}

    class BaseModel:
        def make_mll(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(**kwargs)

    model = object.__new__(_BaseProjectedOrdinalGP)
    model.base_model = BaseModel()

    mll = model.make_mll(beta=0.01)

    assert calls == {"beta": 0.01}
    assert mll.beta == 0.01


def test_projected_ordinal_make_mll_without_kwargs_delegates_cleanly() -> None:
    calls = []

    class BaseModel:
        def make_mll(self, **kwargs):
            calls.append(kwargs)
            return object()

    model = object.__new__(_BaseProjectedOrdinalGP)
    model.base_model = BaseModel()

    result = model.make_mll()

    assert calls == [{}]
    assert result is not None
