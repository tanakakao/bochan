from __future__ import annotations

from types import SimpleNamespace

from bochan.api import FitConfig
from bochan.fit.classification import binary as binary_fit


def test_fit_config_beta_populates_mll_kwargs() -> None:
    config = FitConfig(beta=0.01)

    assert config.beta == 0.01
    assert config.mll_kwargs["beta"] == 0.01


def test_fit_config_explicit_mll_beta_takes_precedence() -> None:
    config = FitConfig(beta=0.01, mll_kwargs={"beta": 0.2})

    assert config.mll_kwargs["beta"] == 0.2


def test_binary_deep_model_uses_full_batch_fitter(monkeypatch) -> None:
    calls = {}

    def fake_fit_deep_full_batch_mll(mll, **kwargs):
        calls["mll"] = mll
        calls["kwargs"] = kwargs
        return "fitted"

    monkeypatch.setattr(
        "bochan.fit.deep.common.fit_deep_full_batch_mll",
        fake_fit_deep_full_batch_mll,
    )

    deep_model_type = type(
        "DeepBinaryClassificationGPModel",
        (),
        {"__module__": "bochan.models.classification.binary.deep.deepgp"},
    )
    mll = SimpleNamespace(model=deep_model_type())

    result = binary_fit.fit_binary_classifier_mll(
        mll,
        lr=0.005,
        num_epochs=250,
        batch_size=16,
        shuffle=False,
        verbose=True,
    )

    assert result == "fitted"
    assert calls["mll"] is mll
    assert calls["kwargs"]["lr"] == 0.005
    assert calls["kwargs"]["num_epochs"] == 250
    assert calls["kwargs"]["verbose"] is True
