from __future__ import annotations

import os

from bochan.serving.webapp.model_runtime import apply_web_model_runtime_defaults


def test_web_tabpfn_disables_automatic_browser_authentication(monkeypatch) -> None:
    monkeypatch.delenv("TABPFN_NO_BROWSER", raising=False)

    apply_web_model_runtime_defaults(
        {},
        model_type="tabpfn",
        fit_maxiter=128,
    )

    assert os.environ["TABPFN_NO_BROWSER"] == "1"


def test_web_tabpfn_preserves_explicit_browser_authentication_setting(monkeypatch) -> None:
    monkeypatch.setenv("TABPFN_NO_BROWSER", "0")

    apply_web_model_runtime_defaults(
        {},
        model_type="tabpfn",
        fit_maxiter=128,
    )

    assert os.environ["TABPFN_NO_BROWSER"] == "0"


def test_non_tabpfn_web_model_does_not_change_tabpfn_auth_environment(monkeypatch) -> None:
    monkeypatch.delenv("TABPFN_NO_BROWSER", raising=False)

    apply_web_model_runtime_defaults(
        {},
        model_type="random_forest",
        fit_maxiter=128,
    )

    assert "TABPFN_NO_BROWSER" not in os.environ
