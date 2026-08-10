from __future__ import annotations

import os

import pytest

import bochan.tabpfn_assets as tabpfn_assets
from bochan.serving.webapp import model_runtime


def _patch_asset_names(monkeypatch) -> None:
    names = {
        "classifier": "tabpfn-v3-classifier-v3_default.ckpt",
        "regressor": "tabpfn-v3-regressor-v3_default.ckpt",
    }
    monkeypatch.setattr(
        tabpfn_assets,
        "_default_model_filename",
        lambda which: names[which],
    )


def test_tabpfn_asset_status_requires_both_default_checkpoints(tmp_path, monkeypatch) -> None:
    _patch_asset_names(monkeypatch)
    classifier = tmp_path / "tabpfn-v3-classifier-v3_default.ckpt"
    classifier.write_bytes(b"classifier")

    status = tabpfn_assets.tabpfn_asset_status(tmp_path)

    assert status["available"] is False
    assert status["models"]["classifier"]["exists"] is True
    assert status["models"]["regressor"]["exists"] is False
    assert status["missing_models"] == ["tabpfn-v3-regressor-v3_default.ckpt"]


def test_require_preloaded_tabpfn_assets_fails_before_model_construction(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_asset_names(monkeypatch)

    with pytest.raises(RuntimeError, match="preloaded model weight"):
        tabpfn_assets.require_preloaded_tabpfn_assets(tmp_path)


def test_preload_downloads_classifier_and_regressor_without_browser_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_asset_names(monkeypatch)
    monkeypatch.delenv("TABPFN_NO_BROWSER", raising=False)
    downloaded: list[str] = []

    def fake_download(root, which):
        downloaded.append(which)
        path = root / tabpfn_assets._default_model_filename(which)
        path.write_bytes(which.encode())
        return path

    monkeypatch.setattr(tabpfn_assets, "_download_default_model", fake_download)

    status = tabpfn_assets.preload_tabpfn_assets(tmp_path)

    assert downloaded == ["classifier", "regressor"]
    assert status["available"] is True
    assert os.environ["TABPFN_NO_BROWSER"] == "1"


def test_preload_can_explicitly_allow_interactive_browser_auth(tmp_path, monkeypatch) -> None:
    _patch_asset_names(monkeypatch)
    monkeypatch.setenv("TABPFN_NO_BROWSER", "1")

    def fake_download(root, which):
        path = root / tabpfn_assets._default_model_filename(which)
        path.write_bytes(which.encode())
        return path

    monkeypatch.setattr(tabpfn_assets, "_download_default_model", fake_download)

    tabpfn_assets.preload_tabpfn_assets(tmp_path, allow_browser_auth=True)

    assert "TABPFN_NO_BROWSER" not in os.environ


def test_web_tabpfn_requires_preloaded_assets_and_pins_noninteractive_v3(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setenv("TABPFN_NO_BROWSER", "0")
    monkeypatch.setenv("TABPFN_MODEL_VERSION", "v2.6")
    monkeypatch.setattr(model_runtime, "_web_request_context_active", lambda: True)
    monkeypatch.setattr(
        model_runtime,
        "require_preloaded_tabpfn_assets",
        lambda: calls.append(True) or {"available": True},
    )

    kwargs = model_runtime.apply_web_model_runtime_defaults(
        {},
        model_type="tabpfn",
        fit_maxiter=128,
    )

    assert calls == [True]
    assert os.environ["TABPFN_NO_BROWSER"] == "1"
    assert os.environ["TABPFN_MODEL_VERSION"] == "v3"
    assert kwargs["n_estimators"] == 4
    assert kwargs["show_progress_bar"] is False
    assert kwargs["n_preprocessing_jobs"] == 1


def test_web_tabpfn_missing_assets_error_is_not_replaced_by_auth_fallback(monkeypatch) -> None:
    monkeypatch.setattr(model_runtime, "_web_request_context_active", lambda: True)
    monkeypatch.setattr(
        model_runtime,
        "require_preloaded_tabpfn_assets",
        lambda: (_ for _ in ()).throw(RuntimeError("weights missing")),
    )

    with pytest.raises(RuntimeError, match="weights missing"):
        model_runtime.apply_web_model_runtime_defaults(
            {},
            model_type="tabpfn",
            fit_maxiter=128,
        )


def test_runtime_default_resolution_outside_request_does_not_require_assets(monkeypatch) -> None:
    monkeypatch.setattr(model_runtime, "_web_request_context_active", lambda: False)
    monkeypatch.setattr(
        model_runtime,
        "require_preloaded_tabpfn_assets",
        lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    kwargs = model_runtime.apply_web_model_runtime_defaults(
        {},
        model_type="tabpfn",
        fit_maxiter=128,
    )

    assert kwargs["n_estimators"] == 4


def test_injected_tabpfn_estimator_does_not_require_deployment_assets(monkeypatch) -> None:
    estimator = object()
    monkeypatch.setattr(model_runtime, "_web_request_context_active", lambda: True)
    monkeypatch.setattr(
        model_runtime,
        "require_preloaded_tabpfn_assets",
        lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    kwargs = model_runtime.apply_web_model_runtime_defaults(
        {"estimator": estimator},
        model_type="tabpfn",
        fit_maxiter=128,
    )

    assert kwargs == {"estimator": estimator}


def test_non_tabpfn_web_model_does_not_change_tabpfn_auth_environment(monkeypatch) -> None:
    monkeypatch.delenv("TABPFN_NO_BROWSER", raising=False)
    monkeypatch.delenv("TABPFN_MODEL_VERSION", raising=False)

    model_runtime.apply_web_model_runtime_defaults(
        {},
        model_type="random_forest",
        fit_maxiter=128,
    )

    assert "TABPFN_NO_BROWSER" not in os.environ
    assert "TABPFN_MODEL_VERSION" not in os.environ
