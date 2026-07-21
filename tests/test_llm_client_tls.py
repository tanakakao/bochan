"""TLS configuration tests for the provider-backed LLM client."""

from __future__ import annotations

import ssl
import sys
from types import ModuleType
from typing import Any

import pytest

import bochan.llm.client as client_module
from bochan.llm import LLMConfig


class _FakeSSLContext:
    def __init__(self) -> None:
        self.loaded_cafile: str | None = None
        self.check_hostname = True
        self.verify_mode = ssl.CERT_REQUIRED

    def load_verify_locations(self, *, cafile: str) -> None:
        self.loaded_cafile = cafile


class _FakeHTTPXClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _clear_ca_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        monkeypatch.delenv(name, raising=False)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    module = ModuleType("openai")

    class FakeOpenAI:
        last_kwargs: dict[str, Any] | None = None

        def __init__(self, **kwargs: Any) -> None:
            type(self).last_kwargs = kwargs

    module.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return FakeOpenAI


def _install_fake_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("httpx")
    module.Client = _FakeHTTPXClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", module)


def test_openai_client_keeps_sdk_default_tls_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ca_environment(monkeypatch)
    fake_openai = _install_fake_openai(monkeypatch)

    client_module.OpenAIClient(LLMConfig(api_key="test-key"))

    assert fake_openai.last_kwargs == {
        "api_key": "test-key",
        "timeout": 60.0,
    }


def test_openai_client_loads_configured_ca_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _clear_ca_environment(monkeypatch)
    fake_openai = _install_fake_openai(monkeypatch)
    _install_fake_httpx(monkeypatch)
    ssl_context = _FakeSSLContext()
    monkeypatch.setattr(client_module.ssl, "create_default_context", lambda: ssl_context)
    ca_bundle = tmp_path / "company-ca.pem"
    ca_bundle.write_text("placeholder", encoding="utf-8")

    client_module.OpenAIClient(
        LLMConfig(
            api_key="test-key",
            ca_bundle_path=str(ca_bundle),
        )
    )

    assert ssl_context.loaded_cafile == str(ca_bundle)
    assert fake_openai.last_kwargs is not None
    http_client = fake_openai.last_kwargs["http_client"]
    assert isinstance(http_client, _FakeHTTPXClient)
    assert http_client.kwargs["verify"] is ssl_context


def test_openai_client_warns_when_certificate_verification_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_ca_environment(monkeypatch)
    fake_openai = _install_fake_openai(monkeypatch)
    _install_fake_httpx(monkeypatch)
    ssl_context = _FakeSSLContext()
    monkeypatch.setattr(client_module.ssl, "create_default_context", lambda: ssl_context)

    with pytest.warns(RuntimeWarning, match="disables HTTPS certificate verification"):
        client_module.OpenAIClient(
            LLMConfig(
                api_key="test-key",
                ssl_verify=False,
            )
        )

    assert ssl_context.check_hostname is False
    assert ssl_context.verify_mode == ssl.CERT_NONE
    assert fake_openai.last_kwargs is not None
    assert "http_client" in fake_openai.last_kwargs


def test_openai_client_rejects_missing_ca_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _clear_ca_environment(monkeypatch)
    _install_fake_openai(monkeypatch)
    missing_path = tmp_path / "missing-company-ca.pem"

    with pytest.raises(FileNotFoundError, match="CA bundle file was not found"):
        client_module.OpenAIClient(
            LLMConfig(
                api_key="test-key",
                ca_bundle_path=str(missing_path),
            )
        )
