"""Smoke tests for the React-oriented FastAPI application."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from bochan.serving.webapp.app import create_app


def test_web_health() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "application": "bochan-web"}


def test_web_capabilities() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_types"] == ["regression"]
    assert "base" in payload["model_types"]
    assert "EI" in payload["acquisitions"]
