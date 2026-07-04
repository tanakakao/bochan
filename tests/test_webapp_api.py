"""Smoke tests for the React-oriented FastAPI application."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from bochan.serving.webapp.app import create_app
from bochan.serving.webapp.workflows import _figure_payload


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
    assert payload["visualizations"] == ["yyplot", "prediction-1d", "prediction-2d"]


def test_figure_payload_is_json_safe() -> None:
    graph_objects = pytest.importorskip("plotly.graph_objects")
    figure = graph_objects.Figure(
        data=[graph_objects.Scatter(x=[0.0, 1.0], y=[1.0, 2.0])]
    )

    payload = _figure_payload(
        figure,
        figure_id="test",
        title="Test figure",
        description="JSON serialization test.",
    )

    assert payload["id"] == "test"
    assert payload["figure"]["data"][0]["type"] == "scatter"
    json.dumps(payload)
