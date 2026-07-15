"""Smoke tests for the React-oriented FastAPI application."""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from bochan.serving.webapp.app import create_app
from bochan.serving.webapp.logging import JsonLogFormatter, reset_request_id, set_request_id
from bochan.serving.webapp.workflows import _figure_payload


def test_web_health() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


def test_request_id_header_is_preserved() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/health", headers={"X-Request-ID": "test-request-id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-id"


def test_web_capabilities() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_types"] == ["regression"]
    assert "base" in payload["model_types"]
    assert "EI" in payload["acquisitions"]
    assert payload["visualizations"] == ["yyplot", "prediction-1d", "prediction-2d"]
    assert payload["logging"]["recent_logs_endpoint"] == "/api/v1/logs"


def test_recent_logs_endpoint() -> None:
    client = TestClient(create_app())
    client.get("/api/v1/health", headers={"X-Request-ID": "recent-log-test"})
    response = client.get("/api/v1/logs", params={"limit": 100, "request_id": "recent-log-test"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["log_file"].endswith("bochan-web.jsonl")
    assert any(entry["event"] == "http_request_started" for entry in payload["entries"])
    assert any(entry["event"] == "http_request_completed" for entry in payload["entries"])


def test_json_formatter_includes_structured_fields() -> None:
    token = set_request_id("formatter-test")
    try:
        record = logging.LogRecord(
            name="bochan.web.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="structured message",
            args=(),
            exc_info=None,
        )
        record.event = "structured_test"
        record.duration_ms = 12.5
        payload = json.loads(JsonLogFormatter().format(record))
    finally:
        reset_request_id(token)

    assert payload["request_id"] == "formatter-test"
    assert payload["event"] == "structured_test"
    assert payload["duration_ms"] == 12.5


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
