"""Regression tests for project ZIP filename handling."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from bochan.serving.webapp.app import create_app


def test_renamed_zip_is_validated_as_project_archive() -> None:
    """A browser-added suffix must not route a ZIP through model deserialization."""

    client = TestClient(create_app(include_core_api=False))
    response = client.post(
        "/api/v1/model-artifacts/import?trust_pickle=false",
        content=b"not-a-project-archive",
        headers={
            "Content-Type": "application/zip",
            "X-Model-Filename": "experiment.bochan-project (1).zip",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The selected ZIP file is not a valid bochan project archive."
    )
