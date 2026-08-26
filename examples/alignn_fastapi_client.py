"""Minimal ALIGNN-GP FastAPI smoke client.

Start the API first:

    python -m uvicorn bochan.serving.fastapi.app:app --host 127.0.0.1 --port 8000

The example intentionally uses ``skip_fit=True`` so it is a quick integration
smoke test. Remove ``skip_fit`` and configure a scientifically appropriate
ALIGNN checkpoint/training setup before using the result for materials work.
"""

from __future__ import annotations

import os

import httpx

BASE_URL = os.environ.get("BOCHAN_API_URL", "http://127.0.0.1:8000/api/v1")


def cubic_si(a: float) -> dict[str, object]:
    """Return a tiny periodic Si structure in the FastAPI mapping format."""

    return {
        "format": "mapping",
        "lattice_mat": [
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, a],
        ],
        "coords": [[0.0, 0.0, 0.0]],
        "elements": ["Si"],
        "cartesian": False,
    }


def api(client: httpx.Client, method: str, path: str, **kwargs):
    """Call one API endpoint and raise with the server response on failure."""

    response = client.request(method, f"{BASE_URL}{path}", **kwargs)
    if response.is_error:
        raise RuntimeError(
            f"{method} {path} -> {response.status_code}: {response.text}"
        )
    return response.json()


def main() -> None:
    fit_payload = {
        "data": [
            {"phase": "alpha", "temperature": 900.0, "pressure": 0.8, "property": 0.40},
            {"phase": "beta", "temperature": 930.0, "pressure": 1.0, "property": 0.72},
            {"phase": "alpha", "temperature": 980.0, "pressure": 1.2, "property": 0.68},
            {"phase": "beta", "temperature": 1020.0, "pressure": 1.4, "property": 1.05},
            {"phase": "alpha", "temperature": 1070.0, "pressure": 1.6, "property": 0.93},
            {"phase": "beta", "temperature": 1100.0, "pressure": 1.8, "property": 1.32},
        ],
        "input_cols": ["phase", "temperature", "pressure"],
        "target_cols": "property",
        "bounds": {
            "temperature": [850.0, 1150.0],
            "pressure": [0.5, 2.0],
        },
        "structure_col": "phase",
        "structure_catalog": {
            "alpha": cubic_si(5.43),
            "beta": cubic_si(5.55),
        },
        "model_config": {
            "task_type": "regression",
            "model_type": "alignn_gp",
            "model_kwargs": {
                "latent_dim": 8,
            },
        },
        "fit_config": {
            "skip_fit": True,
        },
    }

    with httpx.Client(timeout=120.0) as client:
        fitted = api(client, "POST", "/tabular/alignn/models", json=fit_payload)
        model_id = fitted["model_id"]
        print("FIT")
        print(fitted)

        predicted = api(
            client,
            "POST",
            f"/tabular/alignn/models/{model_id}/predict",
            json={
                "data": [
                    {"phase": "alpha", "temperature": 1000.0, "pressure": 1.1},
                    {"phase": "beta", "temperature": 1000.0, "pressure": 1.1},
                ],
                "include_input": True,
            },
        )
        print("\nPREDICT")
        print(predicted)

        candidates = api(
            client,
            "POST",
            f"/tabular/alignn/models/{model_id}/candidates",
            json={
                "acquisition_config": {"name": "logei"},
                "optimize_config": {
                    "q": 1,
                    "num_restarts": 4,
                    "raw_samples": 32,
                },
                "structure_ids": ["alpha", "beta"],
            },
        )
        print("\nCANDIDATE")
        print(candidates)


if __name__ == "__main__":
    main()
