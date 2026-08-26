"""ALIGNN FastAPI tell/save/load roundtrip example.

Start the API first::

    python -m pip install -e ".[api,tabular]"
    python -m pip install "alignn==2026.8.11"
    python -m uvicorn bochan.serving.fastapi.app:app --host 127.0.0.1 --port 8000

The example intentionally uses ``skip_fit=True`` for a quick plumbing test.
Model artifacts use pickle on load; trust only files created by a trusted
Bochan process.
"""

from __future__ import annotations

import os

import httpx

BASE_URL = os.environ.get("BOCHAN_API_URL", "http://127.0.0.1:8000/api/v1")


def structure(scale: float) -> dict[str, object]:
    return {
        "format": "mapping",
        "lattice_mat": [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ],
        "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "elements": ["Si", "Si"],
        "cartesian": False,
    }


def api(client: httpx.Client, method: str, path: str, **kwargs):
    response = client.request(method, f"{BASE_URL}{path}", **kwargs)
    if response.is_error:
        raise RuntimeError(
            f"{method} {path} -> {response.status_code}: {response.text}"
        )
    return response.json()


def main() -> None:
    fit_payload = {
        "data": [
            {
                "phase": "alpha",
                "temperature": 900.0,
                "pressure": 0.8,
                "furnace": "A",
                "atmosphere": "air",
                "property": 0.40,
            },
            {
                "phase": "beta",
                "temperature": 950.0,
                "pressure": 1.0,
                "furnace": "B",
                "atmosphere": "N2",
                "property": 0.80,
            },
            {
                "phase": "alpha",
                "temperature": 1000.0,
                "pressure": 1.2,
                "furnace": "A",
                "atmosphere": "Ar",
                "property": 0.70,
            },
            {
                "phase": "beta",
                "temperature": 1050.0,
                "pressure": 1.4,
                "furnace": "B",
                "atmosphere": "N2",
                "property": 1.10,
            },
        ],
        "input_cols": [
            "phase",
            "temperature",
            "pressure",
            "furnace",
            "atmosphere",
        ],
        "categorical_cols": ["furnace", "atmosphere"],
        "target_cols": "property",
        "bounds": {
            "temperature": [850.0, 1150.0],
            "pressure": [0.5, 2.0],
        },
        "structure_col": "phase",
        "structure_catalog": {
            "alpha": structure(5.43),
            "beta": structure(5.55),
        },
        "model_config": {
            "task_type": "regression",
            "model_type": "alignn_gp",
            "model_kwargs": {"latent_dim": 8},
        },
        "fit_config": {"skip_fit": True},
    }

    with httpx.Client(timeout=120.0) as client:
        fitted = api(client, "POST", "/tabular/alignn/models", json=fit_payload)
        model_id = fitted["model_id"]
        print("FIT", model_id)

        told = api(
            client,
            "POST",
            f"/tabular/alignn/models/{model_id}/tell",
            json={
                "data": [
                    {
                        "phase": "beta",
                        "temperature": 1080.0,
                        "pressure": 1.6,
                        "furnace": "B",
                        "atmosphere": "Ar",
                        "property": 1.25,
                    }
                ],
                "refit": False,
            },
        )
        print("TELL n_train=", told["n_train"])

        saved = api(
            client,
            "POST",
            f"/tabular/alignn/models/{model_id}/save",
            json={"filename": "alignn-process-model", "overwrite": True},
        )
        print("SAVE", saved["filename"])

        loaded = api(
            client,
            "POST",
            "/tabular/alignn/models/load",
            json={
                "filename": saved["filename"],
                "map_location": "cpu",
                "trust_pickle": True,
            },
        )
        restored_id = loaded["model_id"]
        print("LOAD", restored_id)
        print("ALIGNN METADATA", loaded["metadata"]["alignn"])

        prediction = api(
            client,
            "POST",
            f"/tabular/alignn/models/{restored_id}/predict",
            json={
                "data": [
                    {
                        "phase": "beta",
                        "temperature": 1010.0,
                        "pressure": 1.3,
                        "furnace": "B",
                        "atmosphere": "Ar",
                    }
                ],
                "include_input": True,
            },
        )
        print("PREDICT", prediction)


if __name__ == "__main__":
    main()
