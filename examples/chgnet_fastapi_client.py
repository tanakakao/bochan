"""CHGNet FastAPI fit/candidate/save/load example.

Start the API first::

    python -m pip install -e ".[api,tabular,materials]"
    python -m uvicorn bochan.serving.fastapi.app:app --host 127.0.0.1 --port 8000

The example uses ``skip_fit=True`` to keep the plumbing demo short while still
loading the real pretrained CHGNet representation model. Artifact loading uses
pickle; trust only artifacts from trusted sources.
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
        raise RuntimeError(f"{method} {path} -> {response.status_code}: {response.text}")
    return response.json()


def main() -> None:
    fit_payload = {
        "data": [
            {"phase": "alpha", "temperature": 900.0, "property": 0.40},
            {"phase": "beta", "temperature": 950.0, "property": 0.80},
            {"phase": "alpha", "temperature": 1000.0, "property": 0.70},
            {"phase": "beta", "temperature": 1050.0, "property": 1.10},
        ],
        "input_cols": ["phase", "temperature"],
        "target_cols": "property",
        "bounds": {"temperature": [850.0, 1150.0]},
        "structure_col": "phase",
        "structure_catalog": {
            "alpha": structure(5.43),
            "beta": structure(5.55),
        },
        "model_config": {
            "task_type": "regression",
            "model_type": "chgnet_gp",
            "model_kwargs": {
                "model_name": "0.3.0",
                "latent_dim": 16,
            },
        },
        "fit_config": {"skip_fit": True},
    }

    with httpx.Client(timeout=180.0) as client:
        fitted = api(client, "POST", "/tabular/chgnet/models", json=fit_payload)
        model_id = fitted["model_id"]
        print("FIT", model_id)
        print("CHGNET", fitted["metadata"]["chgnet"])

        candidates = api(
            client,
            "POST",
            f"/tabular/chgnet/models/{model_id}/candidates",
            json={
                "acquisition_config": {"name": "logei"},
                "optimize_config": {
                    "q": 1,
                    "num_restarts": 4,
                    "raw_samples": 64,
                },
                "structure_ids": ["alpha", "beta"],
            },
        )
        print("CANDIDATES", candidates)

        saved = api(
            client,
            "POST",
            f"/tabular/chgnet/models/{model_id}/save",
            json={"filename": "chgnet-structure-process", "overwrite": True},
        )
        print("SAVE", saved["filename"])

        loaded = api(
            client,
            "POST",
            "/tabular/chgnet/models/load",
            json={
                "filename": saved["filename"],
                "map_location": "cpu",
                "trust_pickle": True,
            },
        )
        restored_id = loaded["model_id"]
        print("LOAD", restored_id)

        prediction = api(
            client,
            "POST",
            f"/tabular/chgnet/models/{restored_id}/predict",
            json={
                "data": [{"phase": "beta", "temperature": 1010.0}],
                "include_input": True,
            },
        )
        print("PREDICT", prediction)


if __name__ == "__main__":
    main()
