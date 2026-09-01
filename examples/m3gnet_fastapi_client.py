"""Minimal client for the M3GNet structure-aware FastAPI surface."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:8000/api/v1/tabular/m3gnet/models"
MODEL_NAME = "M3GNet-PES-MatPES-PBE-2025.2"


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def structure(scale: float) -> dict[str, Any]:
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
        "bounds": {"temperature": [850.0, 1100.0]},
        "structure_col": "phase",
        "structure_catalog": {
            "alpha": structure(5.43),
            "beta": structure(5.50),
        },
        "model_config": {
            "task_type": "regression",
            "model_type": "m3gnet_gp",
            "model_kwargs": {"model_name": MODEL_NAME, "latent_dim": 16},
        },
    }
    fitted = post_json(BASE_URL, fit_payload)
    model_id = fitted["model_id"]
    print("model_id:", model_id)
    print("m3gnet metadata:", fitted["metadata"]["m3gnet"])

    prediction = post_json(
        f"{BASE_URL}/{model_id}/predict",
        {
            "data": [{"phase": "alpha", "temperature": 980.0}],
            "include_input": True,
        },
    )
    print("prediction:", prediction)

    candidate = post_json(
        f"{BASE_URL}/{model_id}/candidates",
        {
            "acquisition_config": {"name": "logei"},
            "optimize_config": {"q": 1},
            "structure_ids": ["alpha", "beta"],
        },
    )
    print("candidate:", candidate)


if __name__ == "__main__":
    main()
