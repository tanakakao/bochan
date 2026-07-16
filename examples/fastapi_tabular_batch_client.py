"""Run the resin multi-model candidate matrix through the bochan FastAPI API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pandas as pd


def build_payload(csv_path: str | Path = "resin.csv") -> dict[str, Any]:
    """Load the training CSV and build the FastAPI request payload.

    Args:
        csv_path: Path to the resin training data.

    Returns:
        JSON-compatible request payload using the same settings as the original
        direct Python loop.
    """
    df = pd.read_csv(csv_path)
    return {
        "data": df.to_dict(orient="records"),
        "model_types": [
            "base",
            "deepgp",
            "deepkernel",
            "hetero",
            "rrp",
            "pca",
            "rembo",
            "saas",
        ],
        "acquisition_names": [
            "ehvi",
            "nehvi",
            "nparego",
            "bald",
            "entropy",
            "variance",
            "straddle",
            "icu",
            "nsgaii",
        ],
        "optimizers": [
            "optimize_acqf",
            "torch",
            "thompson_sampling",
            "pso",
            "ga",
            "cmaes",
        ],
        "input_cols": [
            "raw material 1",
            "raw material 2",
            "raw material 3",
            "temperature",
            "time",
        ],
        "target_cols": ["property", "property2"],
        "model_config": {
            "task_type": "regression",
            "input_transform_config": {
                "perturbation": False,
                "n_w": 4,
                "std": 0.1,
            },
        },
        "fit_config": {"maxiter": 128},
        "optimize_config": {
            "q": 2,
            "num_restarts": 2,
            "raw_samples": 4,
        },
        "continue_on_error": True,
    }


def run(
    csv_path: str | Path = "resin.csv",
    base_url: str = "http://127.0.0.1:8000",
) -> dict[str, Any]:
    """Submit the full candidate matrix and return the parsed response.

    Args:
        csv_path: Path to the resin training data.
        base_url: Base URL of the running bochan FastAPI server.

    Returns:
        Parsed batch result containing successful candidates and per-run errors.
    """
    payload = build_payload(csv_path)
    with httpx.Client(timeout=None) as client:
        response = client.post(
            f"{base_url}/api/v1/tabular/batch-candidates",
            json=payload,
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    result = run()
    print(
        "models={n_models}, runs={n_runs}, success={n_success}, failed={n_failed}".format(
            **result
        )
    )
    for item in result["results"]:
        print(
            item["model_type"],
            item.get("acquisition_name"),
            item.get("optimizer"),
            item["status"],
            item.get("error"),
        )
