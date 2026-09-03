from __future__ import annotations

import pytest
from pydantic import ValidationError

from bochan.serving.fastapi.schemas.material_residual import MaterialResidualTabularFitModelRequest


_STRUCTURE = {
    "format": "mapping",
    "lattice_mat": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
    "coords": [[0.0, 0.0, 0.0]],
    "elements": ["Si"],
}


def _payload(model_type: str, *, categorical: bool = False) -> dict:
    input_cols = ["structure", "temperature"]
    categorical_cols = []
    bounds = {"temperature": [300.0, 1000.0]}
    data = [
        {"structure": "s0", "temperature": 400.0, "energy": -1.0, "strength": 2.0},
        {"structure": "s0", "temperature": 800.0, "energy": -0.8, "strength": 2.5},
    ]
    if categorical:
        input_cols.append("route")
        categorical_cols.append("route")
        for index, row in enumerate(data):
            row["route"] = "A" if index == 0 else "B"

    return {
        "model_config": {
            "task_type": "multi_objective",
            "model_type": model_type,
            "model_kwargs": {
                "model_name": "medium-mpa-0" if model_type.startswith("mace") else "0.3.0",
                "pretrained_output_index": 0,
            },
        },
        "data": data,
        "input_cols": input_cols,
        "target_cols": ["energy", "strength"],
        "categorical_cols": categorical_cols,
        "bounds": bounds,
        "structure_col": "structure",
        "structure_catalog": {"s0": _STRUCTURE},
    }


def test_fastapi_accepts_independent_multioutput_residual() -> None:
    request = MaterialResidualTabularFitModelRequest.model_validate(
        _payload("mace_multioutput_residual_gp")
    )

    assert request.bo_model_config.model_type == "mace_multioutput_residual_gp"
    assert request.bo_model_config.model_kwargs["pretrained_output_index"] == 0


def test_fastapi_accepts_mixed_independent_multioutput_residual() -> None:
    request = MaterialResidualTabularFitModelRequest.model_validate(
        _payload("mace_mixed_multioutput_residual_gp", categorical=True)
    )

    assert request.categorical_cols == ["route"]


def test_fastapi_rejects_out_of_range_independent_baseline_index() -> None:
    payload = _payload("mace_multioutput_residual_gp")
    payload["model_config"]["model_kwargs"]["pretrained_output_index"] = 2

    with pytest.raises(ValidationError, match="pretrained_output_index"):
        MaterialResidualTabularFitModelRequest.model_validate(payload)


def test_fastapi_rejects_single_target_independent_multioutput() -> None:
    payload = _payload("mace_multioutput_residual_gp")
    payload["target_cols"] = ["energy"]

    with pytest.raises(ValidationError, match="at least two"):
        MaterialResidualTabularFitModelRequest.model_validate(payload)
