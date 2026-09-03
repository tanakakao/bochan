from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from bochan.models.regression.gaussian.materials.common.residual import (
    DirectMaterialPredictor,
    ResidualMaterialGPModel,
)
from bochan.serving.fastapi.routers import create_api_router
from bochan.serving.fastapi.schemas.material_residual import (
    MaterialResidualTabularFitModelRequest,
)


_STRUCTURE = {
    "format": "mapping",
    "lattice_mat": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
    "coords": [[0.0, 0.0, 0.0]],
    "elements": ["Si"],
}


def _request(model_type: str, *, targets: list[str], categorical: bool = False, **kwargs):
    data = [
        {"structure": "s0", "temperature": 300.0, **{target: 1.0 for target in targets}}
    ]
    input_cols = ["structure", "temperature"]
    categorical_cols: list[str] = []
    bounds = {"temperature": [250.0, 500.0]}
    if categorical:
        data[0]["route"] = "A"
        input_cols.append("route")
        categorical_cols.append("route")
    return MaterialResidualTabularFitModelRequest(
        data=data,
        model_config={
            "task_type": "multi_objective" if len(targets) > 1 else "regression",
            "model_type": model_type,
            "model_kwargs": kwargs,
        },
        input_cols=input_cols,
        target_cols=targets,
        categorical_cols=categorical_cols,
        bounds=bounds,
        structure_col="structure",
        structure_catalog={"s0": _STRUCTURE},
    )


def test_scalar_and_multitask_residual_requests_are_json_safe():
    scalar = _request("chgnet_residual_gp", targets=["energy"])
    assert scalar.bo_model_config.model_kwargs["model_name"] == "0.3.0"

    multitask = _request(
        "mace_multitask_residual_gp",
        targets=["energy", "hardness"],
        pretrained_output_index=1,
    )
    assert multitask.bo_model_config.model_kwargs["pretrained_output_index"] == 1
    assert multitask.bo_model_config.model_kwargs["head"] if "head" in multitask.bo_model_config.model_kwargs else True


def test_mixed_residual_request_requires_categorical_process_input():
    with pytest.raises(ValueError, match="mixed_residual_gp"):
        _request("m3gnet_mixed_residual_gp", targets=["energy"])

    request = _request(
        "m3gnet_mixed_residual_gp",
        targets=["energy"],
        categorical=True,
    )
    assert request.categorical_cols == ["route"]


def test_pretrained_output_index_is_multitask_only_and_range_checked():
    with pytest.raises(ValueError, match="only valid for multitask"):
        _request(
            "mace_residual_gp",
            targets=["energy"],
            pretrained_output_index=0,
        )

    with pytest.raises(ValueError, match="0 <= index < 2"):
        _request(
            "chgnet_multitask_residual_gp",
            targets=["energy", "hardness"],
            pretrained_output_index=2,
        )


class _ZeroPredictor(DirectMaterialPredictor):
    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.new_zeros(*X.shape[:-1], 1)


def test_residual_wrapper_builds_mll_for_internal_exact_gp():
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [0.2], [0.9]], dtype=torch.double)
    residual_gp = SingleTaskGP(train_X, train_Y)
    model = ResidualMaterialGPModel(
        predictor=_ZeroPredictor(),
        residual_model=residual_gp,
    )

    mll = model.make_mll()
    assert isinstance(mll, ExactMarginalLogLikelihood)
    assert mll.model is residual_gp
    assert model.likelihood is residual_gp.likelihood


def test_material_residual_router_is_mounted():
    paths = {route.path for route in create_api_router(prefix="/api/v1").routes}
    assert "/api/v1/tabular/material-residual/models" in paths
    assert "/api/v1/tabular/material-residual/models/{model_id}/candidates" in paths
    assert "/api/v1/tabular/material-residual/models/{model_id}/save" in paths
