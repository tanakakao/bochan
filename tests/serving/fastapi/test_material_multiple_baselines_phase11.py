from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.models import SingleTaskGP
from pydantic import ValidationError

from bochan.api import MultiOutputConfig
from bochan.models.regression.gaussian.materials.common import (
    DirectMaterialPredictor,
    MaterialBaselineSpec,
    MaterialPropertyContract,
    MultipleBaselineModelListGP,
    ResidualMaterialGPModel,
)
from bochan.serving.fastapi.schemas.material_residual import MaterialResidualTabularFitModelRequest
from bochan.serving.fastapi.services.material_residual import _model_config_from_request
from bochan.tabular.structure.material_multi_baseline import _build_multiple_baseline_wrapper

_STRUCTURE = {
    "format": "mapping",
    "lattice_mat": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
    "coords": [[0.0, 0.0, 0.0]],
    "elements": ["Si"],
}


def _payload(*, mixed: bool = False) -> dict:
    input_cols = ["structure", "temperature"]
    categorical_cols = []
    data = [
        {
            "structure": "s0",
            "temperature": 400.0,
            "energy": -1.0,
            "band_gap": 1.1,
            "strength": 2.0,
        },
        {
            "structure": "s0",
            "temperature": 800.0,
            "energy": -0.8,
            "band_gap": 1.3,
            "strength": 2.5,
        },
    ]
    if mixed:
        input_cols.append("route")
        categorical_cols.append("route")
        data[0]["route"] = "A"
        data[1]["route"] = "B"

    return {
        "model_config": {
            "task_type": "multi_objective",
            "model_type": (
                "material_mixed_multi_baseline_residual_gp"
                if mixed
                else "material_multi_baseline_residual_gp"
            ),
            "model_kwargs": {},
        },
        "data": data,
        "input_cols": input_cols,
        "target_cols": ["energy", "band_gap", "strength"],
        "categorical_cols": categorical_cols,
        "bounds": {"temperature": [300.0, 1000.0]},
        "structure_col": "structure",
        "structure_catalog": {"s0": _STRUCTURE},
        "baseline_specs": [
            {
                "family": "mace",
                "output_name": "energy",
                "quantity": "energy",
                "unit": "eV",
                "aggregation": "total",
                "model_name": "medium-mpa-0",
            },
            {
                "family": "m3gnet",
                "output_name": "band_gap",
                "quantity": "band_gap",
                "unit": "eV",
                "aggregation": "intensive",
                "model_name": "M3GNet-PES-MatPES-PBE-2025.2",
            },
        ],
        "ordinary_family": "chgnet",
        "ordinary_model_kwargs": {"model_name": "0.3.0"},
    }


def test_schema_accepts_cross_family_multiple_baselines() -> None:
    request = MaterialResidualTabularFitModelRequest.model_validate(_payload())

    assert request.bo_model_config.model_type == "material_multi_baseline_residual_gp"
    assert [route.family for route in request.baseline_specs or []] == ["mace", "m3gnet"]
    assert request.ordinary_family == "chgnet"


def test_schema_accepts_mixed_multiple_baselines() -> None:
    request = MaterialResidualTabularFitModelRequest.model_validate(_payload(mixed=True))

    assert request.bo_model_config.model_type == "material_mixed_multi_baseline_residual_gp"
    assert request.categorical_cols == ["route"]


def test_schema_rejects_duplicate_baseline_output() -> None:
    payload = _payload()
    payload["baseline_specs"][1]["output_name"] = "energy"

    with pytest.raises(ValidationError, match="Multiple baselines target output"):
        MaterialResidualTabularFitModelRequest.model_validate(payload)


def test_schema_requires_ordinary_family_for_unassigned_output() -> None:
    payload = _payload()
    payload["ordinary_family"] = None
    payload["ordinary_model_kwargs"] = {}

    with pytest.raises(ValidationError, match="ordinary_family is required"):
        MaterialResidualTabularFitModelRequest.model_validate(payload)


def test_service_builds_material_baseline_specs() -> None:
    request = MaterialResidualTabularFitModelRequest.model_validate(_payload())

    config = _model_config_from_request(request)
    routes = config["model_kwargs"]["baseline_routes"]

    assert len(routes) == 2
    assert isinstance(routes[0]["spec"], MaterialBaselineSpec)
    assert routes[0]["spec"].family == "mace"
    assert routes[0]["spec"].property.unit == "eV"
    assert routes[1]["spec"].output_name == "band_gap"
    assert config["model_kwargs"]["ordinary_family"] == "chgnet"


class _ConstantPredictor(DirectMaterialPredictor):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.full((*X.shape[:-1], 1), self.value, dtype=X.dtype, device=X.device)


def _residual(value: float) -> ResidualMaterialGPModel:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    predictor = _ConstantPredictor(value)
    train_Y = torch.tensor([[value], [value + 0.1], [value - 0.1]], dtype=torch.double)
    return ResidualMaterialGPModel(
        predictor=predictor,
        residual_model=SingleTaskGP(train_X, train_Y - predictor(train_X)),
    )


def test_wrapper_factory_attaches_specs_and_builds_validated_model_list() -> None:
    energy_spec = MaterialBaselineSpec(
        family="mace",
        output_name="energy",
        property=MaterialPropertyContract("energy", "eV", "total"),
    )
    gap_spec = MaterialBaselineSpec(
        family="m3gnet",
        output_name="band_gap",
        property=MaterialPropertyContract("band_gap", "eV", "intensive"),
    )
    energy_model = _residual(-1.0)
    gap_model = _residual(1.2)
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    strength_model = SingleTaskGP(
        train_X,
        torch.tensor([[2.0], [2.5], [2.2]], dtype=torch.double),
    )
    config = MultiOutputConfig(
        wrapper_kwargs={
            "output_names": ["energy", "band_gap", "strength"],
            "baseline_specs": [energy_spec, gap_spec],
        }
    )

    model = _build_multiple_baseline_wrapper(
        submodels=[energy_model, gap_model, strength_model],
        output_configs=[],
        config=SimpleNamespace(wrapper_kwargs=config.wrapper_kwargs),
    )

    assert isinstance(model, MultipleBaselineModelListGP)
    assert energy_model.baseline_spec == energy_spec
    assert gap_model.baseline_spec == gap_spec
    assert model.baseline_plan.baseline_output_indices == (0, 1)
    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    assert posterior.mean.shape == torch.Size([2, 3])
