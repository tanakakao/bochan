"""Real pretrained M3GNet integration closure for the FastAPI-to-BoTorch path."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("matgl")
pytest.importorskip("pymatgen")

from bochan.models.regression.gaussian.deep import M3GNetGPModel
from bochan.serving.fastapi.schemas.m3gnet_tabular import M3GNetTabularFitModelRequest
from bochan.serving.fastapi.services.m3gnet_tabular import (
    build_m3gnet_fit_response,
    fit_m3gnet_tabular_optimizer,
)

_MODEL_NAME = "M3GNet-PES-MatPES-PBE-2025.2"


def _structure(scale: float) -> dict[str, object]:
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


def test_real_pretrained_m3gnet_closes_fastapi_tabular_posterior_path() -> None:
    request = M3GNetTabularFitModelRequest.model_validate(
        {
            "data": [
                {"phase": "alpha", "property": 0.4},
                {"phase": "beta", "property": 0.8},
                {"phase": "alpha", "property": 0.6},
                {"phase": "beta", "property": 1.0},
            ],
            "input_cols": ["phase"],
            "target_cols": "property",
            "structure_col": "phase",
            "structure_catalog": {
                "alpha": _structure(5.43),
                "beta": _structure(5.55),
            },
            "model_config": {
                "task_type": "regression",
                "model_type": "m3gnet_gp",
                "model_kwargs": {
                    "model_name": _MODEL_NAME,
                    "latent_dim": 8,
                },
            },
            "fit_config": {"skip_fit": True},
        }
    )

    optimizer = fit_m3gnet_tabular_optimizer(request)
    bundle = optimizer.bo.bundle
    assert bundle is not None
    assert isinstance(bundle.model, M3GNetGPModel)
    assert bundle.model.material_encoder.initialization == "pretrained"
    assert bundle.model.material_encoder.model_name == _MODEL_NAME
    assert bundle.model.material_encoder.representation_mode == "mean_node"
    assert bundle.model.material_encoder.output_dim > 0

    encoder_parameter = next(bundle.model.material_encoder.encoder.parameters())
    projection_parameter = next(bundle.model.projection.parameters())
    assert encoder_parameter.dtype == torch.float32
    assert optimizer.dataset.X.dtype == torch.float64
    assert projection_parameter.dtype == torch.float64

    posterior = bundle.model.posterior(optimizer.dataset.X[:2])
    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

    response = build_m3gnet_fit_response("real-m3gnet", optimizer)
    metadata = response.metadata["m3gnet"]
    assert metadata["encoder_initialization"] == "pretrained"
    assert metadata["model_name"] == _MODEL_NAME
    assert metadata["encoder_output_dim"] == bundle.model.material_encoder.output_dim
    assert metadata["representation_mode"] == "mean_node"
    assert metadata["structure_ids"] == ["alpha", "beta"]
