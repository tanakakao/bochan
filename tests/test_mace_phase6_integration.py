"""Real pretrained MACE integration closure for the FastAPI-to-BoTorch path."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("mace")

from bochan.models.regression.gaussian.deep import MACEGPModel
from bochan.serving.fastapi.schemas.mace_tabular import MACETabularFitModelRequest
from bochan.serving.fastapi.services.mace_tabular import (
    build_mace_fit_response,
    fit_mace_tabular_optimizer,
)

_MODEL_NAME = "medium-mpa-0"


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


def test_real_pretrained_mace_closes_fastapi_tabular_posterior_path() -> None:
    request = MACETabularFitModelRequest.model_validate(
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
                "model_type": "mace_gp",
                "model_kwargs": {
                    "model_name": _MODEL_NAME,
                    "latent_dim": 8,
                },
            },
            "fit_config": {"skip_fit": True},
        }
    )

    optimizer = fit_mace_tabular_optimizer(request)
    bundle = optimizer.bo.bundle
    assert bundle is not None
    assert isinstance(bundle.model, MACEGPModel)
    assert bundle.model.material_encoder.initialization == "pretrained"
    assert bundle.model.material_encoder.model_name == _MODEL_NAME
    assert bundle.model.material_encoder.output_dim > 0

    encoder_parameter = next(bundle.model.material_encoder.encoder.parameters())
    projection_parameter = next(bundle.model.projection.parameters())
    assert encoder_parameter.dtype in {torch.float32, torch.float64}
    assert optimizer.dataset.X.dtype == torch.float64
    assert projection_parameter.dtype == torch.float64

    posterior = bundle.model.posterior(optimizer.dataset.X[:2])
    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

    response = build_mace_fit_response("real-mace", optimizer)
    metadata = response.metadata["mace"]
    assert metadata["encoder_initialization"] == "pretrained"
    assert metadata["model_name"] == _MODEL_NAME
    assert metadata["encoder_output_dim"] == bundle.model.material_encoder.output_dim
    assert metadata["representation_mode"] == "invariant_l0"
    assert metadata["pooling"] == "mean"
    assert metadata["structure_ids"] == ["alpha", "beta"]
