from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from bochan.serving.fastapi import create_app
from bochan.serving.fastapi.schemas.tabular import TabularFitModelRequest
from bochan.serving.fastapi.services.tabular import (
    build_fit_response,
    fit_tabular_optimizer,
)
from bochan.serving.webapp.routers.capabilities import WEB_CAPABILITIES
from bochan.serving.webapp.schemas.regression import RegressionRunRequest
from bochan.serving.webapp.workflows import run_regression_web_workflow
from bochan.serving.webapp.workflows.tabular import _resolve_crabnet_web_model
from bochan.serving.workbench.datasets import DatasetStore, build_dataset_record


class _FakeTransformerEncoder(nn.Module):
    def __init__(self, width: int = 6, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(num_layers))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = torch.tanh(layer(values))
        return values


class _FakeCrabNet(nn.Module):
    def __init__(self, width: int = 6) -> None:
        super().__init__()
        self.d_model = width
        self.embedding = nn.Embedding(119, width)
        self.transformer_encoder = _FakeTransformerEncoder(width)
        self.double()

    def forward(
        self,
        element_ids: torch.Tensor,
        fractions: torch.Tensor,
    ) -> torch.Tensor:
        values = self.embedding(element_ids) * fractions.unsqueeze(-1)
        return self.transformer_encoder(values)


class _HTTPFakeCrabNet(nn.Module):
    """Small deterministic replacement for the upstream JSON-created encoder."""

    def __init__(self, *, d_model: int, **_: object) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = nn.Parameter(torch.ones(d_model))
        self.transformer_encoder = _FakeTransformerEncoder(d_model, num_layers=1)

    def forward(
        self,
        element_ids: torch.Tensor,
        fractions: torch.Tensor,
    ) -> torch.Tensor:
        del element_ids
        values = fractions.unsqueeze(-1) * self.scale
        return self.transformer_encoder(values)


def _records() -> list[dict[str, object]]:
    return [
        {
            "formula": formula,
            "temperature": temperature,
            "holding_time": holding_time,
            "property": value,
        }
        for formula, temperature, holding_time, value in zip(
            [
                "Ba0.45Sr0.15Ti0.10O0.30",
                "Ba0.35Sr0.20Ti0.15O0.30",
                "Ba0.25Sr0.30Ti0.20O0.25",
                "Ba0.20Sr0.25Ti0.25O0.30",
                "Ba0.30Sr0.15Ti0.25O0.30",
                "Ba0.40Sr0.10Ti0.20O0.30",
                "Ba0.15Sr0.35Ti0.20O0.30",
                "Ba0.30Sr0.25Ti0.15O0.30",
            ],
            [1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0],
            [1.5, 2.0, 3.0, 4.0, 5.5, 6.5, 8.0, 9.0],
            [0.4, 0.7, 1.1, 1.4, 1.8, 2.2, 2.5, 1.9],
            strict=True,
        )
    ]


def _composition_site() -> dict[str, object]:
    return {
        "column": "formula",
        "elements": ["Ba", "Sr", "Ti", "O"],
        "representation": "ilr",
        "coordinate_bounds": [-3.0, 3.0],
        "bounds": {
            "Ba": [0.05, 0.70],
            "Sr": [0.05, 0.70],
            "Ti": [0.05, 0.70],
            "O": [0.05, 0.80],
        },
    }


def _fastapi_payload(model_type: str) -> dict[str, object]:
    return {
        "data": _records(),
        "input_cols": ["formula", "temperature", "holding_time"],
        "target_cols": ["property"],
        "bounds": {
            "temperature": [950.0, 1400.0],
            "holding_time": [1.0, 10.0],
        },
        "composition_sites": {"formula": _composition_site()},
        "model_config": {
            "task_type": "regression",
            "model_type": model_type,
            "model_kwargs": {
                "encoder": _FakeCrabNet(),
                "latent_dim": 4,
            },
        },
        "fit_config": {"skip_fit": True},
    }


def test_fastapi_schema_defaults_crabnet_dkl_to_partial_training() -> None:
    request = TabularFitModelRequest.model_validate(_fastapi_payload("crabnet_dkl"))

    assert request.bo_model_config.model_kwargs["encoder_training"] == "partial"
    assert request.composition_constraint_rerank is True
    assert request.composition_constraint_rerank_factor == 4


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"composition_sites": None}, "exactly one composition site"),
        ({"categorical_cols": ["holding_time"]}, "continuous process columns"),
        ({"target_cols": ["property", "second"]}, "one target column"),
    ],
)
def test_fastapi_schema_rejects_unsupported_crabnet_combinations(
    update: dict[str, object],
    match: str,
) -> None:
    payload = _fastapi_payload("crabnet_gp")
    payload.update(update)

    with pytest.raises(ValueError, match=match):
        TabularFitModelRequest.model_validate(payload)


def test_fastapi_schema_rejects_low_level_or_invalid_encoder_controls() -> None:
    payload = _fastapi_payload("crabnet_dkl")
    model_config = dict(payload["model_config"])
    model_kwargs = dict(model_config["model_kwargs"])
    model_kwargs["trainable_encoder_layers"] = 2
    model_config["model_kwargs"] = model_kwargs
    payload["model_config"] = model_config

    with pytest.raises(ValueError, match="Python API"):
        TabularFitModelRequest.model_validate(payload)

    payload = _fastapi_payload("crabnet_dkl")
    model_config = dict(payload["model_config"])
    model_kwargs = dict(model_config["model_kwargs"])
    model_kwargs["encoder_training"] = "frozen"
    model_config["model_kwargs"] = model_kwargs
    payload["model_config"] = model_config

    with pytest.raises(ValueError, match="partial.*full"):
        TabularFitModelRequest.model_validate(payload)


def test_fastapi_service_fits_crabnet_and_reports_public_metadata() -> None:
    request = TabularFitModelRequest.model_validate(_fastapi_payload("crabnet_gp"))
    optimizer = fit_tabular_optimizer(request)
    response = build_fit_response("crabnet-model", optimizer)

    assert response.model_type == "crabnet_gp"
    assert response.metadata["crabnet"] == {
        "encoder_training": "frozen",
        "encoder_initialization": "injected",
        "checkpoint_configured": False,
        "composition_site": "formula",
        "composition_column": "formula",
        "process_dim": 2,
    }


def test_fastapi_endpoint_returns_formula_and_process_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bochan.composition.encoders import crabnet as crabnet_encoder

    monkeypatch.setattr(
        crabnet_encoder,
        "_upstream_encoder_class",
        lambda: _HTTPFakeCrabNet,
    )
    payload = _fastapi_payload("crabnet_gp")
    model_config = dict(payload["model_config"])
    model_kwargs = dict(model_config["model_kwargs"])
    model_kwargs.pop("encoder")
    model_config["model_kwargs"] = model_kwargs
    payload["model_config"] = model_config

    client = TestClient(create_app(title="CrabNet FastAPI endpoint test"))
    fit_response = client.post("/api/v1/tabular/models", json=payload)

    assert fit_response.status_code == 200, fit_response.text
    fit_body = fit_response.json()
    assert fit_body["metadata"]["crabnet"]["encoder_initialization"] == "random"

    candidate_response = client.post(
        f"/api/v1/tabular/models/{fit_body['model_id']}/candidates",
        json={
            "acquisition_config": {"name": "ei"},
            "optimize_config": {
                "q": 1,
                "optimizer": "optimize_acqf",
                "num_restarts": 1,
                "raw_samples": 8,
                "sequential": True,
                "optimizer_kwargs": {"options": {"maxiter": 5}},
            },
        },
    )

    assert candidate_response.status_code == 200, candidate_response.text
    candidate = candidate_response.json()["candidates"][0]
    assert {"formula", "temperature", "holding_time"}.issubset(candidate)
    assert isinstance(candidate["formula"], str)


def test_web_crabnet_validation_defaults_dkl_and_rejects_unsupported_inputs() -> None:
    common = {
        "target_columns": ["property"],
        "internal_tasks": ["regression"],
        "feature_columns": ["formula", "temperature"],
        "encoded_features": {
            "cat_dims": [],
            "feature_columns": ["formula__ilr__1", "temperature"],
        },
        "composition_config": {"column": "formula"},
        "input_perturbation": False,
    }
    kwargs, metadata = _resolve_crabnet_web_model(
        model_type="crabnet_dkl",
        model_kwargs={},
        **common,
    )

    assert kwargs["encoder_training"] == "partial"
    assert metadata is not None
    assert metadata["encoder_training"] == "partial"

    with pytest.raises(ValueError, match="composition formula"):
        _resolve_crabnet_web_model(
            model_type="crabnet_gp",
            model_kwargs={},
            **{**common, "composition_config": None},
        )
    with pytest.raises(ValueError, match="categorical process"):
        _resolve_crabnet_web_model(
            model_type="crabnet_gp",
            model_kwargs={},
            **{
                **common,
                "encoded_features": {
                    "cat_dims": [1],
                    "feature_columns": ["formula__ilr__1", "furnace"],
                },
            },
        )
    with pytest.raises(ValueError, match="input perturbation"):
        _resolve_crabnet_web_model(
            model_type="crabnet_gp",
            model_kwargs={},
            **{**common, "input_perturbation": True},
        )
    with pytest.raises(ValueError, match="always freezes"):
        _resolve_crabnet_web_model(
            model_type="crabnet_gp",
            model_kwargs={"encoder_training": "partial"},
            **common,
        )


def _web_request(
    dataset_id: str,
    model_type: str,
    *,
    encoder_training: str | None = None,
) -> RegressionRunRequest:
    model_kwargs: dict[str, object] = {
        "encoder": _FakeCrabNet(),
        "latent_dim": 4,
        "web_target_settings": [
            {
                "target": "property",
                "task_type": "regression",
                "optimize": True,
                "direction": "maximize",
                "goal": "none",
                "value": None,
            }
        ],
        "web_composition": {
            "enabled": True,
            **_composition_site(),
            "normalization": "atomic_fraction",
            "precision": 6,
            "total": 1.0,
            "min_components": 1,
            "max_components": 4,
            "required_components": [],
            "steps": {},
            "element_constraints": [],
        },
    }
    if encoder_training is not None:
        model_kwargs["encoder_training"] = encoder_training
    return RegressionRunRequest(
        dataset_id=dataset_id,
        feature_columns=["formula", "temperature", "holding_time"],
        target_column="property",
        target_columns=["property"],
        model_type=model_type,
        model_kwargs=model_kwargs,
        fit_maxiter=1,
        normalize=True,
        search_space=[
            {"name": "formula", "type": "auto"},
            {
                "name": "temperature",
                "type": "numeric",
                "lower": 950.0,
                "upper": 1400.0,
            },
            {
                "name": "holding_time",
                "type": "numeric",
                "lower": 1.0,
                "upper": 10.0,
            },
        ],
        acquisition={
            "name": "EI",
            "acqf_kwargs": {"web_family": "bayesian_optimization"},
        },
        optimizer={
            "name": "optimize_acqf",
            "q": 1,
            "num_restarts": 1,
            "raw_samples": 8,
            "sequential": True,
            "minimum_candidate_distance_ratio": 0.0,
        },
    )


@pytest.mark.parametrize(
    ("model_type", "encoder_training"),
    [("crabnet_gp", None), ("crabnet_dkl", "partial")],
)
def test_web_crabnet_returns_formula_and_process_candidates_end_to_end(
    model_type: str,
    encoder_training: str | None,
) -> None:
    torch.manual_seed(0)
    frame = pd.DataFrame.from_records(_records())
    record = build_dataset_record(
        data=frame,
        name="crabnet.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)

    result = run_regression_web_workflow(
        _web_request(
            record.dataset_id,
            model_type,
            encoder_training=encoder_training,
        ),
        store,
    )

    assert result["model_type"] == model_type
    assert result["metadata"]["hybrid_model"] is False
    assert result["metadata"]["crabnet"]["encoder_training"] == ("frozen" if model_type == "crabnet_gp" else "partial")
    assert len(result["candidates"]) == 1
    values = result["candidates"][0]["values"]
    assert {"formula", "temperature", "holding_time"}.issubset(values)
    assert isinstance(values["formula"], str)


def test_web_capabilities_publish_only_canonical_crabnet_selectors() -> None:
    assert "crabnet_gp" in WEB_CAPABILITIES["model_types"]
    assert "crabnet_dkl" in WEB_CAPABILITIES["model_types"]
    assert WEB_CAPABILITIES["crabnet"]["default_encoder_training"] == "partial"
    assert "deep_kernel" not in WEB_CAPABILITIES["model_types"]


def test_react_selector_and_controls_use_canonical_crabnet_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    model_options = (root / "web/src/modelOptions.ts").read_text(encoding="utf-8")
    likelihoods = (root / "web/src/regressionLikelihoodOptions.ts").read_text(encoding="utf-8")
    settings = (root / "web/src/components/CrabNetModelSettings.tsx").read_text(encoding="utf-8")
    api = (root / "web/src/api.ts").read_text(encoding="utf-8")

    assert '{ value: "deepkernel", label: "Deep Kernel GP"' in model_options
    assert '{ value: "crabnet_gp", label: "CrabNet-GP"' in model_options
    assert '{ value: "crabnet_dkl", label: "CrabNet-DKL"' in model_options
    assert 'deepkernel: "Deep Kernel GP"' in likelihoods
    assert '<option value="partial">Partial（推奨）</option>' in settings
    assert '<option value="full">Full</option>' in settings
    assert "modelKwargs.checkpoint = checkpoint" in api
    assert "modelKwargs.encoder_training = input.crabnetEncoderTraining" in api
    assert "markFormulaLikeColumns(imported.dataset, compositionColumn)" in api
    assert "deep_kernel" not in model_options
    assert "window.fetch =" not in api
