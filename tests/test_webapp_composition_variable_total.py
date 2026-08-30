from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bochan.serving.webapp.composition.support import (
    add_composition_candidate_rows,
    composition_model_feature_columns,
    composition_response_metadata,
    composition_site,
    normalize_web_composition_settings,
    prepare_composition_encoded_features,
)
from bochan.serving.webapp.routers.composition import create_composition_router
from bochan.tabular import TabularBayesianOptimizer


def _variable_settings(representation: str = "ilr") -> dict[str, object]:
    return {
        "column": "formula",
        "elements": ["Al", "Ti", "V", "Nb"],
        "representation": representation,
        "reference_element": "Nb" if representation == "alr" else None,
        "pseudocount": 1e-8,
        "total": None,
        "total_bounds": [0.8, 1.4],
        "bounds": {
            "Al": [0.10, 0.90],
            "Ti": [0.00, 0.90],
            "V": [0.00, 0.90],
            "Nb": [0.00, 0.90],
        },
        "min_components": 3,
        "max_components": 3,
        "required_components": ["Al"],
        "support_selection": "best_subset",
        "best_subset_strategy": "exact",
        "best_subset_max_combinations": 20,
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "formula": [
                "Al0.40Ti0.25V0.20Nb0.15",
                "Al0.42Ti0.36V0.18Nb0.24",
                "Al0.30Ti0.20V0.30Nb0.20",
                "Al0.35Ti0.35V0.28Nb0.42",
                "Al0.36Ti0.12V0.20Nb0.12",
                "Al0.32Ti0.28V0.22Nb0.18",
                "Al0.456Ti0.216V0.216Nb0.312",
                "Al0.252Ti0.198V0.252Nb0.198",
            ],
            "temperature": [850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0],
            "property": [0.8, 1.0, 1.15, 1.25, 1.35, 1.5, 1.65, 1.8],
        }
    )


def test_web_normalization_exposes_variable_total_contract() -> None:
    config = normalize_web_composition_settings(_variable_settings())

    assert config["variable_total"] is True
    assert config["total_bounds"] == (0.8, 1.4)
    assert config["total_feature"] == "formula__total"
    assert config["total"] == pytest.approx(1.1)

    site = composition_site(config)
    assert site["total_bounds"] == (0.8, 1.4)
    assert site["total_feature"] == "formula__total"
    assert "total" not in site


def test_web_normalization_rejects_both_fixed_and_variable_total() -> None:
    with pytest.raises(ValueError, match="either total or total_bounds"):
        normalize_web_composition_settings(
            {
                **_variable_settings(),
                "total": 1.0,
            }
        )


def test_web_variable_total_best_subset_rejects_steps_explicitly() -> None:
    with pytest.raises(ValueError, match="does not yet support component steps"):
        normalize_web_composition_settings(
            {
                **_variable_settings(),
                "steps": {"Al": 0.05},
            }
        )


@pytest.mark.parametrize("representation", ["fractions", "clr", "alr", "ilr"])
def test_web_variable_total_encoding_adds_total_feature(
    representation: str,
) -> None:
    config = normalize_web_composition_settings(_variable_settings(representation))
    encoded, resolved = prepare_composition_encoded_features(
        data=_frame(),
        feature_columns=["formula", "temperature"],
        search_space=[
            SimpleNamespace(
                name="temperature",
                type="numeric",
                lower=800.0,
                upper=1250.0,
                step=None,
                fixed=False,
                fixed_value=None,
            )
        ],
        config=config,
    )

    total_name = resolved["total_feature"]
    assert total_name == "formula__total"
    assert total_name in encoded["feature_columns"]
    total_index = encoded["feature_columns"].index(total_name)
    assert encoded["bounds"][0][total_index] == pytest.approx(0.8)
    assert encoded["bounds"][1][total_index] == pytest.approx(1.4)
    assert encoded["X"][0][total_index] == pytest.approx(1.0)
    assert encoded["X"][1][total_index] == pytest.approx(1.2)

    model_columns = composition_model_feature_columns(
        ["formula", "temperature"],
        resolved,
    )
    assert model_columns == [
        *resolved["feature_names"],
        "formula__total",
        "temperature",
    ]


def test_typed_composition_endpoint_transports_variable_total() -> None:
    test_app = FastAPI()

    def base_run(request):
        return request.model_dump()

    test_app.include_router(
        create_composition_router(run_regression=base_run),
        prefix="/api/v1",
    )
    response = TestClient(test_app).post(
        "/api/v1/composition/regression/run",
        json={
            "run": {
                "dataset_id": "dataset-1",
                "feature_columns": ["formula", "temperature"],
                "target_column": "property",
            },
            "composition": _variable_settings(),
        },
    )

    assert response.status_code == 200, response.text
    composition = response.json()["model_kwargs"]["web_composition"]
    assert "total" not in composition
    assert composition["total_bounds"] == [0.8, 1.4]
    assert composition["support_selection"] == "best_subset"


@pytest.mark.parametrize("representation", ["clr", "alr", "ilr"])
def test_web_variable_total_candidate_restores_exact_amount_support_and_total(
    representation: str,
) -> None:
    config = normalize_web_composition_settings(_variable_settings(representation))
    _encoded, resolved = prepare_composition_encoded_features(
        data=_frame(),
        feature_columns=["formula", "temperature"],
        search_space=[
            SimpleNamespace(
                name="temperature",
                type="numeric",
                lower=800.0,
                upper=1250.0,
                step=None,
                fixed=False,
                fixed_value=None,
            )
        ],
        config=config,
    )
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        fit_config={"maxiter": 32},
        input_cols=["formula", "temperature"],
        target_cols="property",
        composition_sites={"composition": composition_site(resolved)},
        bounds={"temperature": [800.0, 1250.0]},
    ).fit(_frame())

    result = optimizer.candidate(
        acq_name="logei",
        q=1,
        num_restarts=2,
        raw_samples=16,
        optimizer_kwargs={
            "best_subset_strategy": "exact",
            "options": {"maxiter": 12, "batch_limit": 2},
        },
        return_result=True,
    )
    raw = result.raw_composition_candidates
    bridge = result.composition_raw_bridge
    amounts = bridge.amount_values(raw)
    total = float(amounts.sum().item())
    assert 0.8 - 1e-7 <= total <= 1.4 + 1e-7
    assert int((amounts > 1e-8).sum().item()) == 3

    rows = [{"values": {}, "constraints": [], "constraints_ok": True}]
    add_composition_candidate_rows(
        rows,
        tabular_optimizer=optimizer,
        candidates=result.candidates,
        config=resolved,
        candidate_result=result,
    )
    values = rows[0]["values"]
    assert values["formula__total"] == pytest.approx(total, abs=1e-7)
    fractions = [
        float(values[f"formula__fraction__{element}"])
        for element in ["Al", "Ti", "V", "Nb"]
    ]
    assert sum(value > 1e-8 for value in fractions) == 3
    assert sum(fractions) == pytest.approx(1.0, abs=1e-7)
    assert isinstance(values["formula"], str)
    assert torch.isfinite(torch.as_tensor(result.acq_value)).all()

    metadata = composition_response_metadata(resolved)
    assert metadata is not None
    assert metadata["variable_total"] is True
    assert metadata["total"] is None
    assert metadata["total_bounds"] == [0.8, 1.4]
    assert metadata["support_space"] == "raw_amount"
