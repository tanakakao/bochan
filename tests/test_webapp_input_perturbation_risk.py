from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.api import ObjectiveConfig
from bochan.serving.webapp.risk_settings import (
    apply_web_risk_to_objective_config,
    attach_web_risk_metadata,
    current_web_risk_report,
    normalize_web_prediction_rows,
    resolve_web_risk_settings,
    web_risk_run,
)
from bochan.serving.webapp.workflows_tabular import _acquisition_family


def _request(
    *,
    risk_type: str = "cvar",
    alpha: float = 0.25,
    family: str = "bayesian_optimization",
    perturbation: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_perturbation=perturbation,
        acquisition=SimpleNamespace(
            acqf_kwargs={
                "web_family": family,
                "web_risk_type": risk_type,
                "web_risk_alpha": alpha,
            }
        ),
    )


def test_web_risk_context_normalizes_request_settings() -> None:
    with web_risk_run(_request(risk_type="var", alpha=0.4)) as report:
        assert report == {
            "input_perturbation": True,
            "risk_type": "var",
            "risk_alpha": 0.4,
            "risk_enabled": True,
            "acquisition_family": "bayesian_optimization",
        }
        assert current_web_risk_report() == report

    assert current_web_risk_report() == {}


def test_web_risk_allows_level_set_estimation() -> None:
    report = resolve_web_risk_settings(
        _request(family="level_set_estimation", risk_type="cvar")
    )
    assert report["risk_enabled"] is True
    assert report["acquisition_family"] == "level_set_estimation"


def test_web_risk_requires_input_perturbation() -> None:
    with pytest.raises(ValueError, match="requires input_perturbation"):
        resolve_web_risk_settings(_request(perturbation=False))


def test_web_risk_rejects_active_learning() -> None:
    with pytest.raises(ValueError, match="Bayesian optimization or level-set estimation"):
        resolve_web_risk_settings(_request(family="active_learning"))


def test_web_risk_keys_are_removed_before_acquisition_construction() -> None:
    kwargs = {
        "web_family": "level_set_estimation",
        "web_risk_type": "cvar",
        "web_risk_alpha": 0.25,
    }

    family = _acquisition_family(kwargs)

    assert family == "level_set_estimation"
    assert kwargs == {}


def test_web_risk_is_applied_to_objective_config_without_engine_patch() -> None:
    config = ObjectiveConfig(mode="scalar", output=0)
    report = resolve_web_risk_settings(_request(risk_type="cvar", alpha=0.25))

    resolved = apply_web_risk_to_objective_config(config, report)

    assert resolved is not config
    assert resolved.risk_type == "cvar"
    assert resolved.alpha == 0.25


def test_workflow_baseline_uses_same_cvar_aggregation() -> None:
    values = torch.tensor([[1.0], [2.0], [8.0], [9.0]], dtype=torch.double)
    report = resolve_web_risk_settings(_request(risk_type="cvar", alpha=0.5))

    actual = normalize_web_prediction_rows(values, n_rows=1, report=report)

    torch.testing.assert_close(
        actual,
        torch.tensor([[1.5]], dtype=torch.double),
    )


def test_web_risk_metadata_is_serialized() -> None:
    result = {"metadata": {"existing": True}}

    metadata = attach_web_risk_metadata(
        result,
        {
            "risk_type": "var",
            "risk_alpha": 0.2,
            "risk_enabled": True,
        },
    )

    assert metadata["existing"] is True
    assert metadata["input_perturbation_risk_type"] == "var"
    assert metadata["input_perturbation_risk_alpha"] == 0.2
    assert metadata["input_perturbation_risk_enabled"] is True
