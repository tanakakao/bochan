"""Phase-8 numerically stable log-acquisition coverage for MACE."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

pytest.importorskip("mace")

from bochan.api import DataContext, resolve_acqf_cls
from bochan.models.regression.gaussian.deep import MACEMixedMultiTaskGPModel
from bochan.serving.fastapi.schemas.mace_tabular import MACETabularCandidateRequest
from bochan.serving.fastapi.services import mace_tabular as service
from bochan.tabular import TabularBayesianOptimizer
from tests.test_mace_phase7_integration import (
    _catalog,
    _material_encoder,
    _multi_output_frame,
)


def _independent_optimizer() -> TabularBayesianOptimizer:
    return TabularBayesianOptimizer(
        task_type="regression",
        model_type="mace_gp",
        input_cols=["phase", "temperature", "pressure"],
        target_cols=["strength", "conductivity"],
        structure_col="phase",
        structure_catalog=_catalog(6),
        bounds={"temperature": [850.0, 1150.0], "pressure": [0.5, 2.0]},
        model_kwargs={"encoder": _material_encoder(), "latent_dim": 3},
        fit_config={"skip_fit": True},
    ).fit(_multi_output_frame())


def _correlated_mixed_optimizer() -> TabularBayesianOptimizer:
    return TabularBayesianOptimizer(
        task_type="multi_objective",
        model_type="mace_multitask",
        input_cols=["temperature", "furnace", "phase", "pressure", "atmosphere"],
        categorical_cols=["furnace", "atmosphere"],
        target_cols=["strength", "conductivity"],
        structure_col="phase",
        structure_catalog=_catalog(6),
        bounds={"temperature": [850.0, 1150.0], "pressure": [0.5, 2.0]},
        model_kwargs={"encoder": _material_encoder(), "latent_dim": 3},
        fit_config={"skip_fit": True},
    ).fit(_multi_output_frame(mixed=True))


def _multiobjective_context(optimizer: TabularBayesianOptimizer) -> DataContext:
    assert optimizer.train_X is not None
    assert optimizer.train_Y is not None
    return DataContext(
        X_baseline=optimizer.train_X,
        Y_baseline=optimizer.train_Y,
        ref_point=optimizer.train_Y.min(dim=0).values - 0.1,
    )


def _qlognehvi_candidate(
    optimizer: TabularBayesianOptimizer,
) -> tuple[pd.DataFrame, Any]:
    return optimizer.candidate(
        acq_name="qlognehvi",
        q=1,
        objective_mode="multi_output",
        objective_outputs=["strength", "conductivity"],
        objective_directions=["maximize", "maximize"],
        data_context=_multiobjective_context(optimizer),
        structure_ids=list(_catalog(6)),
        num_restarts=1,
        raw_samples=8,
        optimizer_kwargs={"options": {"maxiter": 5, "batch_limit": 1}},
    )


def test_qlognehvi_alias_resolves_to_botorch_log_acquisition() -> None:
    kwargs = {
        "task_type": "regression",
        "model_type": "mace_gp",
        "multi_output": True,
    }
    assert (
        resolve_acqf_cls("qlognehvi", **kwargs)
        is qLogNoisyExpectedHypervolumeImprovement
    )
    assert (
        resolve_acqf_cls("lognehvi", **kwargs)
        is qLogNoisyExpectedHypervolumeImprovement
    )


def test_mace_independent_multioutput_optimizes_qlognehvi() -> None:
    torch.manual_seed(0)
    optimizer = _independent_optimizer()
    candidates, acq_value = _qlognehvi_candidate(optimizer)

    assert candidates.loc[0, "phase"] in set(_catalog(6))
    assert 850.0 <= candidates.loc[0, "temperature"] <= 1150.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_mace_correlated_multitask_mixed_optimizes_qlognehvi() -> None:
    torch.manual_seed(0)
    optimizer = _correlated_mixed_optimizer()
    bundle = optimizer.bo.bundle
    assert bundle is not None
    assert isinstance(bundle.model, MACEMixedMultiTaskGPModel)

    candidates, acq_value = _qlognehvi_candidate(optimizer)

    assert candidates.loc[0, "phase"] in set(_catalog(6))
    assert candidates.loc[0, "furnace"] in {"A", "B"}
    assert candidates.loc[0, "atmosphere"] in {"air", "N2"}
    assert 850.0 <= candidates.loc[0, "temperature"] <= 1150.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_mace_fastapi_candidate_contract_preserves_qlognehvi() -> None:
    captured: dict[str, object] = {}

    class FakeOptimizer:
        def candidate(self, **kwargs: Any):
            captured.update(kwargs)
            return (
                pd.DataFrame([{"phase": "beta", "temperature": 1015.0}]),
                -0.25,
            )

    request = MACETabularCandidateRequest.model_validate(
        {
            "acquisition_config": {"name": "qlognehvi"},
            "optimize_config": {"q": 1},
            "objective_mode": "multi_output",
            "objective_outputs": ["strength", "conductivity"],
            "objective_directions": ["maximize", "maximize"],
            "structure_ids": ["beta"],
        }
    )
    response = service.mace_candidate_response("model-1", FakeOptimizer(), request)

    acq_config = captured["acq_config"]
    assert isinstance(acq_config, dict)
    assert acq_config["name"] == "qlognehvi"
    assert captured["objective_mode"] == "multi_output"
    assert captured["objective_outputs"] == ["strength", "conductivity"]
    assert captured["objective_directions"] == ["maximize", "maximize"]
    assert captured["structure_ids"] == ["beta"]
    assert response.candidates == [{"phase": "beta", "temperature": 1015.0}]
    assert response.acq_value == pytest.approx(-0.25)
