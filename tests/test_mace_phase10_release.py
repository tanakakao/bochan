"""Phase-10 release-readiness coverage for MACE integration."""

from __future__ import annotations

import io
import tomllib
from pathlib import Path

import pandas as pd
import pytest
import torch

pytest.importorskip("mace")

from bochan.model_artifact import deserialize_model_artifact, serialize_model_artifact
from bochan.models.regression.gaussian.deep import MACEDKLModel
from bochan.serving.fastapi.services.mace_tabular import _register_pending_candidates
from bochan.tabular import TabularBayesianOptimizer
from tests.test_mace_phase7_integration import (
    _catalog,
    _material_encoder,
    _single_output_frame,
    _single_output_optimizer,
)


def test_mace_ask_pending_state_uses_canonical_observations_and_is_artifact_safe() -> None:
    optimizer = _single_output_optimizer(3)
    assert optimizer.dataset is not None
    assert optimizer.bo.observations is not None
    before_dataset = int(optimizer.dataset.X.shape[0])
    before_train = int(optimizer.bo.train_X.shape[0])
    before_observations = int(optimizer.bo.observations.X.shape[0])

    candidate = pd.DataFrame(
        [
            {
                "phase": "s1",
                "temperature": 975.0,
                "pressure": 1.1,
            }
        ]
    )
    _register_pending_candidates(optimizer, candidate)

    observations = optimizer.bo.observations
    assert observations is not None
    assert int(observations.X.shape[0]) == before_observations + 1
    assert int(observations.pending_mask.sum().item()) == 1
    assert int(observations.pending_X.shape[0]) == 1
    assert torch.isnan(observations.Y[observations.pending_mask]).all()

    # Pending experiments are acquisition state, not successful training observations.
    assert int(optimizer.bo.train_X.shape[0]) == before_train
    assert int(optimizer.dataset.X.shape[0]) == before_dataset
    context = optimizer.bo._resolve_data_context()  # noqa: SLF001
    assert context.X_pending is not None
    torch.testing.assert_close(context.X_pending, observations.pending_X)

    payload = serialize_model_artifact(
        optimizer,
        backend="tabular",
        metadata={"model_family": "mace", "phase": 10},
    )
    artifact = deserialize_model_artifact(
        io.BytesIO(payload),
        trust_pickle=True,
        expected_backend="tabular",
    )
    restored = artifact["optimizer"]
    restored_observations = restored.bo.observations

    assert artifact["metadata"]["model_family"] == "mace"
    assert restored_observations is not None
    assert int(restored_observations.pending_mask.sum().item()) == 1
    torch.testing.assert_close(restored_observations.pending_X, observations.pending_X)
    assert torch.isnan(restored_observations.Y[restored_observations.pending_mask]).all()
    assert int(restored.bo.train_X.shape[0]) == before_train
    assert int(restored.dataset.X.shape[0]) == before_dataset


def test_mace_dkl_closes_candidate_optimization_path() -> None:
    torch.manual_seed(0)
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="mace_dkl",
        input_cols=["phase", "temperature", "pressure"],
        target_cols="property",
        structure_col="phase",
        structure_catalog=_catalog(4),
        bounds={"temperature": [850.0, 1150.0], "pressure": [0.5, 2.0]},
        model_kwargs={
            "encoder": _material_encoder(),
            "latent_dim": 3,
            "trainable_encoder_layers": 1,
        },
        fit_config={"skip_fit": True},
    ).fit(_single_output_frame(4))
    bundle = optimizer.bo.bundle

    assert bundle is not None
    assert isinstance(bundle.model, MACEDKLModel)
    assert bundle.model.structure_feature_cache_enabled is False

    candidates, acq_value = optimizer.candidate(
        acq_name="logei",
        q=1,
        structure_ids=list(_catalog(4)),
        num_restarts=1,
        raw_samples=8,
        optimizer_kwargs={"options": {"maxiter": 5, "batch_limit": 1}},
    )

    assert candidates.loc[0, "phase"] in set(_catalog(4))
    assert 850.0 <= candidates.loc[0, "temperature"] <= 1150.0
    assert 0.5 <= candidates.loc[0, "pressure"] <= 2.0
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_real_pretrained_mace_closes_candidate_optimization_path() -> None:
    torch.manual_seed(0)
    frame = pd.DataFrame(
        [
            {"phase": "s0", "temperature": 900.0, "property": 0.30},
            {"phase": "s1", "temperature": 960.0, "property": 0.55},
            {"phase": "s2", "temperature": 1020.0, "property": 0.80},
        ]
    )
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="mace_gp",
        input_cols=["phase", "temperature"],
        target_cols="property",
        structure_col="phase",
        structure_catalog=_catalog(3),
        bounds={"temperature": [850.0, 1100.0]},
        model_kwargs={"model_name": "medium-mpa-0", "latent_dim": 4},
        fit_config={"skip_fit": True},
    ).fit(frame)

    candidates, acq_value = optimizer.candidate(
        acq_name="logei",
        q=1,
        structure_ids=["s0", "s1", "s2"],
        num_restarts=1,
        raw_samples=4,
        optimizer_kwargs={"options": {"maxiter": 3, "batch_limit": 1}},
    )

    assert candidates.loc[0, "phase"] in {"s0", "s1", "s2"}
    assert 850.0 <= candidates.loc[0, "temperature"] <= 1100.0
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_mace_has_focused_optional_dependency_extra() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert extras["mace"] == ["mace-torch>=0.3.16,<0.4"]
    assert "mace-torch>=0.3.16,<0.4" in extras["materials"]
