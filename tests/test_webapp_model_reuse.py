from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from bochan.serving.webapp.model_reuse import (
    model_reuse_run,
    prepare_model_reuse_request,
    register_fitted_model,
    reuse_fitted_tabular_optimizer,
)
from bochan.serving.webapp.visualization_sessions import (
    VisualizationSession,
    register_visualization_session,
)


def _request(*, model_type: str = "base", acquisition: str = "EI") -> SimpleNamespace:
    return SimpleNamespace(
        dataset_id="dataset-1",
        feature_columns=["x"],
        target_column="y",
        target_columns=["y"],
        direction="maximize",
        directions={"y": "maximize"},
        model_type=model_type,
        model_kwargs={"web_target_settings": [{"target": "y", "task_type": "regression"}]},
        fit_maxiter=32,
        normalize=True,
        outcome_transform=True,
        input_perturbation=False,
        n_w=8,
        perturbation_std=0.1,
        search_space=[{"name": "x", "type": "numeric", "lower": 0.0, "upper": 1.0}],
        drop_missing=True,
        acquisition=SimpleNamespace(name=acquisition),
    )


class _FakeTabularOptimizer:
    def __init__(self) -> None:
        self.bo = SimpleNamespace(model=object())
        self.dataset = SimpleNamespace(
            X=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            Y=torch.tensor([[1.0], [2.0]], dtype=torch.double),
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            cat_dims=[],
            feature_names=["x"],
            target_names=["y"],
        )

    def candidate(self, *args, **kwargs):
        return SimpleNamespace(
            candidates=torch.tensor([[0.5]], dtype=torch.double),
            acq_value=torch.tensor([1.0], dtype=torch.double),
        )


def _register_source(run_id: str, request: SimpleNamespace) -> _FakeTabularOptimizer:
    optimizer = _FakeTabularOptimizer()
    session = VisualizationSession(
        optimizer=optimizer.bo,
        tabular_optimizer=optimizer,
        data=pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
        encoded_targets=pd.DataFrame({"y": [1.0, 2.0]}),
        feature_columns=["x"],
        target_columns=["y"],
        target_metadata={"y": {"internal_task": "regression"}},
        hybrid_model=False,
    )
    register_visualization_session(run_id, session)
    with model_reuse_run(request, None):
        register_fitted_model(run_id)
    return optimizer


def test_prepare_model_reuse_request_removes_web_only_key() -> None:
    request = _request()
    request.model_kwargs = {
        **request.model_kwargs,
        "web_reuse_model_run_id": "source-run",
    }

    cleaned, source_run_id = prepare_model_reuse_request(request)

    assert source_run_id == "source-run"
    assert "web_reuse_model_run_id" not in cleaned.model_kwargs
    assert "web_target_settings" in cleaned.model_kwargs


def test_reuse_clones_optimizer_shell_and_skips_fitting() -> None:
    request = _request()
    source = _register_source("source-run", request)
    data = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})

    with model_reuse_run(request, "source-run") as report:
        reused = reuse_fitted_tabular_optimizer(
            source_run_id="source-run",
            current_run_id="next-run",
            data=data,
            feature_columns=["x"],
            target_columns=["y"],
            target_metadata={"y": {"internal_task": "regression"}},
            hybrid_model=False,
        )

    assert reused is not source
    assert reused.bo is source.bo
    assert reused.dataset is not source.dataset
    assert torch.equal(reused.dataset.X, source.dataset.X)
    assert torch.equal(reused.dataset.Y, source.dataset.Y)
    assert reused.web_model_reused is True
    assert report["model_reused"] is True
    assert report["fit_skipped"] is True
    assert report["source_run_id"] == "source-run"


def test_reuse_rejects_changed_model_settings() -> None:
    source_request = _request()
    _register_source("mismatch-source", source_request)
    changed_request = _request(model_type="saas")

    with (
        model_reuse_run(changed_request, "mismatch-source"),
        pytest.raises(ValueError, match="cannot be reused"),
    ):
        reuse_fitted_tabular_optimizer(
            source_run_id="mismatch-source",
            current_run_id="mismatch-next",
            data=pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
            feature_columns=["x"],
            target_columns=["y"],
            target_metadata={"y": {"internal_task": "regression"}},
            hybrid_model=False,
        )


def test_acquisition_changes_do_not_change_model_fingerprint() -> None:
    source_request = _request(acquisition="EI")
    source = _register_source("acq-source", source_request)
    changed_acquisition = _request(acquisition="UCB")

    with model_reuse_run(changed_acquisition, "acq-source") as report:
        reused = reuse_fitted_tabular_optimizer(
            source_run_id="acq-source",
            current_run_id="acq-next",
            data=pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
            feature_columns=["x"],
            target_columns=["y"],
            target_metadata={"y": {"internal_task": "regression"}},
            hybrid_model=False,
        )

    assert reused.bo is source.bo
    assert report["model_reused"] is True
