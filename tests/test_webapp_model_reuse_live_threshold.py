from __future__ import annotations

import pandas as pd
import pytest
import torch


def _request(
    dataset_id: str,
    threshold: float,
    *,
    task_type: str = "regression",
    regression_goal: str = "target",
    reuse_run_id: str | None = None,
):
    from bochan.serving.webapp.schemas.regression import RegressionRunRequest

    if task_type == "classification":
        target_setting: dict[str, object] = {
            "target": "y",
            "task_type": "classification",
            "optimize": True,
            "direction": "maximize",
            "goal": "above",
            "value": threshold,
            "target_class": None,
            "target_classes": [2],
            "level_set_weight": 1.0,
        }
    else:
        target_setting = {
            "target": "y",
            "task_type": "regression",
            "optimize": True,
            "direction": "maximize",
            "goal": regression_goal,
            "value": threshold,
            "level_set_weight": 1.0,
        }

    model_kwargs: dict[str, object] = {
        "web_target_settings": [target_setting]
    }
    if reuse_run_id is not None:
        model_kwargs["web_reuse_model_run_id"] = reuse_run_id

    return RegressionRunRequest(
        dataset_id=dataset_id,
        feature_columns=["x"],
        target_column="y",
        target_columns=["y"],
        directions={"y": "maximize"},
        model_type="base",
        model_kwargs=model_kwargs,
        fit_maxiter=12 if task_type == "regression" else 8,
        normalize=True,
        outcome_transform=True,
        search_space=[
            {
                "name": "x",
                "type": "numeric",
                "lower": 0.0,
                "upper": 1.0,
                "fixed": False,
            }
        ],
        acquisition={
            "name": "straddle",
            "beta": 1.96,
            "acqf_kwargs": {
                "web_family": "level_set_estimation",
                "web_level_set_parameter": 1.96,
            },
        },
        optimizer={
            "name": "normal",
            "q": 1,
            "num_restarts": 2,
            "raw_samples": 16,
            "sequential": True,
            "minimum_candidate_distance_ratio": 0.01,
        },
        feature_importance={"enabled": False},
    )


def _run_with_id(run_id: str, request, store):
    from bochan.serving.webapp.logging import reset_request_id, set_request_id
    from bochan.serving.webapp.workflows import run_regression_web_workflow

    token = set_request_id(run_id)
    try:
        return run_regression_web_workflow(request, store)
    finally:
        reset_request_id(token)


def _regression_store():
    from bochan.serving.workbench.datasets import DatasetStore, build_dataset_record

    x = torch.linspace(0.0, 1.0, 9, dtype=torch.double)
    y = 0.1 + 0.8 * x
    record = build_dataset_record(
        data=pd.DataFrame({"x": x.numpy(), "y": y.numpy()}),
        name="live-threshold-reuse.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)
    return store, record.dataset_id


def test_reused_web_lse_uses_current_target_value_in_actual_acquisition() -> None:
    """Regression target 0.6 -> reuse -> 0.3 must change the acquisition optimized."""

    from bochan.serving.webapp.services.visualization_sessions import (
        get_visualization_session,
    )

    store, dataset_id = _regression_store()
    source_run_id = "live-target-source-06"
    source_result = _run_with_id(source_run_id, _request(dataset_id, 0.6), store)
    source_session = get_visualization_session(source_run_id)
    source_acqf = source_session.candidate_result.acqf

    reused_run_id = "live-target-reuse-03"
    reused_result = _run_with_id(
        reused_run_id,
        _request(dataset_id, 0.3, reuse_run_id=source_run_id),
        store,
    )
    reused_session = get_visualization_session(reused_run_id)
    reused_acqf = reused_session.candidate_result.acqf

    assert source_result["target_settings"][0]["value"] == pytest.approx(0.6)
    assert reused_result["target_settings"][0]["value"] == pytest.approx(0.3)
    assert reused_result["metadata"]["model_reused"] is True
    assert reused_result["metadata"]["fit_skipped"] is True
    assert source_acqf.model.specs[0].eq_target == pytest.approx(0.6)
    assert reused_acqf.model.specs[0].eq_target == pytest.approx(0.3)
    assert reused_acqf.model.models[0] is source_acqf.model.models[0]
    assert reused_acqf.threshold.item() == pytest.approx(0.0)

    probe = torch.tensor([[[0.37]]], dtype=torch.double)
    posterior = reused_acqf.model.models[0].posterior(probe, observation_noise=False)
    mean = posterior.mean.squeeze(-1)
    std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
    expected = 1.96 * std - (mean - 0.3).abs()
    actual = reused_acqf(probe)
    assert torch.allclose(actual, expected.squeeze(-1), atol=1e-7, rtol=1e-5)


def test_reused_regression_lse_uses_current_boundary_threshold() -> None:
    """Regression above threshold 0.6 -> reuse -> 0.3 must update actual LSE."""

    from bochan.serving.webapp.services.visualization_sessions import (
        get_visualization_session,
    )

    store, dataset_id = _regression_store()
    source_run_id = "live-boundary-source-06"
    source_result = _run_with_id(
        source_run_id,
        _request(dataset_id, 0.6, regression_goal="above"),
        store,
    )
    source_acqf = get_visualization_session(source_run_id).candidate_result.acqf

    reused_run_id = "live-boundary-reuse-03"
    reused_result = _run_with_id(
        reused_run_id,
        _request(
            dataset_id,
            0.3,
            regression_goal="above",
            reuse_run_id=source_run_id,
        ),
        store,
    )
    reused_acqf = get_visualization_session(reused_run_id).candidate_result.acqf

    assert source_result["target_settings"][0]["value"] == pytest.approx(0.6)
    assert reused_result["target_settings"][0]["value"] == pytest.approx(0.3)
    assert reused_result["metadata"]["model_reused"] is True
    assert reused_result["metadata"]["fit_skipped"] is True
    assert source_acqf.threshold.item() == pytest.approx(0.6)
    assert reused_acqf.threshold.item() == pytest.approx(0.3)
    assert reused_acqf.model.models[0] is source_acqf.model.models[0]

    probe = torch.tensor([[[0.37]]], dtype=torch.double)
    posterior = reused_acqf.model.posterior(probe, output_mode="objective")
    mean = posterior.mean.squeeze(-1)
    std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
    expected = 1.96 * std - (mean - 0.3).abs()
    actual = reused_acqf(probe)
    assert torch.allclose(actual, expected.squeeze(-1), atol=1e-7, rtol=1e-5)


def test_reused_classification_lse_uses_current_probability_threshold() -> None:
    """Classification p=0.6 -> reuse -> p=0.3 must update the actual LSE threshold."""

    from bochan.serving.webapp.services.visualization_sessions import (
        get_visualization_session,
    )
    from bochan.serving.workbench.datasets import DatasetStore, build_dataset_record

    x = torch.linspace(0.0, 1.0, 15, dtype=torch.double)
    y = (x >= 1.0 / 3.0).long() + (x >= 2.0 / 3.0).long()
    record = build_dataset_record(
        data=pd.DataFrame({"x": x.numpy(), "y": y.numpy()}),
        name="live-classification-threshold-reuse.csv",
        source_type="csv",
    )
    store = DatasetStore()
    store.add(record)

    source_run_id = "live-classification-threshold-source-06"
    source_result = _run_with_id(
        source_run_id,
        _request(record.dataset_id, 0.6, task_type="classification"),
        store,
    )
    source_session = get_visualization_session(source_run_id)
    source_acqf = source_session.candidate_result.acqf

    reused_run_id = "live-classification-threshold-reuse-03"
    reused_result = _run_with_id(
        reused_run_id,
        _request(
            record.dataset_id,
            0.3,
            task_type="classification",
            reuse_run_id=source_run_id,
        ),
        store,
    )
    reused_session = get_visualization_session(reused_run_id)
    reused_acqf = reused_session.candidate_result.acqf

    assert source_result["target_settings"][0]["value"] == pytest.approx(0.6)
    assert reused_result["target_settings"][0]["value"] == pytest.approx(0.3)
    assert reused_result["metadata"]["model_reused"] is True
    assert reused_result["metadata"]["fit_skipped"] is True
    assert source_acqf.threshold.item() == pytest.approx(0.6)
    assert reused_acqf.threshold.item() == pytest.approx(0.3)
    assert reused_acqf.model.models[0] is source_acqf.model.models[0]

    probe = torch.tensor([[[0.57]]], dtype=torch.double)
    posterior = reused_acqf.model.posterior(probe, output_mode="objective")
    mean = posterior.mean.squeeze(-1)
    std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
    expected = 1.96 * std - (mean - 0.3).abs()
    actual = reused_acqf(probe)
    assert torch.allclose(actual, expected.squeeze(-1), atol=1e-7, rtol=1e-5)


def test_web_lse_current_threshold_overrides_stale_acquisition_kwargs() -> None:
    """Web-derived candidate settings must replace values carried by an older run."""

    from bochan.serving.webapp.settings.level_set import configure_level_set_acqf_kwargs

    train_x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    kwargs: dict[str, object] = {
        "thresholds": [0.6],
        "output_weights": [9.0],
        "output_reduction": "sum",
        "X_observed": torch.tensor([[0.5]], dtype=torch.double),
        "beta": 9.0,
        "web_level_set_parameter": 1.96,
    }
    setting = {
        "target": "y",
        "task_type": "classification",
        "optimize": True,
        "direction": "maximize",
        "goal": "above",
        "value": 0.3,
        "level_set_weight": 1.0,
    }
    metadata = {
        "y": {
            **setting,
            "internal_task": "multiclass",
            "configured_value": 0.3,
            "class_index": 2,
            "class_indices": [2],
            "num_classes": 3,
        }
    }

    configure_level_set_acqf_kwargs(
        kwargs,
        acq_key="straddle",
        train_x=train_x,
        target_columns=["y"],
        target_settings=[setting],
        target_metadata=metadata,
        objective_targets=["y"],
        input_perturbation=False,
        n_w=4,
    )

    assert list(kwargs["thresholds"]) == pytest.approx([0.3])
    assert kwargs["output_weights"] == pytest.approx([1.0])
    assert kwargs["output_reduction"] == "weighted_mean"
    assert torch.equal(kwargs["X_observed"], train_x)
    assert kwargs["beta"] == pytest.approx(1.96)
