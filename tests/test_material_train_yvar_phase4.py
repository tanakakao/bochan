from __future__ import annotations

import pandas as pd
import pytest
import torch

import bochan.api.optimizer as optimizer_module
from bochan.api import BayesianOptimizer, ModelConfig, MultiOutputConfig, ObservationData, OutputConfig
from bochan.api.observation.service import build_objective_bundle
from bochan.models.multitask.wide import WideMultiTaskGP
from bochan.tabular import TabularDataConfig
from bochan.tabular.observation.data import dataframe_to_observation_tensors


class _CaptureModel:
    def __init__(self, train_X, train_Y, train_Yvar=None, **kwargs):
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.kwargs = kwargs


class _CaptureWrapper:
    def __init__(self, submodels):
        self.models = list(submodels)


def _capture_wrapper(*, submodels, output_configs, config):
    del output_configs, config
    return _CaptureWrapper(submodels)


def _single_config() -> ModelConfig:
    return ModelConfig(
        task_type="regression",
        model_type="capture",
        model_factory=_CaptureModel,
    )


def _multi_config() -> ModelConfig:
    return ModelConfig(
        task_type="multi_objective",
        model_type="capture",
        model_factory=_CaptureModel,
        multi_output_config=MultiOutputConfig(
            output_configs=[
                OutputConfig(task_type="regression", model_type="capture"),
                OutputConfig(task_type="regression", model_type="capture"),
            ],
            output_names=["a", "b"],
            use_hybrid=False,
            wrapper_factory=_capture_wrapper,
        ),
    )


def test_observation_data_canonicalizes_unobserved_yvar() -> None:
    obs = ObservationData(
        X=torch.tensor([[0.0], [1.0], [2.0]]),
        Y=torch.tensor([[1.0, float("nan")], [2.0, 3.0], [9.0, 9.0]]),
        Yvar=torch.tensor([[0.1, 999.0], [0.2, 0.3], [999.0, 999.0]]),
        failed_mask=[False, False, False],
        pending_mask=[False, False, True],
    )
    assert obs.Yvar is not None
    assert torch.isnan(obs.Yvar[0, 1])
    assert torch.isnan(obs.Yvar[2]).all()
    torch.testing.assert_close(obs.Yvar[1], torch.tensor([0.2, 0.3]))
    assert obs.report()["known_observation_variance"] is True


def test_observation_data_requires_yvar_for_observed_cell() -> None:
    with pytest.raises(ValueError, match="strictly positive Yvar"):
        ObservationData(
            X=torch.tensor([[0.0], [1.0]]),
            Y=torch.tensor([[1.0], [2.0]]),
            Yvar=torch.tensor([[0.1], [float("nan")]]),
        )


def test_observation_append_rejects_known_noise_mode_mixing() -> None:
    known = ObservationData(
        X=torch.tensor([[0.0]]),
        Y=torch.tensor([[1.0]]),
        Yvar=torch.tensor([[0.1]]),
    )
    unknown = ObservationData(X=torch.tensor([[1.0]]), Y=torch.tensor([[2.0]]))
    with pytest.raises(ValueError, match="cannot be mixed"):
        known.append(unknown)


def test_resolve_pending_replaces_yvar() -> None:
    pending = ObservationData.from_status(
        torch.tensor([[1.0]]),
        torch.tensor([[float("nan")]]),
        Yvar=torch.tensor([[float("nan")]]),
        status=["pending"],
    )
    completed = ObservationData.from_status(
        torch.tensor([[1.0]]),
        torch.tensor([[4.0]]),
        Yvar=torch.tensor([[0.25]]),
        status=["success"],
    )
    resolved = pending.resolve_pending(completed)
    assert not bool(resolved.pending_mask.any())
    torch.testing.assert_close(resolved.Y, torch.tensor([[4.0]]))
    torch.testing.assert_close(resolved.Yvar, torch.tensor([[0.25]]))


def test_partial_split_builder_slices_yvar_per_output() -> None:
    X = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double)
    Y = torch.tensor(
        [[1.0, float("nan")], [2.0, 4.0], [float("nan"), 5.0]],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [[0.1, float("nan")], [0.2, 0.3], [float("nan"), 0.4]],
        dtype=torch.double,
    )
    bundle = build_objective_bundle(
        train_X=X,
        train_Y=Y,
        train_Yvar=Yvar,
        config=_multi_config(),
    )
    sub_bundles = bundle.metadata["sub_bundles"]
    torch.testing.assert_close(sub_bundles[0].model.train_Yvar, torch.tensor([[0.1], [0.2]], dtype=torch.double))
    torch.testing.assert_close(sub_bundles[1].model.train_Yvar, torch.tensor([[0.3], [0.4]], dtype=torch.double))


def test_observation_dataframe_allows_sparse_variance_and_excludes_columns() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y1": [1.0, 2.0, None],
            "y2": [None, 4.0, None],
            "v1": [0.1, 0.2, None],
            "v2": [None, 0.3, None],
            "status": ["success", "success", "pending"],
        }
    )
    config = TabularDataConfig(
        target_cols=["y1", "y2"],
        target_variance_cols=["v1", "v2"],
        experiment_status_col="status",
        target_missing_strategy="keep",
    )
    dataset = dataframe_to_observation_tensors(frame, config)
    assert dataset.feature_names == ["x"]
    assert dataset.Yvar is not None
    torch.testing.assert_close(dataset.Yvar[1], torch.tensor([0.2, 0.3], dtype=torch.double))
    assert torch.isnan(dataset.Yvar[0, 1])
    assert torch.isnan(dataset.Yvar[2]).all()


def test_observation_dataframe_rejects_missing_variance_for_observed_target() -> None:
    frame = pd.DataFrame(
        {"x": [0.0], "y": [1.0], "v": [float("nan")], "status": ["success"]}
    )
    config = TabularDataConfig(
        target_cols=["y"],
        target_variance_cols=["v"],
        experiment_status_col="status",
        target_missing_strategy="keep",
    )
    with pytest.raises(ValueError, match="strictly positive target variance"):
        dataframe_to_observation_tensors(frame, config)


def test_public_optimizer_observation_fit_routes_yvar(monkeypatch) -> None:
    monkeypatch.setattr(optimizer_module, "fit_model", lambda bundle, config: bundle)
    optimizer = BayesianOptimizer(model_config=_single_config())
    optimizer.fit(
        torch.tensor([[0.0], [1.0], [2.0]]),
        torch.tensor([[1.0], [2.0], [float("nan")]]),
        torch.tensor([[0.1], [0.2], [float("nan")]]),
        pending_mask=[False, False, True],
    )
    assert optimizer.observations is not None
    assert optimizer.train_Yvar is not None
    torch.testing.assert_close(optimizer.model.train_Yvar, torch.tensor([[0.1], [0.2]]))


def test_public_optimizer_pending_then_tell_with_yvar(monkeypatch) -> None:
    monkeypatch.setattr(optimizer_module, "fit_model", lambda bundle, config: bundle)
    optimizer = BayesianOptimizer(model_config=_single_config())
    optimizer.fit(
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([[1.0], [float("nan")]]),
        torch.tensor([[0.1], [float("nan")]]),
        pending_mask=[False, True],
    )
    optimizer.tell(
        torch.tensor([[1.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([[0.2]]),
        status="success",
        refit=False,
    )
    assert optimizer.observations is not None
    assert not bool(optimizer.observations.pending_mask.any())
    torch.testing.assert_close(optimizer.train_Yvar, torch.tensor([[0.1], [0.2]]))


def test_wide_multitask_gp_maps_wide_yvar_to_long_noise() -> None:
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    Y = torch.tensor(
        [[1.0, float("nan")], [2.0, 4.0], [float("nan"), 5.0]],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [[0.1, float("nan")], [0.2, 0.3], [float("nan"), 0.4]],
        dtype=torch.double,
    )
    model = WideMultiTaskGP(X, Y, train_Yvar=Yvar)
    assert model.train_Yvar_wide is not None
    torch.testing.assert_close(model.train_Yvar_wide, Yvar, equal_nan=True)
    noise = model.likelihood.noise.detach().reshape(-1)
    assert noise.numel() == 4
    assert bool(torch.isfinite(noise).all())
    assert bool((noise > 0).all())
