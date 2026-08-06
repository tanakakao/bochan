from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from bochan.serving.webapp.visualization_sessions import (
    VisualizationSession,
    build_visualization,
    discard_visualization_run,
    register_visualization_session,
)
from bochan.visualization import show_1dplot_from_optimizer


class _Posterior:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self.mean = mean
        self.variance = variance


class _HeteroscedasticModel:
    def posterior(
        self,
        X: torch.Tensor,
        *,
        observation_noise: bool = False,
        **_: object,
    ) -> _Posterior:
        x = X[..., :1]
        mean = 2.0 * x
        epistemic_variance = torch.full_like(mean, 0.04)
        aleatoric_variance = 0.01 + 0.09 * x.square()
        variance = epistemic_variance
        if observation_noise:
            variance = variance + aleatoric_variance
        return _Posterior(mean, variance)

    def predict_noise_var(
        self,
        X: torch.Tensor,
        ref_like: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = 0.01 + 0.09 * X[..., :1].square()
        return noise if ref_like is None else noise.expand_as(ref_like)


class _Optimizer:
    def __init__(self) -> None:
        self.train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
        self.train_Y = 2.0 * self.train_X
        self.bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        self.model = _HeteroscedasticModel()
        self.bundle = SimpleNamespace(task_type="regression", metadata={})
        self.model_config = SimpleNamespace(
            task_type="regression",
            input_transform_config=None,
        )
        self.history: list[object] = []

    def predict(
        self,
        X: torch.Tensor,
        *,
        return_type: str = "posterior",
        posterior_kwargs: dict[str, object] | None = None,
    ) -> object:
        posterior = self.model.posterior(X, **dict(posterior_kwargs or {}))
        if return_type == "mean_variance":
            return posterior.mean, posterior.variance
        if return_type == "posterior":
            return posterior
        raise ValueError(return_type)


def _trace_names(figure: object) -> set[str]:
    return {
        str(trace.name)
        for trace in figure.data
        if getattr(trace, "name", None) is not None
    }


def test_heteroscedastic_1d_plot_shows_all_uncertainty_components() -> None:
    optimizer = _Optimizer()

    figure = show_1dplot_from_optimizer(
        optimizer,
        "x",
        "y",
        feature_cols=["x"],
        target_cols=["y"],
        n=9,
    )

    names = _trace_names(figure)
    assert "モデル不確実性 ±1σ" in names
    assert "観測ノイズ ±1σ" in names
    assert "総予測誤差 ±1σ（モデル + 観測ノイズ）" in names

    mean_trace = next(trace for trace in figure.data if trace.name == "y 予測平均")
    uncertainty = np.asarray(mean_trace.customdata, dtype=float)
    epistemic = uncertainty[:, 0]
    aleatoric = uncertainty[:, 1]
    total = uncertainty[:, 2]
    np.testing.assert_allclose(
        np.square(total),
        np.square(epistemic) + np.square(aleatoric),
        rtol=1e-10,
        atol=1e-12,
    )
    assert aleatoric[-1] > aleatoric[0]


def test_web_1d_visualization_preserves_heteroscedastic_traces() -> None:
    optimizer = _Optimizer()
    data = pd.DataFrame({"x": [0.0, 0.5, 1.0], "y": [0.0, 1.0, 2.0]})
    run_id = f"hetero-test-{uuid4().hex}"
    session = VisualizationSession(
        optimizer=optimizer,
        tabular_optimizer=SimpleNamespace(
            dataset=SimpleNamespace(cat_dims=[]),
        ),
        data=data,
        encoded_targets=data[["y"]].copy(),
        feature_columns=["x"],
        target_columns=["y"],
        target_metadata={"y": {"internal_task": "regression"}},
        hybrid_model=False,
        request_details={"requested_model_type": "hetero"},
    )
    register_visualization_session(run_id, session)

    try:
        payload = build_visualization(
            run_id,
            {
                "kind": "1d",
                "target": "y",
                "features": ["x"],
                "fixed_values": {},
                "n": 9,
            },
        )
    finally:
        discard_visualization_run(run_id)

    names = {
        str(trace.get("name"))
        for trace in payload["figure"]["data"]
        if trace.get("name") is not None
    }
    assert "モデル不確実性 ±1σ" in names
    assert "観測ノイズ ±1σ" in names
    assert "総予測誤差 ±1σ（モデル + 観測ノイズ）" in names
