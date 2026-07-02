from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from bochan.api import InputTransformConfig, ModelConfig
from bochan.visualization import candidates_dataframe, prediction_dataframe


class _PerturbedPredictionObject:
    def __init__(self, n_w: int = 4) -> None:
        self.n_w = n_w
        self.model_config = ModelConfig(
            task_type="regression",
            model_type="base",
            input_transform_config=InputTransformConfig(
                normalize=True,
                perturbation=True,
                n_w=n_w,
            ),
            outcome_transform=False,
        )
        self.train_X = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.double,
        )
        self.train_Y = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def predict(self, X, *, return_type="mean_variance"):
        assert return_type == "mean_variance"
        n_points = int(X.shape[-2])
        base = torch.arange(
            n_points,
            dtype=X.dtype,
            device=X.device,
        ).reshape(-1, 1) * 10.0
        offsets = torch.arange(
            self.n_w,
            dtype=X.dtype,
            device=X.device,
        ).reshape(1, -1)
        mean = (base + offsets).reshape(-1, 1)
        var = torch.ones_like(mean)
        return mean, var


def test_prediction_dataframe_aggregates_input_perturbation_rows() -> None:
    obj = _PerturbedPredictionObject(n_w=4)
    X = torch.tensor([[0.2, 0.3], [0.7, 0.8]], dtype=torch.double)

    mean_df, std_df = prediction_dataframe(obj, X, target_cols=["y"])

    assert len(mean_df) == len(X)
    assert len(std_df) == len(X)
    np.testing.assert_allclose(mean_df["y"], [1.5, 11.5])
    np.testing.assert_allclose(std_df["y"], [1.5, 1.5])


def test_candidates_dataframe_keeps_one_row_per_candidate() -> None:
    obj = _PerturbedPredictionObject(n_w=4)
    candidates = torch.tensor(
        [[0.2, 0.3], [0.7, 0.8]],
        dtype=torch.double,
    )
    result = SimpleNamespace(
        candidates=candidates,
        acq_value=torch.tensor(0.25, dtype=torch.double),
    )

    df = candidates_dataframe(
        obj,
        candidate_result=result,
        feature_cols=["x0", "x1"],
        target_cols=["y"],
    )

    assert df is not None
    assert len(df) == len(candidates)
    np.testing.assert_allclose(df["y_mean"], [1.5, 11.5])
    np.testing.assert_allclose(df["y_std"], [1.5, 1.5])
    np.testing.assert_allclose(df["acq_value"], [0.25, 0.25])
