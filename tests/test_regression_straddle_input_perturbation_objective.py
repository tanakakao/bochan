from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.acquisition.objective import RegressionScalarObjective
from bochan.acquisition.regression.levelset_estimation import (
    qRegressionICU,
    qRegressionStraddle,
)


class _PerturbedRegressionModel(torch.nn.Module):
    """Minimal posterior model that expands each candidate ``n_w`` times."""

    def __init__(self, n_w: int = 2) -> None:
        super().__init__()
        self.n_w = int(n_w)

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    @property
    def num_outputs(self) -> int:
        return 1

    def transform_inputs(self, X: torch.Tensor) -> torch.Tensor:
        return X.repeat_interleave(self.n_w, dim=-2)

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: bool = False,
    ) -> SimpleNamespace:
        del observation_noise
        transformed = self.transform_inputs(X)
        mean = transformed[..., :1]
        variance = torch.full_like(mean, 0.25)
        return SimpleNamespace(mean=mean, variance=variance)


def _candidate_batch() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.1], [0.2], [0.3]],
            [[0.2], [0.4], [0.6]],
            [[0.0], [0.5], [1.0]],
            [[0.3], [0.3], [0.3]],
        ],
        dtype=torch.double,
    )


def test_regression_straddle_preserves_batch_shape_with_perturbation_objective() -> None:
    model = _PerturbedRegressionModel(n_w=2).double()
    objective = RegressionScalarObjective(n_w=2, risk_type=None)
    acquisition = qRegressionStraddle(
        model=model,
        objective=objective,
        beta=1.0,
        threshold=0.0,
        reduction="mean",
    )
    X = _candidate_batch()

    value = acquisition(X)

    expected = (0.5 - X.squeeze(-1).abs()).mean(dim=-1)
    assert value.shape == torch.Size([4])
    assert torch.allclose(value, expected)


def test_regression_icu_does_not_reaggregate_joint_score_by_n_w() -> None:
    model = _PerturbedRegressionModel(n_w=2).double()
    objective = RegressionScalarObjective(n_w=2, risk_type=None)
    acquisition = qRegressionICU(
        model=model,
        objective=objective,
        threshold=0.0,
    )
    X = _candidate_batch()

    value = acquisition(X)

    expanded_mean = model.transform_inputs(X).squeeze(-1)
    contour_weight = torch.exp(-0.5 * (expanded_mean / 0.5).pow(2))
    expected = torch.log(
        1.0 + 0.25 * contour_weight + acquisition.eps
    ).sum(dim=-1)

    assert value.shape == torch.Size([4])
    assert torch.isfinite(value).all()
    assert torch.allclose(value, expected)
