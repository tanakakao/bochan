from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.acquisition.objective import RegressionScalarObjective
from bochan.acquisition.regression.levelset_estimation import qRegressionStraddle


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
    X = torch.tensor(
        [
            [[0.1], [0.2], [0.3]],
            [[0.2], [0.4], [0.6]],
            [[0.0], [0.5], [1.0]],
            [[0.3], [0.3], [0.3]],
        ],
        dtype=torch.double,
    )

    value = acquisition(X)

    expected = (0.5 - X.squeeze(-1).abs()).mean(dim=-1)
    assert value.shape == torch.Size([4])
    assert torch.allclose(value, expected)
