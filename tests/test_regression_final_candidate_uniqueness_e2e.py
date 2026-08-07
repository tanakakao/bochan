from __future__ import annotations

import torch

from bochan.api import OptimizeConfig
from bochan.api.candidate_uniqueness import count_unique_candidate_rows
from bochan.api.optimizer_dispatch import optimize_candidates
from bochan.serving.webapp.target_results import (
    _batch_acq_value,
    _broadcast_acq_values,
)
from bochan.serving.webapp.workflows_tabular import _candidate_distance_tolerances


class _PendingAwareRegressionAcquisition(torch.nn.Module):
    """Smooth acquisition whose unconstrained q points share one optimum."""

    def __init__(self) -> None:
        super().__init__()
        self.X_pending = None

    def set_X_pending(self, X_pending=None) -> None:
        self.X_pending = X_pending

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        score = -((X - 0.5) ** 2).sum(dim=(-1, -2))
        if self.X_pending is None or self.X_pending.numel() == 0:
            return score
        pending = self.X_pending.to(device=X.device, dtype=X.dtype).reshape(-1, X.shape[-1])
        distance_squared = (X.unsqueeze(-2) - pending.view(*([1] * (X.ndim - 2)), 1, *pending.shape)).pow(2).sum(dim=-1)
        repulsion = torch.exp(-distance_squared / 0.0025).sum(dim=(-1, -2))
        return score - 10.0 * repulsion


def test_actual_optimize_acqf_q3_refills_unique_final_regression_candidates() -> None:
    torch.manual_seed(0)
    acquisition = _PendingAwareRegressionAcquisition()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def final_postprocess(X: torch.Tensor) -> torch.Tensor:
        return (torch.round(X / 0.1) * 0.1).clamp(0.0, 1.0)

    candidates, acq_value = optimize_candidates(
        acqf=acquisition,
        bounds=bounds,
        config=OptimizeConfig(
            q=3,
            num_restarts=8,
            raw_samples=128,
            sequential=False,
            duplicate_tolerances=[0.049],
            duplicate_pool_restarts=8,
            duplicate_refill_attempts=4,
            final_candidate_postprocess=final_postprocess,
            optimizer_kwargs={"options": {"maxiter": 100}},
        ),
    )

    assert candidates.shape == (3, 1)
    assert torch.allclose(candidates, final_postprocess(candidates))
    assert (
        count_unique_candidate_rows(
            candidates,
            tolerances=[0.049],
        )
        == 3
    )
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_web_minimum_distances_follow_range_step_and_category_resolution() -> None:
    encoded = {
        "bounds": [[0.0, 0.0, 0.0, 0.0], [10.0, 4.0, 2.0, 1.0]],
        "cat_dims": [2],
        "fixed_features": {3: 1.0},
        "steps": {1: 1.0},
    }

    tolerances = _candidate_distance_tolerances(
        encoded,
        relative_distance=1e-3,
    )

    assert tolerances == [0.01, 0.5, 0.0, 0.0]


def test_joint_batch_acquisition_value_is_not_repeated_per_candidate() -> None:
    joint_value = torch.tensor(0.3304, dtype=torch.double)

    assert _batch_acq_value(joint_value, 3) == 0.3304
    assert _broadcast_acq_values(joint_value, 3) == [None, None, None]

    point_values = torch.tensor([0.3, 0.2, 0.1], dtype=torch.double)
    assert _batch_acq_value(point_values, 3) is None
    assert _broadcast_acq_values(point_values, 3) == [0.3, 0.2, 0.1]


def test_candidate_config_validates_featurewise_tolerances_and_postprocess() -> None:
    config = OptimizeConfig(
        q=3,
        duplicate_tolerances=[0.01, 0.0],
        final_candidate_postprocess=lambda X: X,
    )

    assert config.duplicate_tolerances == (0.01, 0.0)
    assert callable(config.final_candidate_postprocess)
