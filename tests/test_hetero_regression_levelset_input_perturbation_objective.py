from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.acquisition.objective import RegressionScalarObjective
from bochan.acquisition.regression.levelset_estimation import qHeteroRegressionStraddle


class _RepeatInputTransform(torch.nn.Module):
    """Expand each raw candidate into ``n_w`` perturbation slots."""

    def __init__(self, n_w: int) -> None:
        super().__init__()
        self.n_w = int(n_w)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.repeat_interleave(self.n_w, dim=-2)


class _PerturbedHeteroModel(torch.nn.Module):
    """Minimal hetero posterior model with an InputPerturbation-like transform."""

    def __init__(self, n_w: int = 4) -> None:
        super().__init__()
        self.input_transform = _RepeatInputTransform(n_w)

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: bool = False,
    ) -> SimpleNamespace:
        transformed = self.input_transform(X)
        mean = transformed[..., :1]
        variance_value = 0.36 if observation_noise else 0.25
        variance = torch.full_like(mean, variance_value)
        return SimpleNamespace(mean=mean, variance=variance)


def _candidate_batch() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.1], [0.2]],
            [[0.2], [0.4]],
            [[0.0], [0.5]],
            [[0.3], [0.3]],
        ],
        dtype=torch.double,
    )


def test_hetero_straddle_preserves_batch_shape_with_perturbation_objective() -> None:
    model = _PerturbedHeteroModel(n_w=4).double()
    objective = RegressionScalarObjective(n_w=4, risk_type=None)
    acquisition = qHeteroRegressionStraddle(
        model=model,
        objective=objective,
        beta=1.0,
        threshold=0.0,
        reduction="mean",
        exclude_same_batch_duplicates=False,
        exclude_pending_duplicates=False,
        exclude_observed_duplicates=False,
    )
    X = _candidate_batch()

    value = acquisition(X)

    expected = (0.5 - X.squeeze(-1).abs()).mean(dim=-1)
    assert value.shape == torch.Size([4])
    assert torch.allclose(value, expected)


def test_hetero_joint_score_bypasses_generic_q_shape_validation() -> None:
    model = _PerturbedHeteroModel(n_w=4).double()
    objective = RegressionScalarObjective(
        n_w=4,
        risk_type=None,
        sign=-1.0,
        weight=2.0,
    )
    acquisition = qHeteroRegressionStraddle(
        model=model,
        objective=objective,
        threshold=0.0,
    )
    X = _candidate_batch()
    score = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)

    value = acquisition._apply_objective_to_score(
        score,
        X,
        name="hetero joint score test",
    )

    assert value.shape == torch.Size([4])
    assert torch.allclose(value, -2.0 * score)


def test_hetero_lse_duplicate_exclusion_uses_raw_candidates_with_perturbation() -> None:
    model = _PerturbedHeteroModel(n_w=4).double()
    objective = RegressionScalarObjective(n_w=4, risk_type=None)
    acquisition = qHeteroRegressionStraddle(
        model=model,
        objective=objective,
        beta=1.0,
        threshold=0.0,
        reduction="mean",
        exclude_pending_duplicates=False,
        exclude_observed_duplicates=False,
    )
    X = torch.tensor(
        [
            [[0.1], [0.2]],
            [[0.3], [0.3]],
        ],
        dtype=torch.double,
    )

    value = acquisition(X)

    assert value.shape == torch.Size([2])
    assert torch.isfinite(value[0])
    assert torch.isneginf(value[1])
