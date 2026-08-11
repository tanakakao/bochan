from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.acquisition.multiclass.bayesian_optimization import (
    MulticlassTargetProbabilityObjective,
)
from bochan.api.optimizer_api import OptimizeConfig, optimize_candidates
from bochan.optim import optimize_thompson_sampling
from bochan.optim.thompson.adapter import ThompsonScalarizedObjective


class _ScalarMarkedMultiObjective:
    """Mimic an objective whose flag and actual output rank disagree."""

    _is_mo = True

    def forward(self, samples: torch.Tensor, X=None) -> torch.Tensor:
        del X
        return samples[..., 0]


class _IdentityMultiObjective:
    _is_mo = True

    def forward(self, samples: torch.Tensor, X=None) -> torch.Tensor:
        del X
        return samples


class _DeterministicPosterior:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean

    def rsample(self, sample_shape: torch.Size) -> torch.Tensor:
        return self.mean.expand(*sample_shape, *self.mean.shape)


class _DeterministicModel:
    def __init__(self) -> None:
        self.train_inputs = (torch.zeros(2, 1, dtype=torch.double),)

    def eval(self):
        return self

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise=False,
        posterior_transform=None,
    ) -> _DeterministicPosterior:
        del observation_noise, posterior_transform
        return _DeterministicPosterior(X[..., :1])


class _MulticlassProbabilityModel:
    """Return two-task, three-class probabilities increasing with x."""

    def __init__(self) -> None:
        self.train_inputs = (torch.zeros(2, 1, dtype=torch.double),)

    def eval(self):
        return self

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise=False,
        posterior_transform=None,
    ) -> _DeterministicPosterior:
        del observation_noise, posterior_transform
        x = X[..., 0].clamp(0.0, 1.0)
        zeros = torch.zeros_like(x)
        task_0 = torch.stack([1.0 - x, x, zeros], dim=-1)
        task_1 = torch.stack([1.0 - x, zeros, x], dim=-1)
        return _DeterministicPosterior(torch.stack([task_0, task_1], dim=-2))


class _InputPerturbationLikeModel:
    def __init__(self, n_w: int = 3) -> None:
        self.n_w = int(n_w)
        self.train_inputs = (torch.zeros(2, 1, dtype=torch.double),)
        self.offsets = torch.linspace(0.0, 0.02, self.n_w, dtype=torch.double)

    def eval(self):
        return self

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise=False,
        posterior_transform=None,
    ) -> _DeterministicPosterior:
        del observation_noise, posterior_transform
        base = X[..., :1]
        offsets = self.offsets.to(dtype=X.dtype, device=X.device).reshape(
            *((1,) * (base.ndim - 2)),
            self.n_w,
            1,
        )
        mean = (base.unsqueeze(-2) + offsets).reshape(
            *base.shape[:-2],
            base.shape[-2] * self.n_w,
            1,
        )
        return _DeterministicPosterior(mean)


def test_scalar_multiobjective_flag_does_not_break_q_batch_validation() -> None:
    X = torch.zeros(1024, 3, dtype=torch.double)
    samples = torch.rand(2, 1024, 1, dtype=torch.double)
    objective = ThompsonScalarizedObjective(
        objective=_ScalarMarkedMultiObjective(),
    )

    scores = objective(samples, X=X)

    assert scores.shape == torch.Size([2, 1024])
    torch.testing.assert_close(scores, samples[..., 0])


def test_multiclass_probability_objective_reduces_class_axis() -> None:
    X = torch.zeros(1024, 3, dtype=torch.double)
    logits = torch.randn(2, 1024, 2, 3, dtype=torch.double)
    samples = torch.softmax(logits, dim=-1)
    objective = ThompsonScalarizedObjective(
        objective=MulticlassTargetProbabilityObjective(num_outputs=2),
    )

    scores = objective(samples, X=X)

    assert scores.shape == torch.Size([2, 1024])
    assert torch.isfinite(scores).all()


def test_expanded_input_perturbation_samples_are_averaged_to_original_pool() -> None:
    X = torch.zeros(4, 1, dtype=torch.double)
    samples = torch.tensor(
        [[[1.0], [3.0], [10.0], [14.0], [2.0], [4.0], [20.0], [24.0]]],
        dtype=torch.double,
    )
    objective = ThompsonScalarizedObjective()

    scores = objective(samples, X=X)

    expected = torch.tensor([[2.0, 12.0, 3.0, 22.0]], dtype=torch.double)
    assert scores.shape == torch.Size([1, 4])
    torch.testing.assert_close(scores, expected)


def test_expanded_outcome_constraints_use_worst_perturbation() -> None:
    X = torch.zeros(3, 1, dtype=torch.double)
    samples = torch.tensor(
        [[[0.8], [0.9], [0.2], [0.7], [0.1], [0.2]]],
        dtype=torch.double,
    )
    objective = ThompsonScalarizedObjective(
        constraints=[lambda Y: 0.5 - Y[..., 0]],
    )

    scores = objective(samples, X=X)

    assert scores.shape == torch.Size([1, 3])
    assert torch.isfinite(scores[0, 0])
    assert torch.isneginf(scores[0, 1])
    assert torch.isneginf(scores[0, 2])


def test_true_multioutput_values_are_scalarized_per_posterior_sample() -> None:
    torch.manual_seed(0)
    X = torch.zeros(16, 3, dtype=torch.double)
    samples = torch.rand(2, 16, 3, dtype=torch.double)
    objective = ThompsonScalarizedObjective(
        objective=_IdentityMultiObjective(),
    )

    scores = objective(samples, X=X)

    assert scores.shape == torch.Size([2, 16])
    assert torch.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_outcome_constraints_prefer_feasible_candidates() -> None:
    torch.manual_seed(0)
    X = torch.zeros(4, 1, dtype=torch.double)
    samples = torch.tensor(
        [[[0.1, 0.1], [0.6, 0.2], [0.9, 0.5], [0.7, 0.1]]],
        dtype=torch.double,
    )
    objective = ThompsonScalarizedObjective(
        objective=_IdentityMultiObjective(),
        constraints=[
            lambda Y: 0.5 - Y[..., 0],
            lambda Y: Y[..., 1] - 0.3,
        ],
    )

    scores = objective(samples, X=X)

    assert scores.shape == torch.Size([1, 4])
    assert torch.isneginf(scores[0, 0])
    assert torch.isfinite(scores[0, 1])
    assert torch.isneginf(scores[0, 2])
    assert torch.isfinite(scores[0, 3])


def test_all_infeasible_candidates_fall_back_to_minimum_violation() -> None:
    X = torch.zeros(3, 1, dtype=torch.double)
    samples = torch.tensor(
        [[[0.4], [0.2], [0.1]]],
        dtype=torch.double,
    )
    objective = ThompsonScalarizedObjective(
        objective=_ScalarMarkedMultiObjective(),
        constraints=[lambda Y: 0.5 - Y[..., 0]],
    )

    scores = objective(samples, X=X)

    expected = torch.tensor([[-0.1, -0.3, -0.4]], dtype=torch.double)
    torch.testing.assert_close(scores, expected)


def test_public_optimizer_accepts_scalar_output_from_multiobjective_class() -> None:
    model = _DeterministicModel()
    acq_function = SimpleNamespace(
        model=model,
        objective=_ScalarMarkedMultiObjective(),
        posterior_transform=None,
        constraints=None,
    )
    candidate_set = torch.tensor(
        [[0.1], [0.9], [0.4], [0.8]],
        dtype=torch.double,
    )

    candidates, values = optimize_thompson_sampling(
        acq_function=acq_function,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
        options={
            "candidate_set": candidate_set,
            "replacement": False,
        },
    )

    assert candidates.shape == torch.Size([2, 1])
    assert values.shape == torch.Size([2])
    torch.testing.assert_close(
        candidates.sort(dim=0).values,
        torch.tensor([[0.8], [0.9]], dtype=torch.double),
    )


def test_high_level_dispatch_preserves_multiclass_objective() -> None:
    torch.manual_seed(0)
    model = _MulticlassProbabilityModel()
    acq_function = SimpleNamespace(
        model=model,
        objective=MulticlassTargetProbabilityObjective(num_outputs=2),
        posterior_transform=None,
        constraints=None,
    )
    candidate_set = torch.tensor(
        [[0.1], [0.9], [0.4], [0.8]],
        dtype=torch.double,
    )

    candidates, _ = optimize_candidates(
        acqf=acq_function,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        config=OptimizeConfig(
            q=2,
            optimizer="thompson_sampling",
            optimizer_kwargs={
                "options": {
                    "candidate_set": candidate_set,
                    "replacement": False,
                }
            },
        ),
    )

    assert candidates.shape == torch.Size([2, 1])
    torch.testing.assert_close(
        candidates.sort(dim=0).values,
        torch.tensor([[0.8], [0.9]], dtype=torch.double),
    )


def test_public_optimizer_accepts_input_perturbation_expanded_posterior() -> None:
    model = _InputPerturbationLikeModel(n_w=3)
    acq_function = SimpleNamespace(
        model=model,
        objective=None,
        posterior_transform=None,
        constraints=None,
    )
    candidate_set = torch.tensor(
        [[0.1], [0.9], [0.4], [0.8]],
        dtype=torch.double,
    )

    candidates, values = optimize_thompson_sampling(
        acq_function=acq_function,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
        options={
            "candidate_set": candidate_set,
            "replacement": False,
        },
    )

    assert candidates.shape == torch.Size([2, 1])
    assert values.shape == torch.Size([2])
    torch.testing.assert_close(
        candidates.sort(dim=0).values,
        torch.tensor([[0.8], [0.9]], dtype=torch.double),
    )
