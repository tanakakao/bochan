from __future__ import annotations

import torch

from bochan.optim import optimize_thompson_sampling
from bochan.optim.thompson.adapter import _resolve_outcome_constraints


class _ScalarObjective:
    _is_mo = True

    def forward(self, samples: torch.Tensor, X=None) -> torch.Tensor:
        del X
        return samples[..., 0]


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


class _AcquisitionWithInternalConstraintRegistry:
    def __init__(self) -> None:
        self.model = _DeterministicModel()
        self.objective = _ScalarObjective()
        self.posterior_transform = None
        self._constraints = {"raw_lengthscale": object()}

    def constraints(self):
        """Framework API method, not posterior-sample constraints."""
        return self._constraints


class _AcquisitionWithOutcomeConstraints:
    def __init__(self) -> None:
        self.outcome_constraints = [lambda Y: 0.5 - Y[..., 0]]
        self._constraints = {"raw_lengthscale": object()}


def test_internal_constraint_dictionary_is_ignored() -> None:
    acquisition = _AcquisitionWithInternalConstraintRegistry()

    constraints = _resolve_outcome_constraints(acquisition)

    assert constraints == []


def test_explicit_outcome_constraint_sequence_wins_over_internal_dictionary() -> None:
    acquisition = _AcquisitionWithOutcomeConstraints()

    constraints = _resolve_outcome_constraints(acquisition)

    assert len(constraints) == 1
    assert callable(constraints[0])


def test_public_optimizer_ignores_internal_constraint_dictionary() -> None:
    acquisition = _AcquisitionWithInternalConstraintRegistry()
    candidate_set = torch.tensor(
        [[0.1], [0.9], [0.4], [0.8]],
        dtype=torch.double,
    )

    candidates, values = optimize_thompson_sampling(
        acq_function=acquisition,
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
