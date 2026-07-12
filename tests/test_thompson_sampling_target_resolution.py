from __future__ import annotations

from dataclasses import dataclass

import torch

from bochan.api.optimizer_api import (
    OptimizeConfig,
    _resolve_thompson_sampling_target,
    optimize_candidates,
)


class _PosteriorModel:
    def posterior(self, X, **kwargs):
        return X


class _LatentModel:
    pass


@dataclass
class _Acquisition:
    model: object
    objective: object | None = None


class _CallableAcquisition(_Acquisition):
    def __call__(self, X):
        return X


class _PosteriorAcquisition(_Acquisition):
    def posterior(self, X, **kwargs):
        return X


class _DeterministicPosterior:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean

    def rsample(self, sample_shape: torch.Size) -> torch.Tensor:
        return self.mean.expand(*sample_shape, *self.mean.shape)


class _MulticlassProbabilityModel:
    """Return posterior samples shaped ``sample x N x tasks x classes``."""

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


class _ActiveLearningAcquisition:
    """Mimic BALD/entropy acquisitions that already return scalar scores."""

    objective = None
    posterior_transform = None
    constraints = None

    def __init__(self, model: object) -> None:
        self.model = model

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X[..., 0]


def test_resolve_thompson_sampling_target_prefers_configured_public_model() -> None:
    latent_model = _LatentModel()
    public_model = _PosteriorModel()
    acqf = _Acquisition(model=latent_model)

    object.__setattr__(acqf, "_bochan_thompson_model", public_model)

    assert _resolve_thompson_sampling_target(acqf) is public_model


def test_resolve_thompson_sampling_target_uses_model_with_posterior() -> None:
    public_model = _PosteriorModel()
    acqf = _Acquisition(model=public_model)

    assert _resolve_thompson_sampling_target(acqf) is public_model


def test_resolve_thompson_sampling_target_preserves_callable_acquisition() -> None:
    public_model = _PosteriorModel()
    acqf = _CallableAcquisition(model=public_model)

    assert _resolve_thompson_sampling_target(acqf) is acqf


def test_resolve_thompson_sampling_target_preserves_acquisition_objective() -> None:
    public_model = _PosteriorModel()
    objective = object()
    acqf = _Acquisition(model=public_model, objective=objective)

    assert _resolve_thompson_sampling_target(acqf) is acqf


def test_resolve_thompson_sampling_target_uses_acquisition_with_posterior() -> None:
    latent_model = _LatentModel()
    acqf = _PosteriorAcquisition(model=latent_model)

    assert _resolve_thompson_sampling_target(acqf) is acqf


def test_resolve_thompson_sampling_target_keeps_nonposterior_acquisition() -> None:
    latent_model = _LatentModel()
    acqf = _Acquisition(model=latent_model)

    assert _resolve_thompson_sampling_target(acqf) is acqf


def test_active_learning_thompson_backend_scores_acquisition_not_raw_classes() -> None:
    model = _MulticlassProbabilityModel()
    acquisition = _ActiveLearningAcquisition(model)
    candidate_set = torch.tensor(
        [[0.1], [0.9], [0.4], [0.8]],
        dtype=torch.double,
    )

    candidates, values = optimize_candidates(
        acqf=acquisition,
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
    assert values.shape == torch.Size([2])
    torch.testing.assert_close(
        candidates.sort(dim=0).values,
        torch.tensor([[0.8], [0.9]], dtype=torch.double),
    )
    torch.testing.assert_close(
        values.sort().values,
        torch.tensor([0.8, 0.9], dtype=torch.double),
    )
