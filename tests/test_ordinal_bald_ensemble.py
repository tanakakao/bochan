from __future__ import annotations

import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import nn

from bochan.acquisition.ordinal.active_learning.single_output import qOrdinalBALD
from bochan.models.ordinal.neural.deep_ensemble import DeepEnsembleOrdinalModel
from bochan.models.ordinal.posterior import OrdinalEnsemblePosterior


class _ConstantOrdinalMember(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor(float(value), dtype=torch.double))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.value.expand(*X.shape[:-1], 1)


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0], [0.15], [0.3], [0.45], [0.6], [0.75], [0.9], [1.0], [0.85]],
        dtype=torch.double,
    )
    Y = torch.tensor([[0], [0], [0], [1], [1], [1], [2], [2], [2]], dtype=torch.long)
    return X, Y


def _deep_ensemble() -> DeepEnsembleOrdinalModel:
    train_X, train_Y = _training_data()
    model = DeepEnsembleOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        members=[
            _ConstantOrdinalMember(-1.5),
            _ConstantOrdinalMember(0.0),
            _ConstantOrdinalMember(1.5),
        ],
        ensemble_size=3,
        bootstrap=False,
    )
    model._is_fitted = True
    return model


def test_ordinal_latent_ensemble_supports_sobol_normal_sampling() -> None:
    model = _deep_ensemble()
    posterior = model.latent_posterior(torch.tensor([[[0.35]]], dtype=torch.double))

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
    samples = sampler(posterior)

    assert samples.shape == torch.Size([16, 1, 1, 1])
    assert torch.isfinite(samples).all()


def test_ordinal_bald_runs_for_deep_ensemble_latent_posterior() -> None:
    model = _deep_ensemble()

    value = qOrdinalBALD(
        model=model,
        num_samples=16,
        exclude_observed_duplicates=False,
    )(torch.tensor([[[0.35]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
    assert torch.all(value >= -1e-10)


def test_probability_space_ordinal_posterior_keeps_finite_member_sampling() -> None:
    values = torch.tensor(
        [
            [[0.7, 0.2, 0.1]],
            [[0.2, 0.6, 0.2]],
            [[0.1, 0.3, 0.6]],
        ],
        dtype=torch.double,
    )
    posterior = OrdinalEnsemblePosterior(values=values)

    samples = posterior.rsample(torch.Size([8]))

    assert samples.shape == torch.Size([8, 1, 3])
    for sample in samples:
        assert any(torch.equal(sample, member) for member in values)
