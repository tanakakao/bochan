from __future__ import annotations

import pytest
import torch
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.models.regression.beta import BetaGPModel
from bochan.models.regression.count.negative_binomial import NegativeBinomialGPModel
from bochan.models.regression.count.poisson import PoissonGPModel
from bochan.models.regression.gamma import GammaGPModel


@pytest.mark.parametrize(
    ("model_cls", "train_y"),
    [
        (
            BetaGPModel,
            torch.linspace(0.2, 0.8, 6, dtype=torch.double).unsqueeze(-1),
        ),
        (
            GammaGPModel,
            torch.linspace(1.1, 2.0, 6, dtype=torch.double).unsqueeze(-1),
        ),
        (
            PoissonGPModel,
            torch.arange(1, 7, dtype=torch.double).unsqueeze(-1),
        ),
        (
            NegativeBinomialGPModel,
            torch.arange(1, 7, dtype=torch.double).unsqueeze(-1),
        ),
    ],
    ids=["beta", "gamma", "poisson", "negative-binomial"],
)
def test_non_gaussian_posterior_uses_latent_base_sample_protocol(
    model_cls,
    train_y: torch.Tensor,
) -> None:
    train_x = torch.linspace(0.1, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    model = model_cls(train_x, train_y)
    candidate = torch.tensor([[0.3], [0.7]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=17)

    first = sampler(posterior)
    second = sampler(posterior)

    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()
    first.sum().backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
