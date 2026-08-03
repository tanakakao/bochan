from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor, nn

from bochan.acquisition.regression.active_learning import (
    qHeteroRegressionBALD,
    qHeteroRegressionPosteriorVariance,
    qHeteroRegressionPredictiveEntropy,
)
from bochan.acquisition.regression.bayesian_optimization.hetero_single_output import (
    qHeteroRegressionUpperConfidenceBound,
)
from bochan.models.regression.non_gaussian.negative_binomial.robust import (
    negative_binomial_heteroscedastic as hetero_module,
)
from bochan.models.regression.non_gaussian.negative_binomial.robust.negative_binomial_heteroscedastic import (
    HeteroscedasticNegativeBinomialGPModel,
    HeteroscedasticNegativeBinomialMixedGPModel,
)

DTYPE = torch.double


class _DummyNoiseModel(nn.Module):
    """Fitted log-variance GPの高速で微分可能な代替。"""

    def __init__(self, input_dim: int, *, like: Tensor) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False).to(like)
        with torch.no_grad():
            self.linear.weight.fill_(0.05)

    def posterior(self, X: Tensor) -> SimpleNamespace:
        return SimpleNamespace(mean=self.linear(X))


def _fake_noise_model_single(
    train_X: Tensor,
    noise_targets: Tensor,
    input_transform,
) -> _DummyNoiseModel:
    del noise_targets, input_transform
    return _DummyNoiseModel(train_X.shape[-1], like=train_X)


def _fake_noise_model_mixed(
    train_X: Tensor,
    noise_targets: Tensor,
    cat_dims,
    input_transform,
) -> _DummyNoiseModel:
    del noise_targets, cat_dims, input_transform
    return _DummyNoiseModel(train_X.shape[-1], like=train_X)


def _make_count_data(n: int = 8) -> tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(0)
    train_x = torch.rand(n, 2, dtype=DTYPE)
    mean = 1.0 + 2.0 * train_x[:, 0] + train_x[:, 1]
    train_y = mean.round().clamp_min(0)
    train_yvar = (0.2 + 0.1 * train_x[:, :1]).clamp_min(1e-4)
    return train_x, train_y, train_yvar


def _make_model(
    monkeypatch: pytest.MonkeyPatch,
) -> HeteroscedasticNegativeBinomialGPModel:
    train_x, train_y, train_yvar = _make_count_data()
    monkeypatch.setattr(
        hetero_module,
        "_fit_noise_model_single",
        _fake_noise_model_single,
    )
    model = HeteroscedasticNegativeBinomialGPModel(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        num_inducing_points=4,
    )
    model.eval()
    model.likelihood.eval()
    return model


def test_negative_binomial_heteroscedastic_default_noise_path_initializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_x, train_y, _ = _make_count_data()

    monkeypatch.setattr(
        hetero_module,
        "_fit_variational_nb_mll",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        hetero_module,
        "_estimate_nb_noise_targets",
        lambda model, train_X, train_Y, min_noise=1e-6: torch.full(
            (train_X.shape[-2], 1),
            0.25,
            device=train_X.device,
            dtype=train_X.dtype,
        ),
    )
    monkeypatch.setattr(
        hetero_module,
        "_fit_noise_model_single",
        _fake_noise_model_single,
    )

    model = HeteroscedasticNegativeBinomialGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_inducing_points=4,
        aux_num_epochs=1,
    )

    assert isinstance(model.noise_model, nn.Module)
    assert "noise_model" in dict(model.named_modules())


@pytest.mark.parametrize(
    "acquisition_class",
    [
        qHeteroRegressionPredictiveEntropy,
        qHeteroRegressionBALD,
        qHeteroRegressionPosteriorVariance,
    ],
)
def test_negative_binomial_heteroscedastic_active_learning_is_differentiable(
    acquisition_class: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_x, _, _ = _make_count_data()
    model = _make_model(monkeypatch)

    candidates = torch.rand(2, 3, 2, dtype=DTYPE, requires_grad=True)
    acquisition = acquisition_class(
        model=model,
        pending_penalty_weight=0.1,
        X_pending=train_x[:2],
    )
    value = acquisition(candidates)

    assert value.shape == torch.Size([2])
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert candidates.grad is not None
    assert torch.isfinite(candidates.grad).all()


def test_negative_binomial_heteroscedastic_qmc_sampling_is_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_model(monkeypatch)
    candidates = torch.rand(2, 3, 2, dtype=DTYPE, requires_grad=True)
    posterior = model.posterior(candidates, observation_noise=True)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=123)

    first = sampler(posterior)
    second = sampler(posterior)

    assert first.shape == torch.Size([16, 2, 3, 1])
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)
    first.sum().backward()
    assert candidates.grad is not None
    assert torch.isfinite(candidates.grad).all()


def test_negative_binomial_heteroscedastic_qucb_uses_default_qmc_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_x, _, _ = _make_count_data()
    model = _make_model(monkeypatch)
    candidates = torch.rand(2, 3, 2, dtype=DTYPE, requires_grad=True)
    acquisition = qHeteroRegressionUpperConfidenceBound(
        model=model,
        beta=1.0,
        noise_penalty=0.1,
        X_pending=train_x[:2],
    )

    value = acquisition(candidates)

    assert value.shape == torch.Size([2])
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert candidates.grad is not None
    assert torch.isfinite(candidates.grad).all()


def test_negative_binomial_mixed_heteroscedastic_registers_noise_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_x, train_y, train_yvar = _make_count_data()
    categories = torch.randint(0, 2, (train_x.shape[0], 1)).to(dtype=DTYPE)
    mixed_x = torch.cat([train_x, categories], dim=-1)

    monkeypatch.setattr(
        hetero_module,
        "_fit_noise_model_mixed",
        _fake_noise_model_mixed,
    )

    model = HeteroscedasticNegativeBinomialMixedGPModel(
        train_X=mixed_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        cat_dims=[2],
        num_inducing_points=4,
    )

    assert isinstance(model.noise_model, nn.Module)
    assert "noise_model" in dict(model.named_modules())
    posterior = model.posterior(mixed_x[:3], observation_noise=True)
    assert posterior.mean.shape == torch.Size([3, 1])
    assert posterior.variance.shape == torch.Size([3, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
