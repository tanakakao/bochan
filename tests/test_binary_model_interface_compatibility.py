from __future__ import annotations

import torch
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator

from bochan.fit import fit_rrp_binary_classifier_mll
from bochan.models.classification.binary.base._latent_models import (
    _LatentBinarySVGP,
    _LatentMixedBinarySVGP,
)
from bochan.models.classification.binary.base.multioutput import (
    MultiOutputBinaryClassificationModel,
)
from bochan.models.classification.binary.robust import (
    OutlierRelevancePursuitBinaryClassificationGPModel,
    OutlierRelevancePursuitBinaryClassificationMixedGPModel,
)
from bochan.models.components.robust import SparseOutlierBernoulliLikelihood


class _DummyBinaryModel(Model):
    num_outputs = 1
    batch_shape = torch.Size()
    cat_dims: list[int] = []

    def __init__(self) -> None:
        super().__init__()
        self.raw_train_X = torch.zeros(3, 2, dtype=torch.double)
        self.train_Y = torch.zeros(3, 1, dtype=torch.double)
        self.train_targets = self.train_Y.squeeze(-1)

    def latent_posterior(self, X: torch.Tensor) -> GPyTorchPosterior:
        mean = X[..., 0]
        return GPyTorchPosterior(
            MultivariateNormal(
                mean,
                DiagLinearOperator(torch.full_like(mean, 0.2)),
            )
        )

    def posterior(self, X: torch.Tensor, **kwargs) -> GPyTorchPosterior:
        return self.latent_posterior(X)


def _toy_data(*, mixed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0, 0.0], [0.2, 0.1], [0.4, 0.3], [0.6, 0.7], [0.8, 0.9], [1.0, 1.0]],
        dtype=torch.double,
    )
    if mixed:
        X[:, 1] = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.double)
    return X, (X[:, :1] > 0.5).to(dtype=torch.double)


def test_single_selected_output_keeps_multitask_shape() -> None:
    X = torch.randn(5, 2, dtype=torch.double)
    model = MultiOutputBinaryClassificationModel(_DummyBinaryModel(), _DummyBinaryModel())
    posterior = model.latent_posterior(X, output_indices=[0])
    assert posterior.mean.shape == torch.Size([5, 1])
    assert posterior.rsample(torch.Size([4])).shape == torch.Size([4, 5, 1])


def test_inner_latent_models_use_identity_transform_inputs() -> None:
    X, Y = _toy_data()
    model = _LatentBinarySVGP(X[:3], X, Y.squeeze(-1))
    assert model.transform_inputs(X=X) is X

    X_mixed, Y_mixed = _toy_data(mixed=True)
    mixed = _LatentMixedBinarySVGP(X_mixed[:3], [1], X_mixed, Y_mixed.squeeze(-1))
    assert mixed.transform_inputs(X=X_mixed) is X_mixed


def test_dense_rrp_parameter_preserves_gradient() -> None:
    likelihood = SparseOutlierBernoulliLikelihood(dim=4)
    likelihood.to_dense()
    likelihood.sparse_parameter.requires_grad_(True)
    likelihood.dense_delta.sum().backward()
    assert likelihood.sparse_parameter.grad is not None
    assert torch.allclose(likelihood.sparse_parameter.grad, torch.ones(4))


def _fit_one_step(model) -> None:
    fit_rrp_binary_classifier_mll(
        model.make_mll(),
        method="forward",
        sparsity_levels=[0, 1],
        reset_parameters=True,
        reset_dense_parameters=False,
        record_model_trace=False,
        optimizer_kwargs={"num_epochs": 1, "lr": 0.01},
    )
    assert len(model.likelihood.support) <= 1


def test_continuous_rrp_support_expansion_runs() -> None:
    X, Y = _toy_data()
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    _fit_one_step(
        OutlierRelevancePursuitBinaryClassificationGPModel(
            train_X=X,
            train_Y=Y,
            input_transform=Normalize(d=2, bounds=bounds),
        )
    )


def test_mixed_rrp_support_expansion_runs() -> None:
    X, Y = _toy_data(mixed=True)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    _fit_one_step(
        OutlierRelevancePursuitBinaryClassificationMixedGPModel(
            train_X=X,
            train_Y=Y,
            cat_dims=[1],
            input_transform=Normalize(d=2, bounds=bounds, indices=[0]),
        )
    )
