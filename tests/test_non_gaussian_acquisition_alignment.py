"""Alignment tests for non-Gaussian active-learning acquisitions."""

import torch

from bochan.acquisition.non_gaussian.active_learning import (
    qHeteroMultiOutputNonGaussianGreedyJointBALDProxy,
    qHeteroMultiOutputNonGaussianJointBALDProxy,
    qHeteroMultiOutputNonGaussianNegIntegratedResponseMeanVariance,
    qHeteroNonGaussianGreedyJointBALDProxy,
    qHeteroNonGaussianJointBALDProxy,
    qHeteroNonGaussianNegIntegratedResponseMeanVariance,
    qMultiOutputNonGaussianGreedyJointBALDProxy,
    qMultiOutputNonGaussianJointBALDProxy,
    qMultiOutputNonGaussianNegIntegratedResponseMeanVariance,
    qNonGaussianGreedyJointBALDProxy,
    qNonGaussianIntegratedResponseMeanVarianceProxy,
    qNonGaussianJointBALDProxy,
    qNonGaussianNegIntegratedResponseMeanVariance,
)
from bochan.api.registry.acquisition import resolve_acqf_cls
from bochan.models.regression.gamma import GammaGPModel


def _gamma_model() -> GammaGPModel:
    train_x = torch.linspace(0.05, 0.95, 8, dtype=torch.double).unsqueeze(-1)
    train_y = 0.5 + train_x
    model = GammaGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_inducing=5,
    )
    model.eval()
    model.likelihood.eval()
    return model


def test_integrated_response_variance_proxy_depends_on_candidates() -> None:
    """The integrated proxy must be differentiable and non-constant in X."""
    model = _gamma_model()
    reference = torch.linspace(0.0, 1.0, 11, dtype=torch.double).unsqueeze(-1)
    acquisition = qNonGaussianIntegratedResponseMeanVarianceProxy(
        model=model,
        mc_points=reference,
        sample_shape=torch.Size([64]),
        seed=123,
    )
    candidate = torch.tensor(
        [[[0.15]], [[0.82]]],
        dtype=torch.double,
        requires_grad=True,
    )

    value = acquisition(candidate)

    assert value.shape == torch.Size([2])
    assert torch.isfinite(value).all()
    assert not torch.allclose(value[0], value[1])
    value.sum().backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.abs().sum() > 0


def test_greedy_joint_bald_returns_incremental_last_point_gain() -> None:
    """Greedy Joint BALD must differ semantically from total Joint BALD."""
    model = _gamma_model()
    candidate = torch.tensor(
        [[[0.2], [0.7]]],
        dtype=torch.double,
        requires_grad=True,
    )
    joint = qNonGaussianJointBALDProxy(
        model=model,
        sample_shape=torch.Size([64]),
        seed=17,
    )(candidate)
    greedy = qNonGaussianGreedyJointBALDProxy(
        model=model,
        sample_shape=torch.Size([64]),
        seed=17,
    )(candidate)

    assert joint.shape == greedy.shape == torch.Size([1])
    assert torch.isfinite(joint).all()
    assert torch.isfinite(greedy).all()
    assert torch.all(greedy >= 0)
    assert torch.all(greedy <= joint + 1e-8)
    greedy.sum().backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()


def test_non_gaussian_short_names_follow_regression_task_metadata() -> None:
    """Regression task plus family model type must route to non-Gaussian AL."""
    assert resolve_acqf_cls(
        "nipv",
        task_type="regression",
        model_type="gamma_base",
    ).__name__ == "qNonGaussianNegIntegratedResponseMeanVariance"
    assert resolve_acqf_cls(
        "nipv",
        task_type="regression",
        model_type="poisson_base",
        multi_output=True,
    ).__name__ == "qMultiOutputNonGaussianNegIntegratedResponseMeanVariance"
    assert resolve_acqf_cls(
        "joint_bald",
        task_type="regression",
        model_type="gamma_heteroscedastic",
    ).__name__ == "qHeteroNonGaussianJointBALDProxy"
    assert resolve_acqf_cls(
        "greedy_joint_bald",
        task_type="regression",
        model_type="negative_binomial_hetero",
        multi_output=True,
    ).__name__ == "qHeteroMultiOutputNonGaussianGreedyJointBALDProxy"
    assert resolve_acqf_cls(
        "variance",
        task_type="regression",
        model_type="beta_base",
    ).__name__ == "qNonGaussianResponseMeanVariance"


def test_advanced_active_learning_lineup_is_symmetric() -> None:
    """Advanced AL names must exist across output and hetero variants."""
    classes = [
        qNonGaussianNegIntegratedResponseMeanVariance,
        qNonGaussianJointBALDProxy,
        qNonGaussianGreedyJointBALDProxy,
        qMultiOutputNonGaussianNegIntegratedResponseMeanVariance,
        qMultiOutputNonGaussianJointBALDProxy,
        qMultiOutputNonGaussianGreedyJointBALDProxy,
        qHeteroNonGaussianNegIntegratedResponseMeanVariance,
        qHeteroNonGaussianJointBALDProxy,
        qHeteroNonGaussianGreedyJointBALDProxy,
        qHeteroMultiOutputNonGaussianNegIntegratedResponseMeanVariance,
        qHeteroMultiOutputNonGaussianJointBALDProxy,
        qHeteroMultiOutputNonGaussianGreedyJointBALDProxy,
    ]
    assert all(cls.__name__.startswith("q") for cls in classes)
