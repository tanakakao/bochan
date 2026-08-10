from __future__ import annotations

import inspect
from types import MethodType

import torch

from bochan.acquisition.binary.bayesian_optimization import (
    qBinaryExpectedImprovement,
    qBinaryProbabilityOfImprovement,
    qBinaryUpperConfidenceBound,
)
from bochan.acquisition.objective.binary import (
    BinaryClassificationScoreObjective,
)
from bochan.models.classification.binary.base import (
    BinaryClassificationGPModel,
    BinaryClassificationMixedGPModel,
)
from bochan.models.transforms.input import build_input_transform


def _make_model() -> BinaryClassificationGPModel:
    torch.manual_seed(11)
    train_X = torch.rand(16, 2, dtype=torch.double)
    train_Y = (train_X[:, :1] > 0.5).to(dtype=train_X.dtype)
    return BinaryClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_inducing=12,
    )


def _bind_probability_samples(acqf, samples: torch.Tensor) -> None:
    def _posterior_samples_as_prob(self, X: torch.Tensor) -> torch.Tensor:
        return samples.to(device=X.device, dtype=X.dtype)

    acqf._posterior_samples_as_prob = MethodType(
        _posterior_samples_as_prob,
        acqf,
    )


def test_binary_probability_bo_defaults() -> None:
    for cls in (
        qBinaryExpectedImprovement,
        qBinaryProbabilityOfImprovement,
        qBinaryUpperConfidenceBound,
    ):
        signature = inspect.signature(cls.__init__)
        assert signature.parameters["apply_sigmoid_if_needed"].default is True
        assert signature.parameters["q_mode"].default == "pointwise"
        assert signature.parameters["reduction"].default == "mean"
        assert "X_pending" in signature.parameters
        assert "X_observed" in signature.parameters
        assert "X_baseline" in signature.parameters
        assert "same_batch_penalty_weight" in signature.parameters
        assert "observed_penalty_weight" in signature.parameters


def test_ei_pointwise_requires_all_batch_points_to_be_good() -> None:
    model = _make_model()
    X = torch.rand(1, 3, 2, dtype=torch.double)

    all_high = torch.full((8, 1, 3), 0.9, dtype=torch.double)
    one_high = torch.tensor(
        [[[0.9, 0.2, 0.1]]],
        dtype=torch.double,
    ).expand(8, -1, -1)

    pointwise = qBinaryExpectedImprovement(
        model=model,
        best_f=0.5,
        q_mode="pointwise",
    )
    _bind_probability_samples(pointwise, all_high)
    value_all_high = pointwise(X)
    _bind_probability_samples(pointwise, one_high)
    value_one_high = pointwise(X)

    assert value_all_high.shape == torch.Size([1])
    assert value_all_high.item() > value_one_high.item()

    joint = qBinaryExpectedImprovement(
        model=model,
        best_f=0.5,
        q_mode="joint",
    )
    _bind_probability_samples(joint, all_high)
    joint_all_high = joint(X)
    _bind_probability_samples(joint, one_high)
    joint_one_high = joint(X)

    assert torch.allclose(joint_all_high, joint_one_high)


def test_pi_pointwise_requires_all_batch_points_to_improve() -> None:
    model = _make_model()
    X = torch.rand(1, 3, 2, dtype=torch.double)

    all_high = torch.full((8, 1, 3), 0.8, dtype=torch.double)
    one_high = torch.tensor(
        [[[0.8, 0.2, 0.1]]],
        dtype=torch.double,
    ).expand(8, -1, -1)

    pointwise = qBinaryProbabilityOfImprovement(
        model=model,
        best_f=0.5,
        tau=0.02,
        q_mode="pointwise",
    )
    _bind_probability_samples(pointwise, all_high)
    value_all_high = pointwise(X)
    _bind_probability_samples(pointwise, one_high)
    value_one_high = pointwise(X)

    assert value_all_high.item() > value_one_high.item()


def test_ucb_pointwise_differs_from_botorch_style_joint_max() -> None:
    model = _make_model()
    X = torch.rand(1, 3, 2, dtype=torch.double)

    all_high = torch.tensor(
        [
            [[0.75, 0.75, 0.75]],
            [[0.85, 0.85, 0.85]],
        ],
        dtype=torch.double,
    )
    one_high = torch.tensor(
        [
            [[0.75, 0.20, 0.10]],
            [[0.85, 0.30, 0.20]],
        ],
        dtype=torch.double,
    )

    pointwise = qBinaryUpperConfidenceBound(
        model=model,
        beta=0.1,
        q_mode="pointwise",
    )
    _bind_probability_samples(pointwise, all_high)
    value_all_high = pointwise(X)
    _bind_probability_samples(pointwise, one_high)
    value_one_high = pointwise(X)

    assert value_all_high.item() > value_one_high.item()

    joint = qBinaryUpperConfidenceBound(
        model=model,
        beta=0.1,
        q_mode="joint",
    )
    _bind_probability_samples(joint, all_high)
    joint_all_high = joint(X)
    _bind_probability_samples(joint, one_high)
    joint_one_high = joint(X)

    assert torch.allclose(joint_all_high, joint_one_high)


def test_same_batch_penalty_discourages_duplicate_candidates() -> None:
    model = _make_model()
    acqf = qBinaryExpectedImprovement(
        model=model,
        best_f=0.5,
        q_mode="pointwise",
        same_batch_penalty_weight=0.5,
        same_batch_penalty_beta=10.0,
    )
    samples = torch.full((8, 1, 3), 0.8, dtype=torch.double)
    _bind_probability_samples(acqf, samples)

    duplicate = torch.tensor(
        [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]],
        dtype=torch.double,
    )
    separated = torch.tensor(
        [[[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]],
        dtype=torch.double,
    )

    assert acqf(separated).item() > acqf(duplicate).item()


def test_ei_can_infer_best_f_from_binary_model() -> None:
    model = _make_model()
    acqf = qBinaryExpectedImprovement(model=model)

    assert acqf.best_f.ndim == 0
    assert 0.0 < acqf.best_f.item() < 1.0


def test_pointwise_ei_keeps_candidate_gradients() -> None:
    model = _make_model()
    acqf = qBinaryExpectedImprovement(
        model=model,
        best_f=0.4,
        q_mode="pointwise",
    )
    X = torch.rand(
        3,
        2,
        2,
        dtype=torch.double,
        requires_grad=True,
    )

    value = acqf(X)
    gradient = torch.autograd.grad(value.sum(), X)[0]

    assert value.shape == torch.Size([3])
    assert gradient.shape == X.shape
    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()


def test_input_perturbation_penalties_keep_original_q_shape() -> None:
    torch.manual_seed(17)
    n_w = 8
    continuous = torch.rand(20, 2, dtype=torch.double)
    categorical = torch.randint(0, 2, (20, 1)).to(dtype=torch.double)
    train_X = torch.cat([continuous, categorical], dim=-1)
    train_Y = (
        train_X[:, 0] + 0.25 * train_X[:, 1] > 0.6
    ).to(dtype=train_X.dtype).unsqueeze(-1)
    bounds = torch.stack(
        [
            train_X.min(dim=0).values,
            train_X.max(dim=0).values,
        ],
        dim=0,
    )
    input_transform = build_input_transform(
        train_X=train_X,
        bounds=bounds,
        perturbation=True,
        categorical_idx=[2],
        n_w=n_w,
        std=0.1,
    )
    model = BinaryClassificationMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[2],
        input_transform=input_transform,
        num_inducing=12,
    )

    X = torch.tensor(
        [[[0.2, 0.3, 0.0], [0.5, 0.6, 1.0], [0.8, 0.4, 0.0]]],
        dtype=torch.double,
    )
    X_pending = torch.tensor([[0.4, 0.5, 1.0]], dtype=torch.double)
    acqf = qBinaryExpectedImprovement(
        model=model,
        best_f=0.5,
        q_mode="pointwise",
        objective=BinaryClassificationScoreObjective(n_w=n_w),
        X_pending=X_pending,
        pending_penalty_weight=0.2,
        observed_penalty_weight=0.2,
        same_batch_penalty_weight=0.2,
    )
    samples = torch.full((16, 1, 3), 0.8, dtype=torch.double)
    _bind_probability_samples(acqf, samples)

    penalty_X = acqf._apply_penalty_input_transform(X)
    value = acqf(X)

    assert penalty_X.shape == X.shape
    assert torch.equal(penalty_X[..., 2], X[..., 2])
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_zero_weight_penalties_do_not_expand_pointwise_score() -> None:
    torch.manual_seed(19)
    n_w = 8
    train_X = torch.rand(16, 2, dtype=torch.double)
    train_Y = (train_X[:, :1] > 0.5).to(dtype=train_X.dtype)
    bounds = torch.stack(
        [
            train_X.min(dim=0).values,
            train_X.max(dim=0).values,
        ],
        dim=0,
    )
    input_transform = build_input_transform(
        train_X=train_X,
        bounds=bounds,
        perturbation=True,
        n_w=n_w,
        std=0.1,
    )
    model = BinaryClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=input_transform,
        num_inducing=12,
    )
    acqf = qBinaryExpectedImprovement(
        model=model,
        best_f=0.5,
        q_mode="pointwise",
        objective=BinaryClassificationScoreObjective(n_w=n_w),
    )
    X = torch.rand(1, 3, 2, dtype=torch.double)
    samples = torch.full((16, 1, 3), 0.8, dtype=torch.double)
    _bind_probability_samples(acqf, samples)

    value = acqf(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()

