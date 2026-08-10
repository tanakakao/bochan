import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf_mixed

from bochan.acquisition.binary.active_learning import (
    qMultiOutputBinaryPredictiveEntropy,
)
from bochan.acquisition.multiclass.active_learning import (
    qMultiOutputMulticlassPredictiveEntropy,
)
from bochan.acquisition.ordinal.active_learning import (
    qMultiOutputOrdinalPredictiveEntropy,
)
from bochan.acquisition.regression.active_learning.multi_output import (
    qMultiOutputRegressionPosteriorVariance,
)
from bochan.models.classification.binary.base import (
    KroneckerMultiTaskBinaryClassificationMixedGPModel,
)
from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationMixedGPModel,
)
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalMixedGPModel
from bochan.models.regression.gaussian import GaussianMixedKroneckerMultiTaskGP


def _make_mixed_X(dtype=torch.double) -> torch.Tensor:
    continuous = torch.linspace(0.0, 1.0, 8, dtype=dtype).unsqueeze(-1)
    category = torch.tensor(
        [[0], [1], [0], [1], [0], [1], [0], [1]],
        dtype=dtype,
    )
    return torch.cat([continuous, category], dim=-1)


def _continuous_only_normalize() -> Normalize:
    return Normalize(d=2, indices=[0])


def test_mixed_binary_kronecker_posterior_acquisition_and_conditioning():
    train_X = _make_mixed_X()
    train_Y = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        dtype=train_X.dtype,
    )
    model = KroneckerMultiTaskBinaryClassificationMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        rank=2,
        input_transform=_continuous_only_normalize(),
        num_inducing_points=4,
    )
    model.eval()
    model.likelihood.eval()

    X = torch.tensor(
        [
            [[0.2, 0.0], [0.7, 1.0]],
            [[0.4, 1.0], [0.9, 0.0]],
        ],
        dtype=train_X.dtype,
    )
    posterior = model.posterior(X)
    acquisition = qMultiOutputBinaryPredictiveEntropy(model, reduction="mean")

    assert model.cat_dims == [1]
    assert model.cont_dims == [0]
    assert posterior.mean.shape == torch.Size([2, 2, 2])
    assert acquisition(X).shape == torch.Size([2])

    updated = model.condition_on_observations(
        X=torch.tensor([[0.3, 0.0]], dtype=train_X.dtype),
        Y=torch.tensor([[1.0, 0.0]], dtype=train_X.dtype),
    )
    assert isinstance(
        updated,
        KroneckerMultiTaskBinaryClassificationMixedGPModel,
    )
    assert updated.cat_dims == [1]
    assert updated.train_inputs_raw[0].shape[-2] == train_X.shape[-2] + 1


def test_mixed_multiclass_kronecker_shapes_and_class_kernel_batch():
    train_X = _make_mixed_X()
    train_Y = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 0],
            [1, 0],
            [0, 2],
        ],
        dtype=torch.long,
    )
    model = KroneckerMultiTaskMulticlassClassificationMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        num_classes=3,
        rank=2,
        input_transform=_continuous_only_normalize(),
        num_inducing_points=4,
    )
    model.eval()
    model.likelihood.eval()

    X = torch.tensor(
        [
            [[0.2, 0.0], [0.7, 1.0]],
            [[0.4, 1.0], [0.9, 0.0]],
        ],
        dtype=train_X.dtype,
    )
    posterior = model.posterior(X)
    acquisition = qMultiOutputMulticlassPredictiveEntropy(
        model,
        reduction="mean",
        output_mode="mean",
    )

    assert model.cat_dims == [1]
    assert model.model.data_covar_module.batch_shape == torch.Size([3, 1])
    assert posterior.mean.shape == torch.Size([2, 2, 2, 3])
    assert acquisition(X).shape == torch.Size([2])
    assert torch.allclose(
        posterior.mean.sum(dim=-1),
        torch.ones(2, 2, 2, dtype=train_X.dtype),
        atol=1e-6,
    )

    updated = model.condition_on_observations(
        X=torch.tensor([[0.3, 0.0]], dtype=train_X.dtype),
        Y=torch.tensor([[1, 2]], dtype=torch.long),
    )
    assert isinstance(
        updated,
        KroneckerMultiTaskMulticlassClassificationMixedGPModel,
    )
    assert updated.cat_dims == [1]


def test_mixed_ordinal_kronecker_shapes_acquisition_and_conditioning():
    train_X = _make_mixed_X()
    train_Y = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=torch.long,
    )
    model = KroneckerMultiTaskOrdinalMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        num_classes=3,
        rank=2,
        input_transform=_continuous_only_normalize(),
        num_inducing_points=4,
    )
    model.eval()
    model.likelihood.eval()

    X = torch.tensor(
        [
            [[0.2, 0.0], [0.7, 1.0]],
            [[0.4, 1.0], [0.9, 0.0]],
        ],
        dtype=train_X.dtype,
    )
    probs = model.class_probs(X)
    acquisition = qMultiOutputOrdinalPredictiveEntropy(
        model,
        reduction="mean",
    )

    assert model.cat_dims == [1]
    assert probs.shape == torch.Size([2, 2, 2, 3])
    assert acquisition(X).shape == torch.Size([2])

    updated = model.condition_on_observations(
        X=torch.tensor([[0.3, 0.0]], dtype=train_X.dtype),
        Y=torch.tensor([[1, 2]], dtype=torch.long),
        refit=False,
    )
    assert isinstance(updated, KroneckerMultiTaskOrdinalMixedGPModel)
    assert updated.cat_dims == [1]


def test_mixed_gaussian_kronecker_shapes_and_acquisition():
    train_X = _make_mixed_X()
    continuous = train_X[:, 0]
    category = train_X[:, 1]
    train_Y = torch.stack(
        [
            torch.sin(2.0 * torch.pi * continuous) + 0.2 * category,
            0.7 * torch.sin(2.0 * torch.pi * continuous) - 0.3 * category,
        ],
        dim=-1,
    )
    model = GaussianMixedKroneckerMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        rank=1,
        input_transform=_continuous_only_normalize(),
    )
    model.eval()
    model.likelihood.eval()

    X = torch.tensor(
        [
            [[0.2, 0.0], [0.7, 1.0]],
            [[0.4, 1.0], [0.9, 0.0]],
        ],
        dtype=train_X.dtype,
    )
    posterior = model.posterior(X)
    acquisition = qMultiOutputRegressionPosteriorVariance(
        model,
        reduction="mean",
        output_reduction="mean",
    )

    assert model.cat_dims == [1]
    assert posterior.mean.shape == torch.Size([2, 2, 2])
    assert acquisition(X).shape == torch.Size([2])


def test_mixed_kronecker_rejects_transforming_categorical_columns():
    train_X = _make_mixed_X()
    train_Y = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        dtype=train_X.dtype,
    )

    with pytest.raises(ValueError, match="categorical columns"):
        KroneckerMultiTaskBinaryClassificationMixedGPModel(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[1],
            input_transform=Normalize(d=2),
            num_inducing_points=4,
        )


def test_mixed_kronecker_works_with_optimize_acqf_mixed():
    train_X = _make_mixed_X()
    train_Y = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        dtype=train_X.dtype,
    )
    model = KroneckerMultiTaskBinaryClassificationMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        rank=2,
        input_transform=_continuous_only_normalize(),
        num_inducing_points=4,
    )
    model.model.mean_module.constant.data.fill_(0.3)
    model.eval()
    model.likelihood.eval()

    acquisition = qMultiOutputBinaryPredictiveEntropy(
        model,
        reduction="mean",
        pending_penalty_weight=0.1,
    )
    bounds = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]],
        dtype=train_X.dtype,
    )
    candidates, value = optimize_acqf_mixed(
        acq_function=acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=8,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        options={"maxiter": 5, "batch_limit": 4},
    )

    assert candidates.shape == torch.Size([1, 2])
    assert candidates[0, 1].item() in (0.0, 1.0)
    assert torch.isfinite(value).all()
