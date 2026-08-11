import pytest
import torch

from bochan.models.classification.binary.base import (
    KroneckerMultiTaskBinaryClassificationGPModel,
)
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalGPModel


def _make_block_design_X(dtype=torch.double):
    return torch.linspace(0.0, 1.0, 6, dtype=dtype).unsqueeze(-1)


def _assert_psd(matrix: torch.Tensor, atol: float = 1e-8) -> None:
    assert torch.allclose(matrix, matrix.transpose(-1, -2), atol=atol)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    assert bool((eigenvalues >= -atol).all())


def test_kronecker_multitask_binary_shapes_probabilities_and_elbo():
    train_X = _make_block_design_X()
    train_Y = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 1],
        ],
        dtype=torch.double,
    )

    model = KroneckerMultiTaskBinaryClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y,
        rank=2,
        num_inducing=4,
    )

    posterior = model.posterior(train_X[:3])
    latent = model.latent_posterior(train_X[:3])
    class_probs = model.class_probs(train_X[:3])

    assert posterior.mean.shape == torch.Size([3, 2])
    assert posterior.variance.shape == torch.Size([3, 2])
    assert latent.mean.shape == torch.Size([3, 2])
    assert class_probs.shape == torch.Size([3, 2, 2])
    assert torch.all((posterior.mean > 0.0) & (posterior.mean < 1.0))
    assert torch.allclose(
        class_probs.sum(dim=-1),
        torch.ones(3, 2, dtype=train_X.dtype),
        atol=1e-6,
    )

    assert model.task_covar_matrix.shape == torch.Size([2, 2])
    _assert_psd(model.task_covar_matrix)

    mll = model.make_mll()
    output = model.model(model.train_inputs[0])
    loss = -mll(output, model.train_targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_kronecker_multitask_ordinal_shapes_probabilities_and_elbo():
    train_X = _make_block_design_X()
    train_Y = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
            [2, 1],
            [2, 2],
        ],
        dtype=torch.long,
    )

    model = KroneckerMultiTaskOrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=2,
        num_inducing=4,
    )

    posterior = model.posterior(train_X[:3])
    class_probs = model.class_probs(train_X[:3])
    selected_probs = model.class_probs(train_X[:3], output_indices=[1])

    assert posterior.mean.shape == torch.Size([3, 2])
    assert class_probs.shape == torch.Size([3, 2, 3])
    assert selected_probs.shape == torch.Size([3, 1, 3])
    assert torch.allclose(
        class_probs.sum(dim=-1),
        torch.ones(3, 2, dtype=train_X.dtype),
        atol=1e-5,
    )

    assert model.task_covar_matrix.shape == torch.Size([2, 2])
    _assert_psd(model.task_covar_matrix)

    mll = model.make_mll()
    output = model.model(model.train_inputs[0])
    loss = -mll(output, model.train_targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_kronecker_multitask_models_require_block_design_targets():
    train_X = _make_block_design_X()

    with pytest.raises(ValueError, match="block-design"):
        KroneckerMultiTaskBinaryClassificationGPModel(
            train_X=train_X,
            train_Y=torch.zeros(train_X.shape[0], dtype=train_X.dtype),
        )

    with pytest.raises(ValueError, match="block-design"):
        KroneckerMultiTaskOrdinalGPModel(
            train_X=train_X,
            train_Y=torch.zeros(train_X.shape[0], dtype=torch.long),
            num_classes=3,
        )
