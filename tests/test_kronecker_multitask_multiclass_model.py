import pytest
import torch

from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationGPModel,
    KroneckerMultiTaskMulticlassProbsPosterior,
)


def _make_train_data(dtype=torch.double):
    train_X = torch.linspace(0.0, 1.0, 7, dtype=dtype).unsqueeze(-1)
    train_Y = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 0],
            [1, 0],
        ],
        dtype=torch.long,
    )
    return train_X, train_Y


def _assert_psd(matrix: torch.Tensor, atol: float = 1e-8) -> None:
    assert torch.allclose(matrix, matrix.transpose(-1, -2), atol=atol)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    assert bool((eigenvalues >= -atol).all())


def test_kronecker_multitask_multiclass_shapes_probabilities_and_elbo():
    train_X, train_Y = _make_train_data()
    model = KroneckerMultiTaskMulticlassClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=2,
        num_inducing=4,
    )

    posterior = model.posterior(train_X[:2])
    latent = model.latent_posterior(train_X[:2])
    selected = model.posterior(train_X[:2], output_indices=[1])
    probabilities = model.class_probs(train_X[:2])

    assert isinstance(posterior, KroneckerMultiTaskMulticlassProbsPosterior)
    assert posterior.mean.shape == torch.Size([2, 2, 3])
    assert posterior.variance.shape == torch.Size([2, 2, 3])
    assert selected.mean.shape == torch.Size([2, 1, 3])
    assert latent.mean.shape == torch.Size([3, 2, 2])
    assert probabilities.shape == torch.Size([2, 2, 3])
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(2, 2, dtype=train_X.dtype),
        atol=1e-6,
    )

    assert model.num_outputs == 2
    assert model.num_classes_list == [3, 3]
    assert model.task_covar_matrix.shape == torch.Size([3, 2, 2])
    for class_index in range(model.num_classes):
        _assert_psd(model.task_covar_matrix[class_index])

    mll = model.make_mll()
    output = model.model(model.train_inputs[0])
    loss = -mll(output, model.train_targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert model.model.variational_strategy.lmc_coefficients.grad is not None


def test_kronecker_multitask_multiclass_t_batch_and_sampling_shapes():
    train_X, train_Y = _make_train_data()
    model = KroneckerMultiTaskMulticlassClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=2,
        num_inducing=4,
    )

    test_X = train_X[:2].unsqueeze(0).expand(4, 2, 1)
    posterior = model.posterior(test_X)
    samples = posterior.rsample(torch.Size([5]))

    assert posterior.mean.shape == torch.Size([4, 2, 2, 3])
    assert samples.shape == torch.Size([5, 4, 2, 2, 3])
    assert torch.allclose(
        posterior.mean.sum(dim=-1),
        torch.ones(4, 2, 2, dtype=train_X.dtype),
        atol=1e-6,
    )


def test_kronecker_multitask_multiclass_helpers_and_conditioning():
    train_X, train_Y = _make_train_data()
    model = KroneckerMultiTaskMulticlassClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=1,
        num_inducing=4,
    )

    probabilities_list = model.class_probs_list(train_X[:2])
    expected_utility = model.expected_utility(
        train_X[:2],
        utility_values=torch.tensor(
            [
                [0.0, 1.0, 3.0],
                [0.0, 2.0, 4.0],
            ],
            dtype=train_X.dtype,
        ),
    )
    conditioned = model.condition_on_observations(
        train_X[:1] + 0.05,
        torch.tensor([[2, 1]], dtype=torch.long),
    )

    assert len(probabilities_list) == 2
    assert probabilities_list[0].shape == torch.Size([2, 3])
    assert expected_utility.shape == torch.Size([2, 2])
    assert conditioned.train_inputs_raw[0].shape[-2] == train_X.shape[-2] + 1
    assert conditioned.train_targets.shape == torch.Size([train_X.shape[-2] + 1, 2])


def test_kronecker_multitask_multiclass_validates_block_design_and_classes():
    train_X, train_Y = _make_train_data()

    with pytest.raises(ValueError, match="block-design"):
        KroneckerMultiTaskMulticlassClassificationGPModel(
            train_X=train_X,
            train_Y=train_Y[:, 0],
            num_classes=3,
        )

    with pytest.raises(ValueError, match="num_classes must be >= 3"):
        KroneckerMultiTaskMulticlassClassificationGPModel(
            train_X=train_X,
            train_Y=train_Y.clamp_max(1),
            num_classes=2,
        )

    invalid_Y = train_Y.clone()
    invalid_Y[0, 0] = 3
    with pytest.raises(ValueError, match="invalid values"):
        KroneckerMultiTaskMulticlassClassificationGPModel(
            train_X=train_X,
            train_Y=invalid_Y,
            num_classes=3,
        )
