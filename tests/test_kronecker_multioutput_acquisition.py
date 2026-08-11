import torch
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.acquisition.binary.active_learning import (
    qMultiOutputBinaryBALD,
    qMultiOutputBinaryMarginUncertainty,
    qMultiOutputBinaryPredictiveEntropy,
    qMultiOutputBinaryProbabilityVariance,
)
from bochan.acquisition.multiclass.active_learning import (
    qMultiOutputMulticlassPredictiveEntropy,
)
from bochan.acquisition.ordinal.active_learning import (
    qMultiOutputOrdinalBALD,
    qMultiOutputOrdinalMarginUncertainty,
    qMultiOutputOrdinalPredictiveEntropy,
    qMultiOutputOrdinalUtilityVariance,
)
from bochan.acquisition.regression.active_learning.multi_output import (
    qMultiOutputRegressionPosteriorVariance,
)
from bochan.models.classification.binary.base import (
    KroneckerMultiTaskBinaryClassificationGPModel,
)
from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationGPModel,
)
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalGPModel


def _make_X(dtype=torch.double) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 8, dtype=dtype).unsqueeze(-1)


def _make_binary_model() -> KroneckerMultiTaskBinaryClassificationGPModel:
    train_X = _make_X()
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
    model = KroneckerMultiTaskBinaryClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y,
        rank=2,
        num_inducing=4,
    )
    # Avoid a perfectly symmetric zero-mean prior in the optimizer smoke test.
    model.model.mean_module.constant.data.fill_(0.35)
    model.eval()
    model.likelihood.eval()
    return model


def _make_ordinal_model() -> KroneckerMultiTaskOrdinalGPModel:
    train_X = _make_X()
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
    model = KroneckerMultiTaskOrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=2,
        num_inducing=4,
    )
    model.eval()
    model.likelihood.eval()
    return model


def _make_multiclass_model() -> KroneckerMultiTaskMulticlassClassificationGPModel:
    train_X = _make_X()
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
    model = KroneckerMultiTaskMulticlassClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=2,
        num_inducing=4,
    )
    model.eval()
    model.likelihood.eval()
    return model


def test_binary_kronecker_supports_t_batch_multioutput_acquisitions():
    model = _make_binary_model()
    X = torch.rand(5, 2, 1, dtype=torch.double)

    assert model.posterior(X).mean.shape == torch.Size([5, 2, 2])
    assert model.latent_posterior(X).mean.shape == torch.Size([5, 2, 2])

    acquisitions = [
        qMultiOutputBinaryPredictiveEntropy(model, reduction="mean"),
        qMultiOutputBinaryMarginUncertainty(model, reduction="mean"),
        qMultiOutputBinaryProbabilityVariance(
            model,
            reduction="mean",
            num_samples=8,
        ),
        qMultiOutputBinaryBALD(
            model,
            reduction="mean",
            output_mode="mean",
            num_samples=8,
        ),
    ]

    for acquisition in acquisitions:
        value = acquisition(X)
        assert value.shape == torch.Size([5])
        assert torch.isfinite(value).all()


def test_binary_kronecker_works_with_sequential_optimize_acqf():
    model = _make_binary_model()
    acquisition = qMultiOutputBinaryPredictiveEntropy(
        model,
        reduction="mean",
        pending_penalty_weight=0.1,
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidates, acquisition_value = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=2,
        num_restarts=2,
        raw_samples=8,
        sequential=True,
        options={"maxiter": 5, "batch_limit": 4},
    )

    assert candidates.shape == torch.Size([2, 1])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(acquisition_value).all()


def test_ordinal_kronecker_supports_existing_multioutput_acquisitions():
    model = _make_ordinal_model()
    X = torch.rand(4, 2, 1, dtype=torch.double)

    posterior = model.posterior(X)
    assert posterior.mean.shape == torch.Size([4, 2, 2])
    assert len(model.models) == 2
    assert model.models[0].posterior(X).mean.shape == torch.Size([4, 2])

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([8]))
    acquisitions = [
        qMultiOutputOrdinalPredictiveEntropy(model, reduction="mean"),
        qMultiOutputOrdinalMarginUncertainty(model, reduction="mean"),
        qMultiOutputOrdinalUtilityVariance(model, reduction="mean"),
        qMultiOutputOrdinalBALD(
            model,
            reduction="mean",
            sampler=sampler,
        ),
    ]

    for acquisition in acquisitions:
        value = acquisition(X)
        assert value.shape == torch.Size([4])
        assert torch.isfinite(value).all()


def test_multiclass_kronecker_supports_multioutput_acquisition_t_batches():
    model = _make_multiclass_model()
    X = torch.rand(4, 2, 1, dtype=torch.double)
    acquisition = qMultiOutputMulticlassPredictiveEntropy(
        model,
        reduction="mean",
        output_mode="mean",
    )

    value = acquisition(X)

    assert model.posterior(X).mean.shape == torch.Size([4, 2, 2, 3])
    assert value.shape == torch.Size([4])
    assert torch.isfinite(value).all()


def test_gaussian_kronecker_supports_regression_multioutput_acquisition():
    train_X = _make_X()
    base = torch.sin(2.0 * torch.pi * train_X.squeeze(-1))
    train_Y = torch.stack([base, 0.7 * base + 0.2 * train_X.squeeze(-1)], dim=-1)
    model = KroneckerMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        rank=1,
    )
    model.eval()
    model.likelihood.eval()

    X = torch.rand(4, 2, 1, dtype=torch.double)
    acquisition = qMultiOutputRegressionPosteriorVariance(
        model,
        reduction="mean",
        output_reduction="mean",
    )

    value = acquisition(X)

    assert model.posterior(X).mean.shape == torch.Size([4, 2, 2])
    assert value.shape == torch.Size([4])
    assert torch.isfinite(value).all()
