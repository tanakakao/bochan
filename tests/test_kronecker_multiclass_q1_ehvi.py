from __future__ import annotations

import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from bochan.acquisition.multiclass.bayesian_optimization import (
    MulticlassTargetProbabilityObjective,
    compute_observed_multiclass_utility,
    qMultiOutputMulticlassExpectedHypervolumeImprovement,
)
from bochan.acquisition.multiclass.bayesian_optimization.input_perturbation_compat import (
    validate_hypervolume_objective_q,
)
from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationGPModel,
)


def _make_model() -> KroneckerMultiTaskMulticlassClassificationGPModel:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
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
        num_inducing_points=4,
    )
    model.eval()
    model.likelihood.eval()
    return model


def test_kronecker_multiclass_restores_q1_for_t_batch_posterior() -> None:
    model = _make_model()
    batch_size = 32
    X = torch.rand(batch_size, 1, 1, dtype=torch.double)

    posterior = model.posterior(X)
    selected = model.posterior(X, output_indices=[1])
    samples = posterior.rsample(torch.Size([8]))

    assert posterior.mean.shape == torch.Size([batch_size, 1, 2, 3])
    assert posterior.variance.shape == torch.Size([batch_size, 1, 2, 3])
    assert posterior.batch_shape == torch.Size([batch_size])
    assert selected.mean.shape == torch.Size([batch_size, 1, 1, 3])
    assert samples.shape == torch.Size([8, batch_size, 1, 2, 3])


def test_kronecker_multiclass_q1_objective_preserves_candidate_axis() -> None:
    model = _make_model()
    batch_size = 32
    X = torch.rand(batch_size, 1, 1, dtype=torch.double)
    samples = model.posterior(X).rsample(torch.Size([8]))
    objective = MulticlassTargetProbabilityObjective(num_outputs=2)

    values = objective(samples, X=X)

    assert values.shape == torch.Size([8, batch_size, 1, 2])
    validate_hypervolume_objective_q(values, X)


def test_kronecker_multiclass_ehvi_accepts_sequential_q1_t_batches() -> None:
    model = _make_model()
    observed = compute_observed_multiclass_utility(model.train_Y).to(
        dtype=model.train_X.dtype,
        device=model.train_X.device,
    )
    ref_point = observed.min(dim=0).values - 0.1
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=observed,
    )
    acquisition = qMultiOutputMulticlassExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    X = torch.rand(6, 1, 1, dtype=torch.double)

    values = acquisition(X)

    assert values.shape == torch.Size([6])
    assert torch.isfinite(values).all()
