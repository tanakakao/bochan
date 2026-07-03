from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from bochan.acquisition.binary import _align_epistemic_probability_samples
from bochan.acquisition.binary.bayesian_optimization import (
    qMultiOutputBinaryExpectedHypervolumeImprovement,
)
from bochan.acquisition.objective import (
    MultiOutputBinaryClassificationInputPerturbationObjective,
)
from bochan.models.classification.binary.base import (
    MultiOutputBinaryClassificationModel,
)
from bochan.models.classification.binary.high_dim import (
    PCABinaryClassificationGPModel,
    REMBOBinaryClassificationGPModel,
)
from bochan.models.transforms.input import build_input_transform


def test_epistemic_probability_alignment_reduces_only_extra_model_axis() -> None:
    probability_mean = torch.zeros(2, 12, 8, dtype=torch.double)
    posterior = SimpleNamespace(_probability_mean=probability_mean)
    probabilities = torch.arange(
        5 * 2 * 12 * 2 * 8,
        dtype=torch.double,
    ).reshape(5, 2, 12, 2, 8)

    aligned = _align_epistemic_probability_samples(
        posterior,
        probabilities,
        torch.Size([5]),
    )

    expected = probabilities.mean(dim=-2)
    assert aligned.shape == torch.Size([5, 2, 12, 8])
    assert torch.allclose(aligned, expected)


def test_epistemic_probability_alignment_moves_output_axis_to_last() -> None:
    probability_mean = torch.zeros(2, 12, 8, dtype=torch.double)
    posterior = SimpleNamespace(_probability_mean=probability_mean)
    canonical = torch.arange(
        5 * 2 * 12 * 2 * 8,
        dtype=torch.double,
    ).reshape(5, 2, 12, 2, 8)
    misplaced = canonical.movedim(-1, 2)

    aligned = _align_epistemic_probability_samples(
        posterior,
        misplaced,
        torch.Size([5]),
    )

    expected = canonical.mean(dim=-2)
    assert aligned.shape == torch.Size([5, 2, 12, 8])
    assert torch.allclose(aligned, expected)


@pytest.mark.parametrize(
    "model_cls",
    [PCABinaryClassificationGPModel, REMBOBinaryClassificationGPModel],
)
def test_projected_eight_output_binary_epistemic_qehvi_shape(model_cls) -> None:
    torch.manual_seed(0)
    train_X = torch.rand(16, 5, dtype=torch.double)
    train_Y = torch.randint(0, 2, (16, 8), dtype=torch.double)
    bounds = torch.stack(
        (train_X.min(dim=0).values, train_X.max(dim=0).values)
    )

    submodels = []
    for output_index in range(train_Y.shape[-1]):
        input_transform = build_input_transform(
            train_X=train_X,
            bounds=bounds,
            perturbation=True,
            n_w=4,
            normalize=True,
        )
        submodels.append(
            model_cls(
                train_X=train_X,
                train_Y=train_Y[:, output_index],
                input_transform=input_transform,
                n_components=2,
            )
        )

    model = MultiOutputBinaryClassificationModel(*submodels)
    model.eval()

    objective = MultiOutputBinaryClassificationInputPerturbationObjective(
        n_w=4,
        risk_type=None,
    )
    ref_point = torch.full((8,), -0.1, dtype=torch.double)
    observed_Y = torch.rand(12, 8, dtype=torch.double)
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=observed_Y,
    )
    acquisition = qMultiOutputBinaryExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        objective=objective,
    )

    X = torch.rand(2, 3, 5, dtype=torch.double)
    posterior = acquisition.model.posterior(X)
    samples = posterior.rsample(torch.Size([5]))

    assert samples.shape == torch.Size([5, 2, 12, 8])
    objective_values = objective(samples, X=X)
    assert objective_values.shape == torch.Size([5, 2, 3, 8])

    value = acquisition(X)
    assert value.shape == torch.Size([2])
    assert torch.isfinite(value).all()
