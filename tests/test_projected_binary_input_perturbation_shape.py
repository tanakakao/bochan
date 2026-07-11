from __future__ import annotations

import pytest
import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from torch import nn

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
from bochan.models.classification.multiclass.high_dim import (
    PCAMulticlassClassificationGPModel,
    REMBOMulticlassClassificationGPModel,
)
from bochan.models.ordinal.high_dim import (
    PCAOrdinalGPModel,
    REMBOOrdinalGPModel,
)
from bochan.models.projected_input_perturbation import (
    flatten_projected_one_to_many_point_axes,
)
from bochan.models.transforms.input import build_input_transform


class _NestedOneToManyTransform(nn.Module):
    """Return ``batch x q x n_w x d`` only during evaluation."""

    def __init__(self, n_w: int) -> None:
        super().__init__()
        self.n_w = int(n_w)
        self.transform_on_train = False
        self.transform_on_eval = True
        self.transform_on_fantasize = True
        self.is_one_to_many = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.training:
            return X
        offsets = torch.linspace(
            -0.02,
            0.02,
            self.n_w,
            dtype=X.dtype,
            device=X.device,
        )
        offsets = offsets.view(*([1] * (X.ndim - 1)), self.n_w, 1)
        return X.unsqueeze(-2) + offsets


def test_shared_projected_shape_helper_flattens_only_point_axes() -> None:
    X = torch.rand(7, 3, 5, dtype=torch.double)
    projected = torch.rand(7, 3, 4, 2, dtype=torch.double)

    normalized = flatten_projected_one_to_many_point_axes(X, projected)

    assert normalized.shape == torch.Size([7, 12, 2])
    assert torch.allclose(normalized, projected.reshape(7, 12, 2))


def test_projected_shape_support_is_installed_for_classification_and_ordinal() -> None:
    classes = (
        PCABinaryClassificationGPModel,
        REMBOBinaryClassificationGPModel,
        PCAMulticlassClassificationGPModel,
        REMBOMulticlassClassificationGPModel,
        PCAOrdinalGPModel,
        REMBOOrdinalGPModel,
    )

    assert all(
        getattr(cls, "_bochan_projected_perturbation_patched", False)
        for cls in classes
    )


@pytest.mark.parametrize(
    "model_cls",
    [PCABinaryClassificationGPModel, REMBOBinaryClassificationGPModel],
)
def test_projected_binary_transform_flattens_nested_perturbation_axis(
    model_cls,
) -> None:
    torch.manual_seed(0)
    train_X = torch.rand(12, 5, dtype=torch.double)
    train_Y = torch.randint(0, 2, (12,), dtype=torch.double)
    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=_NestedOneToManyTransform(n_w=4),
        n_components=2,
    )
    model.eval()

    X = torch.rand(7, 3, 5, dtype=torch.double)
    transformed = model.transform_inputs(X)

    assert transformed.shape == torch.Size([7, 12, 2])


@pytest.mark.parametrize(
    "model_cls",
    [PCABinaryClassificationGPModel, REMBOBinaryClassificationGPModel],
)
def test_projected_binary_real_input_perturbation_has_one_point_axis(
    model_cls,
) -> None:
    torch.manual_seed(2)
    train_X = torch.rand(12, 5, dtype=torch.double)
    train_Y = torch.randint(0, 2, (12,), dtype=torch.double)
    bounds = torch.stack((train_X.min(dim=0).values, train_X.max(dim=0).values))
    input_transform = build_input_transform(
        train_X=train_X,
        bounds=bounds,
        perturbation=True,
        n_w=4,
        normalize=True,
    )
    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=input_transform,
        n_components=2,
    )
    model.eval()

    X = torch.rand(7, 3, 5, dtype=torch.double)
    transformed = model.transform_inputs(X)

    assert transformed.shape == torch.Size([7, 12, 2])


@pytest.mark.parametrize(
    "model_cls",
    [PCABinaryClassificationGPModel, REMBOBinaryClassificationGPModel],
)
def test_projected_multioutput_binary_ehvi_preserves_objective_axis(
    model_cls,
) -> None:
    torch.manual_seed(1)
    train_X = torch.rand(16, 5, dtype=torch.double)
    train_Y = torch.randint(0, 2, (16, 2), dtype=torch.double)

    submodels = [
        model_cls(
            train_X=train_X,
            train_Y=train_Y[:, output_index],
            input_transform=_NestedOneToManyTransform(n_w=4),
            n_components=2,
        )
        for output_index in range(2)
    ]
    model = MultiOutputBinaryClassificationModel(*submodels)
    model.eval()

    X = torch.rand(2, 3, 5, dtype=torch.double)
    posterior = model.posterior(X)
    assert posterior.mean.shape == torch.Size([2, 12, 2])

    objective = MultiOutputBinaryClassificationInputPerturbationObjective(
        n_w=4,
        risk_type=None,
    )
    objective_values = objective(posterior.mean.unsqueeze(0), X=X)
    assert objective_values.shape == torch.Size([1, 2, 3, 2])

    ref_point = torch.tensor([-0.1, -0.1], dtype=torch.double)
    observed_Y = torch.tensor(
        [[0.2, 0.4], [0.5, 0.1], [0.3, 0.3]],
        dtype=torch.double,
    )
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=observed_Y,
    )
    acquisition = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        objective=objective,
    )

    value = acquisition(X)
    assert value.shape == torch.Size([2])
    assert torch.isfinite(value).all()
