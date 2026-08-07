from __future__ import annotations

import pytest
import torch

from bochan.acquisition.binary.active_learning import qBinaryPredictiveEntropy
from bochan.acquisition.multiclass.active_learning import qMulticlassPredictiveEntropy
from bochan.acquisition.ordinal.active_learning import qOrdinalPredictiveEntropy
from bochan.models.classification.binary.base import BinaryClassificationGPModel
from bochan.models.classification.multiclass.base import MulticlassClassificationGPModel
from bochan.models.ordinal.base import OrdinalGPModel
from bochan.models.transforms.input import build_input_transform

DTYPE = torch.double
BOUNDS = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
N_W = 3


class _RecordingScoreObjective(torch.nn.Module):
    """Record raw X and collapse a perturbation-expanded pointwise score."""

    def __init__(self) -> None:
        super().__init__()
        self.X: torch.Tensor | None = None

    def forward(
        self,
        score: torch.Tensor,
        X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert X is not None
        self.X = X.detach().clone()
        q = int(X.shape[-2])
        q_like = int(score.shape[-1])
        if q_like == q:
            return score
        assert q_like % q == 0
        n_w = q_like // q
        return score.reshape(*score.shape[:-1], q, n_w).mean(dim=-1)


def _input_transform(train_x: torch.Tensor):
    torch.manual_seed(123)
    return build_input_transform(
        train_X=train_x,
        bounds=BOUNDS,
        perturbation=True,
        n_w=N_W,
        std=0.04,
        normalize=False,
    )


def _binary_model() -> BinaryClassificationGPModel:
    train_x = torch.linspace(0.1, 0.9, 8, dtype=DTYPE).unsqueeze(-1)
    train_y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=DTYPE).unsqueeze(-1)
    model = BinaryClassificationGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_inducing_points=6,
        input_transform=_input_transform(train_x),
    )
    model.eval()
    model.likelihood.eval()
    return model


def _multiclass_model() -> MulticlassClassificationGPModel:
    train_x = torch.linspace(0.1, 0.9, 9, dtype=DTYPE).unsqueeze(-1)
    train_y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)
    model = MulticlassClassificationGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        num_inducing_points=6,
        input_transform=_input_transform(train_x),
    )
    model.eval()
    model.likelihood.eval()
    return model


def _ordinal_model() -> OrdinalGPModel:
    train_x = torch.linspace(0.1, 0.9, 9, dtype=DTYPE).unsqueeze(-1)
    train_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    model = OrdinalGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        inducing_points_num=6,
        input_transform=_input_transform(train_x),
        conditioning_steps=1,
    )
    model.eval()
    model.likelihood.eval()
    return model


@pytest.mark.parametrize("task_type", ["binary", "multiclass", "ordinal"])
def test_input_perturbation_uses_raw_x_for_objective_and_transformed_x_for_distance(
    task_type: str,
) -> None:
    objective = _RecordingScoreObjective()
    raw_x = torch.tensor([[[0.43]]], dtype=DTYPE)
    seen: dict[str, torch.Tensor] = {}

    if task_type == "binary":
        model = _binary_model()
        acquisition = qBinaryPredictiveEntropy(
            model=model,
            num_samples=8,
            objective=objective,
            exclude_same_batch_duplicates=False,
            exclude_pending_duplicates=False,
            exclude_observed_duplicates=False,
        )

        def record_penalty(Xt: torch.Tensor) -> torch.Tensor:
            seen["distance_x"] = Xt.detach().clone()
            return Xt.new_zeros(Xt.shape[:-1])

        acquisition._candidate_penalty_per_point = record_penalty
    elif task_type == "multiclass":
        model = _multiclass_model()
        acquisition = qMulticlassPredictiveEntropy(
            model=model,
            num_samples=8,
            objective=objective,
            exclude_same_batch_duplicates=False,
            exclude_pending_duplicates=False,
            exclude_observed_duplicates=False,
        )

        def record_penalty(Xt: torch.Tensor) -> torch.Tensor:
            seen["distance_x"] = Xt.detach().clone()
            return Xt.new_zeros(Xt.shape[:-1])

        acquisition._pending_penalty_per_point = record_penalty
    else:
        model = _ordinal_model()
        acquisition = qOrdinalPredictiveEntropy(
            model=model,
            objective=objective,
            exclude_same_batch_duplicates=False,
            exclude_pending_duplicates=False,
            exclude_observed_duplicates=False,
        )

        def record_penalty(Xt: torch.Tensor) -> torch.Tensor:
            seen["distance_x"] = Xt.detach().clone()
            return Xt.new_zeros(Xt.shape[:-1])

        acquisition._pointwise_reference_penalty = record_penalty

    value = acquisition(raw_x)
    expected_transformed = model.input_transform(raw_x)

    assert torch.isfinite(value).all()
    assert objective.X is not None
    assert torch.equal(objective.X, raw_x)
    assert "distance_x" in seen
    assert seen["distance_x"].shape[-2] == raw_x.shape[-2] * N_W
    assert torch.allclose(seen["distance_x"], expected_transformed)
