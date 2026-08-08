from __future__ import annotations

from types import SimpleNamespace

import torch
from botorch.models import SingleTaskGP

from bochan.acquisition.binary.levelset_estimation import qBinaryICUAcquisition
from bochan.acquisition.multiclass.levelset_estimation import qMulticlassICUAcquisition
from bochan.acquisition.ordinal.levelset_estimation import (
    qMultiOutputOrdinalICUAcquisition,
    qOrdinalICUAcquisition,
)
from bochan.acquisition.regression.levelset_estimation import qRegressionStraddle
from bochan.models.ordinal.base import MultiOutputOrdinalModel, OrdinalGPModel


DTYPE = torch.double


class _DummyBinaryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.train_X = torch.tensor([[0.2], [0.8]], dtype=DTYPE)

    def probability_posterior(self, X: torch.Tensor):
        p = torch.sigmoid(8.0 * (X[..., 0] - 0.5))
        return SimpleNamespace(mean=p.unsqueeze(-1))


class _DummyMulticlassModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.train_X = torch.tensor([[0.2], [0.8]], dtype=DTYPE)
        self.num_classes = 3

    def class_probs(self, X: torch.Tensor) -> torch.Tensor:
        x = X[..., 0]
        logits = torch.stack(
            [x, 1.0 - x, -4.0 * (x - 0.5).square()],
            dim=-1,
        )
        return torch.softmax(logits, dim=-1)


def _ordinal_model(train_y: torch.Tensor | None = None) -> OrdinalGPModel:
    train_x = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=DTYPE)
    if train_y is None:
        train_y = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    model = OrdinalGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        inducing_points_num=5,
        conditioning_steps=1,
    )
    model.eval()
    model.likelihood.eval()
    return model


def _assert_duplicate_contract(acquisition, observed: torch.Tensor, pending: torch.Tensor) -> None:
    observed_value = acquisition(observed.view(1, 1, -1))
    pending_value = acquisition(pending.view(1, 1, -1))
    same_batch = torch.cat([pending.view(1, 1, -1), pending.view(1, 1, -1)], dim=-2)
    same_batch_value = acquisition(same_batch)

    assert torch.isneginf(observed_value).all()
    assert torch.isneginf(pending_value).all()
    assert torch.isneginf(same_batch_value).all()


def test_regression_lse_avoids_observed_pending_and_same_batch_duplicates() -> None:
    train_x = torch.tensor([[0.1], [0.4], [0.8]], dtype=DTYPE)
    train_y = torch.sin(train_x * 3.0)
    model = SingleTaskGP(train_x, train_y)
    model.eval()

    pending = torch.tensor([0.6], dtype=DTYPE)
    acquisition = qRegressionStraddle(
        model=model,
        threshold=0.0,
        X_pending=pending.view(1, -1),
        hard_duplicate_tol=1e-8,
    )

    assert acquisition.X_observed is not None
    _assert_duplicate_contract(acquisition, train_x[0], pending)


def test_binary_lse_exposes_common_duplicate_controls() -> None:
    model = _DummyBinaryModel()
    pending = torch.tensor([0.5], dtype=DTYPE)
    acquisition = qBinaryICUAcquisition(
        model=model,
        X_pending=pending.view(1, -1),
        hard_duplicate_tol=1e-8,
    )

    assert acquisition.X_observed is not None
    _assert_duplicate_contract(acquisition, model.train_X[0], pending)


def test_multiclass_lse_exposes_common_duplicate_controls() -> None:
    model = _DummyMulticlassModel()
    pending = torch.tensor([0.5], dtype=DTYPE)
    acquisition = qMulticlassICUAcquisition(
        model=model,
        target_class=1,
        X_pending=pending.view(1, -1),
        hard_duplicate_tol=1e-8,
    )

    assert acquisition.X_observed is not None
    _assert_duplicate_contract(acquisition, model.train_X[0], pending)


def test_ordinal_lse_avoids_observed_pending_and_same_batch_duplicates() -> None:
    model = _ordinal_model()
    pending = torch.tensor([0.6], dtype=DTYPE)
    acquisition = qOrdinalICUAcquisition(
        model=model,
        X_pending=pending.view(1, -1),
        hard_duplicate_tol=1e-8,
    )

    assert acquisition.X_observed is not None
    _assert_duplicate_contract(acquisition, model.train_X[0], pending)


def test_multi_output_ordinal_lse_uses_same_duplicate_contract() -> None:
    model_a = _ordinal_model(torch.tensor([0, 0, 1, 2, 2], dtype=torch.long))
    model_b = _ordinal_model(torch.tensor([0, 1, 1, 1, 2], dtype=torch.long))
    model = MultiOutputOrdinalModel(model_a, model_b)
    model.eval()

    pending = torch.tensor([0.6], dtype=DTYPE)
    acquisition = qMultiOutputOrdinalICUAcquisition(
        model=model,
        X_pending=pending.view(1, -1),
        hard_duplicate_tol=1e-8,
    )

    assert acquisition.X_observed is not None
    _assert_duplicate_contract(acquisition, model_a.train_X[0], pending)


def test_lse_duplicate_controls_can_be_disabled() -> None:
    point = torch.tensor([[0.5]], dtype=DTYPE)
    model = _DummyMulticlassModel()
    acquisition = qMulticlassICUAcquisition(
        model=model,
        target_class=1,
        X_pending=point,
        X_observed=point,
        exclude_same_batch_duplicates=False,
        exclude_pending_duplicates=False,
        exclude_observed_duplicates=False,
    )

    same_batch = point.view(1, 1, 1).expand(1, 2, 1).clone()
    assert torch.isfinite(acquisition(point.view(1, 1, 1))).all()
    assert torch.isfinite(acquisition(same_batch)).all()


def test_regression_lse_duplicate_tolerance_is_euclidean() -> None:
    train_x = torch.tensor([[0.2], [0.8]], dtype=DTYPE)
    train_y = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
    model = SingleTaskGP(train_x, train_y)
    model.eval()

    acquisition = qRegressionStraddle(
        model=model,
        X_observed=torch.tensor([[0.0]], dtype=DTYPE),
        exclude_same_batch_duplicates=False,
        exclude_pending_duplicates=False,
        hard_duplicate_tol=1e-4,
    )

    inside = torch.tensor([[[0.5e-4]]], dtype=DTYPE)
    outside = torch.tensor([[[2.0e-4]]], dtype=DTYPE)
    assert torch.isneginf(acquisition(inside)).all()
    assert torch.isfinite(acquisition(outside)).all()
