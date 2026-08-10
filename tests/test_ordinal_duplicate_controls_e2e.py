from __future__ import annotations

import torch

from bochan.acquisition.ordinal.active_learning import qOrdinalPredictiveEntropy
from bochan.models.ordinal.base import OrdinalGPModel

DTYPE = torch.double


def _model() -> OrdinalGPModel:
    train_x = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=DTYPE)
    train_y = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    model = OrdinalGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
        num_inducing=5,
        conditioning_steps=1,
    )
    model.eval()
    model.likelihood.eval()
    return model


def test_ordinal_public_duplicate_controls_cover_observed_pending_and_same_batch() -> None:
    model = _model()
    observed = torch.tensor([[0.25]], dtype=DTYPE)
    pending = torch.tensor([[0.75]], dtype=DTYPE)
    acquisition = qOrdinalPredictiveEntropy(
        model=model,
        X_observed=observed,
        X_pending=pending,
        hard_duplicate_tol=1e-8,
        exclude_observed_duplicates=True,
        exclude_pending_duplicates=True,
        exclude_same_batch_duplicates=True,
    )

    observed_value = acquisition(observed.unsqueeze(0))
    pending_value = acquisition(pending.unsqueeze(0))
    same_batch = torch.tensor([[[0.4], [0.4]]], dtype=DTYPE)
    same_batch_value = acquisition(same_batch)

    assert torch.isneginf(observed_value).all()
    assert torch.isneginf(pending_value).all()
    assert torch.isneginf(same_batch_value).all()


def test_ordinal_duplicate_controls_can_be_disabled() -> None:
    model = _model()
    point = torch.tensor([[0.25]], dtype=DTYPE)
    acquisition = qOrdinalPredictiveEntropy(
        model=model,
        X_observed=point,
        X_pending=point,
        exclude_observed_duplicates=False,
        exclude_pending_duplicates=False,
        exclude_same_batch_duplicates=False,
    )

    value = acquisition(point.unsqueeze(0))
    assert torch.isfinite(value).all()
