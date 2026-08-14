from __future__ import annotations

from pathlib import Path

import torch
from botorch.models.transforms.input import Normalize

from bochan.acquisition.ordinal.active_learning.single_output import (
    _apply_input_transform_for_reference,
    _transform_reference_like_candidate,
)
from bochan.models.ordinal.high_dim import SaasOrdinalMixedGPModel


REPO_ROOT = Path(__file__).resolve().parents[1]
STALE_PATCH = REPO_ROOT / "patches" / "ordinal_active_learning_ohe_reference_transform_fix.patch"


def _make_mixed_ordinal_model() -> SaasOrdinalMixedGPModel:
    train_X = torch.tensor(
        [
            [0.10, 5.0],
            [0.20, 10.0],
            [0.30, 15.0],
            [0.60, 5.0],
            [0.70, 10.0],
            [0.80, 15.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    bounds = torch.tensor(
        [[0.0, 5.0], [1.0, 15.0]],
        dtype=train_X.dtype,
    )
    return SaasOrdinalMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        cat_dims=[1],
        input_transform=Normalize(d=2, bounds=bounds),
        num_inducing=4,
    )


def test_mixed_model_input_transform_accepts_raw_reference_space() -> None:
    model = _make_mixed_ordinal_model()
    raw_X = torch.tensor(
        [[[0.25, 5.0], [0.75, 15.0]]],
        dtype=torch.double,
    )

    direct = model.input_transform(raw_X)
    canonical = model.transform_inputs(raw_X)

    assert direct.shape[-1] == model.encoded_dim
    assert direct.shape[-1] > model.raw_dim
    assert torch.allclose(direct, canonical)


def test_active_learning_reference_transform_matches_mixed_candidate_space() -> None:
    model = _make_mixed_ordinal_model()
    candidate_raw = torch.tensor(
        [[[0.25, 5.0], [0.75, 15.0]]],
        dtype=torch.double,
    )
    pending_raw = torch.tensor(
        [[0.25, 5.0]],
        dtype=torch.double,
    )

    candidate_tf = _apply_input_transform_for_reference(model, candidate_raw)
    pending_tf = _transform_reference_like_candidate(
        model,
        pending_raw,
        ref=candidate_tf,
    )

    expected_pending = model.transform_inputs(pending_raw.unsqueeze(0))
    assert candidate_tf.shape[-1] == model.encoded_dim
    assert pending_tf is not None
    assert pending_tf.shape[-1] == candidate_tf.shape[-1]
    assert torch.allclose(pending_tf, expected_pending)


def test_stale_ordinal_ohe_reference_patch_is_removed() -> None:
    assert not STALE_PATCH.exists()
