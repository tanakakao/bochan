from __future__ import annotations

from types import SimpleNamespace

import torch
from botorch.optim import optimize_acqf

from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
)
from bochan.acquisition.binary.active_learning.multi_output import (
    qMultiOutputBinaryPredictiveEntropy,
)
from bochan.acquisition.multiclass.active_learning.single_output import (
    qMulticlassPredictiveEntropy,
)


class _DummyMultiOutputBinaryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.train_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)

    def probability_posterior(self, X: torch.Tensor):
        x = X[..., 0]
        p1 = torch.sigmoid(10.0 * (x - 0.45))
        p2 = torch.sigmoid(-8.0 * (x - 0.65))
        return SimpleNamespace(mean=torch.stack([p1, p2], dim=-1))


class _DummyMulticlassModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.train_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
        self.num_classes = 3

    def class_probs(self, X: torch.Tensor) -> torch.Tensor:
        x = X[..., 0]
        logits = torch.stack(
            [x, 1.0 - x, -4.0 * (x - 0.5).square()],
            dim=-1,
        )
        return torch.softmax(logits, dim=-1)


def test_duplicate_tolerance_is_euclidean_not_squared() -> None:
    tolerance = 1e-4
    inside = torch.tensor([[[0.0], [0.5e-4]]], dtype=torch.double)
    outside = torch.tensor([[[0.0], [2.0e-4]]], dtype=torch.double)

    assert torch.isinf(
        hard_same_batch_duplicate_penalty_per_point(
            inside,
            tolerance=tolerance,
        )
    ).all()
    assert torch.equal(
        hard_same_batch_duplicate_penalty_per_point(
            outside,
            tolerance=tolerance,
        ),
        torch.zeros(1, 2, dtype=torch.double),
    )
    assert torch.equal(
        hard_reference_duplicate_penalty_per_point(
            outside[..., 1:, :],
            torch.tensor([[0.0]], dtype=torch.double),
            tolerance=tolerance,
        ),
        torch.zeros(1, 1, dtype=torch.double),
    )


def test_binary_and_multiclass_resolve_observed_consistently() -> None:
    binary_model = _DummyMultiOutputBinaryModel()
    binary = qMultiOutputBinaryPredictiveEntropy(binary_model)
    assert torch.equal(binary.X_observed, binary_model.train_X)
    assert torch.isinf(binary._observed_penalty_per_point(binary_model.train_X[:1].view(1, 1, 1))).all()

    multiclass_model = _DummyMulticlassModel()
    multiclass = qMulticlassPredictiveEntropy(multiclass_model)
    assert torch.equal(multiclass.X_observed, multiclass_model.train_X)
    assert torch.isinf(multiclass._observed_penalty_per_point(multiclass_model.train_X[:1].view(1, 1, 1))).all()


def test_public_duplicate_controls_can_be_disabled() -> None:
    duplicate = torch.tensor([[[0.5], [0.5]]], dtype=torch.double)

    binary = qMultiOutputBinaryPredictiveEntropy(
        _DummyMultiOutputBinaryModel(),
        exclude_same_batch_duplicates=False,
        exclude_pending_duplicates=False,
        exclude_observed_duplicates=False,
        X_pending=torch.tensor([[0.5]], dtype=torch.double),
        X_observed=torch.tensor([[0.5]], dtype=torch.double),
    )
    assert torch.equal(
        binary._candidate_penalty_per_point(duplicate),
        torch.zeros(1, 2, dtype=torch.double),
    )

    multiclass = qMulticlassPredictiveEntropy(
        _DummyMulticlassModel(),
        exclude_same_batch_duplicates=False,
        exclude_pending_duplicates=False,
        exclude_observed_duplicates=False,
        X_pending=torch.tensor([[0.5]], dtype=torch.double),
        X_observed=torch.tensor([[0.5]], dtype=torch.double),
    )
    assert torch.equal(
        multiclass._pending_penalty_per_point(duplicate)
        + multiclass._observed_penalty_per_point(duplicate)
        + multiclass._same_batch_penalty(duplicate).unsqueeze(-1),
        torch.zeros(1, 2, dtype=torch.double),
    )


def test_optimize_acqf_multi_output_binary_avoids_exact_duplicates() -> None:
    torch.manual_seed(0)
    acquisition = qMultiOutputBinaryPredictiveEntropy(
        _DummyMultiOutputBinaryModel(),
        X_observed=torch.tensor([[0.45], [0.65]], dtype=torch.double),
        hard_duplicate_tol=1e-6,
    )
    candidates, value = optimize_acqf(
        acq_function=acquisition,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
        num_restarts=8,
        raw_samples=128,
        sequential=False,
    )
    assert candidates.shape == torch.Size([2, 1])
    assert torch.isfinite(value).all()
    assert not torch.allclose(candidates[0], candidates[1], rtol=0.0, atol=1e-6)
    for observed in acquisition.X_observed:
        assert not torch.allclose(candidates[0], observed, rtol=0.0, atol=1e-6)
        assert not torch.allclose(candidates[1], observed, rtol=0.0, atol=1e-6)


def test_optimize_acqf_multiclass_avoids_exact_duplicates() -> None:
    torch.manual_seed(1)
    acquisition = qMulticlassPredictiveEntropy(
        _DummyMulticlassModel(),
        X_observed=torch.tensor([[0.5]], dtype=torch.double),
        hard_duplicate_tol=1e-6,
    )
    candidates, value = optimize_acqf(
        acq_function=acquisition,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
        num_restarts=8,
        raw_samples=128,
        sequential=False,
    )
    assert candidates.shape == torch.Size([2, 1])
    assert torch.isfinite(value).all()
    assert not torch.allclose(candidates[0], candidates[1], rtol=0.0, atol=1e-6)
    for observed in acquisition.X_observed:
        assert not torch.allclose(candidates[0], observed, rtol=0.0, atol=1e-6)
        assert not torch.allclose(candidates[1], observed, rtol=0.0, atol=1e-6)
