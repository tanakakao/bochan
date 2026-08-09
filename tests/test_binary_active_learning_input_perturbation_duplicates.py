from __future__ import annotations

import torch
from botorch.models.model import Model
from torch import Tensor

from bochan.acquisition.binary.active_learning import qBinaryProbabilityVariance


class _RepeatedBinaryEvaluationModel(Model):
    """Minimal binary model whose transform repeats every nominal candidate."""

    def __init__(self, n_w: int = 4) -> None:
        super().__init__()
        self.n_w = int(n_w)

    @property
    def num_outputs(self) -> int:
        return 1

    def transform_inputs(self, X: Tensor, input_transform=None) -> Tensor:
        del input_transform
        return X.repeat_interleave(self.n_w, dim=-2)

    def posterior(self, X: Tensor, *args, **kwargs):
        del X, args, kwargs
        raise AssertionError("Penalty-only regression test must not request posterior.")


def _acquisition(**kwargs) -> qBinaryProbabilityVariance:
    return qBinaryProbabilityVariance(
        model=_RepeatedBinaryEvaluationModel(n_w=4),
        exclude_observed_duplicates=False,
        **kwargs,
    )


def _penalty(acqf: qBinaryProbabilityVariance, raw_X: Tensor) -> Tensor:
    Xt = acqf.model.transform_inputs(raw_X)
    acqf._raw_X_for_duplicate_penalty = raw_X
    try:
        return acqf._candidate_penalty_per_point(Xt)
    finally:
        acqf._raw_X_for_duplicate_penalty = None


def test_binary_perturbation_replicas_are_not_same_batch_duplicates() -> None:
    """Repeated q*n_w uncertainty rows must not invalidate nominal candidates."""

    acqf = _acquisition()
    raw_X = torch.tensor(
        [
            [[0.2], [0.8]],
            [[0.1], [0.9]],
        ],
        dtype=torch.double,
    )

    penalty = _penalty(acqf, raw_X)

    assert penalty.shape == torch.Size([2, 8])
    assert torch.isfinite(penalty).all()
    torch.testing.assert_close(penalty, torch.zeros_like(penalty))


def test_binary_raw_same_batch_duplicates_remain_excluded() -> None:
    """True duplicate nominal q candidates must still invalidate the q-batch."""

    acqf = _acquisition()
    raw_X = torch.tensor([[[0.5], [0.5]]], dtype=torch.double)

    penalty = _penalty(acqf, raw_X)

    assert torch.isposinf(penalty).all()


def test_binary_raw_pending_duplicate_remains_excluded() -> None:
    """Pending hard exclusion must compare raw candidates, not perturbation rows."""

    acqf = _acquisition()
    acqf.set_X_pending(torch.tensor([[0.5]], dtype=torch.double))

    duplicate = _penalty(acqf, torch.tensor([[[0.5]]], dtype=torch.double))
    distinct = _penalty(acqf, torch.tensor([[[0.6]]], dtype=torch.double))

    assert torch.isposinf(duplicate).all()
    assert torch.isfinite(distinct).all()
