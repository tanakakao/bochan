from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
    resolve_observed_X,
)
from bochan.acquisition.binary.base import _BinaryClassificationAcqBase
from bochan.acquisition.multiclass.bayesian_optimization.single_output import (
    _MulticlassProbabilityBOBase,
)


class _ConcreteBinary(_BinaryClassificationAcqBase):
    def forward(self, X):
        return X.sum(dim=(-1, -2))


class _ConcreteMulticlass(_MulticlassProbabilityBOBase):
    def forward(self, X):
        return X.sum(dim=(-1, -2))


def test_shared_hard_duplicate_helpers_only_reject_duplicates() -> None:
    duplicate = torch.tensor([[[0.2], [0.2], [0.4]]], dtype=torch.double)
    distinct = torch.tensor([[[0.2], [0.2002], [0.4]]], dtype=torch.double)
    pending = torch.tensor([[0.2]], dtype=torch.double)

    assert torch.isinf(hard_same_batch_duplicate_penalty_per_point(duplicate)).all()
    assert torch.equal(
        hard_same_batch_duplicate_penalty_per_point(distinct),
        torch.zeros(1, 3, dtype=torch.double),
    )
    assert torch.isinf(hard_reference_duplicate_penalty_per_point(duplicate[..., :1, :], pending)).all()
    assert torch.equal(
        hard_reference_duplicate_penalty_per_point(distinct[..., 1:2, :], pending),
        torch.zeros(1, 1, dtype=torch.double),
    )


def test_resolve_observed_x_prefers_public_wide_inputs() -> None:
    wide = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.double)
    long = torch.tensor(
        [[0.1, 0.2, 0.0], [0.1, 0.2, 1.0], [0.3, 0.4, 0.0]],
        dtype=torch.double,
    )
    model = SimpleNamespace(
        train_X_wide=wide,
        train_inputs_raw=(long,),
        train_inputs=(long,),
    )

    resolved = resolve_observed_X(model)

    assert resolved is wide
    assert resolved.shape[-1] == 2


def test_binary_defaults_hard_exclude_same_batch_and_pending() -> None:
    acquisition = _ConcreteBinary(model=torch.nn.Identity())
    duplicate = torch.tensor([[[0.2], [0.2]]], dtype=torch.double)
    distinct = torch.tensor([[[0.2], [0.3]]], dtype=torch.double)

    assert acquisition.pending_penalty_weight == 0.0
    assert torch.isinf(acquisition._candidate_penalty_per_point(duplicate)).all()
    assert torch.equal(
        acquisition._candidate_penalty_per_point(distinct),
        torch.zeros_like(distinct[..., 0]),
    )

    acquisition.set_X_pending(torch.tensor([[0.2]], dtype=torch.double))
    assert torch.isinf(acquisition._candidate_penalty_per_point(torch.tensor([[[0.2]]], dtype=torch.double))).all()


def test_multiclass_defaults_hard_exclude_same_batch_and_pending() -> None:
    acquisition = _ConcreteMulticlass(model=torch.nn.Identity())
    duplicate = torch.tensor([[[0.2], [0.2]]], dtype=torch.double)
    distinct = torch.tensor([[[0.2], [0.3]]], dtype=torch.double)

    assert acquisition.same_batch_penalty_weight == 0.0
    assert torch.isinf(acquisition._same_batch_penalty(duplicate)).all()
    assert torch.equal(
        acquisition._same_batch_penalty(distinct),
        torch.zeros(1, dtype=torch.double),
    )

    acquisition.set_X_pending(torch.tensor([[0.2]], dtype=torch.double))
    assert torch.isinf(acquisition._pending_penalty_per_point(torch.tensor([[[0.2]]], dtype=torch.double))).all()


def test_hard_duplicate_exclusion_can_be_disabled() -> None:
    acquisition = _ConcreteBinary(
        model=torch.nn.Identity(),
        exclude_same_batch_duplicates=False,
        exclude_pending_duplicates=False,
    )
    acquisition.set_X_pending(torch.tensor([[0.2]], dtype=torch.double))
    duplicate = torch.tensor([[[0.2], [0.2]]], dtype=torch.double)

    assert torch.equal(
        acquisition._candidate_penalty_per_point(duplicate),
        torch.zeros(1, 2, dtype=torch.double),
    )
