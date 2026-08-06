from __future__ import annotations

import pytest
import torch

from bochan.acquisition.regression.active_learning.single_output import (
    qRegressionPosteriorVariance,
)


def _acquisition(**kwargs) -> qRegressionPosteriorVariance:
    return qRegressionPosteriorVariance(model=torch.nn.Identity(), **kwargs)


def test_same_batch_duplicates_are_hard_excluded_by_default() -> None:
    acquisition = _acquisition()
    duplicate_batch = torch.tensor(
        [[[0.2], [0.2], [0.4]]],
        dtype=torch.double,
    )
    unique_batch = torch.tensor(
        [[[0.2], [0.3], [0.4]]],
        dtype=torch.double,
    )

    duplicate_penalty = acquisition._same_batch_penalty_per_point(duplicate_batch)
    unique_penalty = acquisition._same_batch_penalty_per_point(unique_batch)

    assert torch.isinf(duplicate_penalty).all()
    assert torch.equal(unique_penalty, torch.zeros_like(unique_penalty))


def test_pending_duplicate_is_hard_excluded_without_soft_weight() -> None:
    acquisition = _acquisition(X_pending=torch.tensor([[0.2]], dtype=torch.double))
    duplicate = torch.tensor([[[0.2]]], dtype=torch.double)
    nearby_distinct = torch.tensor([[[0.2002]]], dtype=torch.double)

    duplicate_penalty = acquisition._reference_penalty_per_point(
        duplicate,
        acquisition.X_pending,
        weight=acquisition.pending_penalty_weight,
        beta=acquisition.pending_penalty_beta,
        exclude_duplicates=acquisition.exclude_pending_duplicates,
    )
    distinct_penalty = acquisition._reference_penalty_per_point(
        nearby_distinct,
        acquisition.X_pending,
        weight=acquisition.pending_penalty_weight,
        beta=acquisition.pending_penalty_beta,
        exclude_duplicates=acquisition.exclude_pending_duplicates,
    )

    assert torch.isinf(duplicate_penalty).all()
    assert torch.equal(distinct_penalty, torch.zeros_like(distinct_penalty))


def test_hard_exclusion_can_be_disabled_without_enabling_soft_penalty() -> None:
    acquisition = _acquisition(
        X_pending=torch.tensor([[0.2]], dtype=torch.double),
        exclude_same_batch_duplicates=False,
        exclude_pending_duplicates=False,
    )

    same_batch = acquisition._same_batch_penalty_per_point(
        torch.tensor([[[0.2], [0.2]]], dtype=torch.double)
    )
    pending = acquisition._reference_penalty_per_point(
        torch.tensor([[[0.2]]], dtype=torch.double),
        acquisition.X_pending,
        weight=0.0,
        beta=10.0,
        exclude_duplicates=False,
    )

    assert torch.equal(same_batch, torch.zeros_like(same_batch))
    assert torch.equal(pending, torch.zeros_like(pending))


def test_finite_hard_duplicate_penalty_is_independent_of_soft_weight() -> None:
    acquisition = _acquisition(
        hard_duplicate_penalty=7.0,
        same_batch_penalty_weight=0.0,
        exclude_same_batch_duplicates=False,
    )

    penalty = acquisition._same_batch_penalty_per_point(
        torch.tensor([[[0.2], [0.2]]], dtype=torch.double)
    )

    assert torch.equal(penalty, torch.full_like(penalty, 7.0))


def test_duplicate_tolerance_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="hard_duplicate_tol"):
        _acquisition(hard_duplicate_tol=-1.0)
