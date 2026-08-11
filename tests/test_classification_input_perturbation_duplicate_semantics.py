from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("botorch")

from bochan.acquisition._duplicate_exclusion import (  # noqa: E402
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
)
from bochan.acquisition._nominal_duplicate_penalties import (  # noqa: E402
    NominalDuplicatePenaltyMixin,
)


class _ExpandedPenaltyBase:
    """Minimal legacy-style base that hard-excludes transformed duplicates."""

    def __init__(self) -> None:
        self.exclude_same_batch_duplicates = True
        self.exclude_pending_duplicates = True
        self.exclude_observed_duplicates = True
        self.hard_duplicate_tol = 1e-8
        self.X_pending = None
        self.X_observed = None

    def _pending_penalty_per_point(self, X):
        return hard_reference_duplicate_penalty_per_point(
            X,
            self.X_pending,
            enabled=self.exclude_pending_duplicates,
            tolerance=self.hard_duplicate_tol,
        )

    def _observed_penalty_per_point(self, X):
        return hard_reference_duplicate_penalty_per_point(
            X,
            self.X_observed,
            enabled=self.exclude_observed_duplicates,
            tolerance=self.hard_duplicate_tol,
        )

    def _same_batch_duplicate_penalty_per_point(self, X):
        return hard_same_batch_duplicate_penalty_per_point(
            X,
            enabled=self.exclude_same_batch_duplicates,
            tolerance=self.hard_duplicate_tol,
        )

    def _pointwise_repulsion_penalty(self, X):
        return (
            self._pending_penalty_per_point(X)
            + self._observed_penalty_per_point(X)
            + self._same_batch_duplicate_penalty_per_point(X)
        )

    def forward(self, X):
        expanded = X.repeat_interleave(4, dim=-2)
        return self._pointwise_repulsion_penalty(expanded)


class _SafeExpandedPenalty(NominalDuplicatePenaltyMixin, _ExpandedPenaltyBase):
    pass


class _JointExpandedPenaltyBase(_ExpandedPenaltyBase):
    """Mimic joint LSE classes with an additional numeric hard penalty."""

    def __init__(self) -> None:
        super().__init__()
        self.hard_duplicate_penalty = 1e6

    def _joint_repulsion_penalty(self, X):
        q = X.shape[-2]
        distance = torch.cdist(X, X)
        eye = torch.eye(q, dtype=torch.bool, device=X.device)
        distance = distance.masked_fill(eye, float("inf"))
        numeric_hard = self.hard_duplicate_penalty * (
            distance.min(dim=-1).values <= self.hard_duplicate_tol
        ).to(X.dtype).sum(dim=-1)
        common_hard = self._pointwise_repulsion_penalty(X).sum(dim=-1)
        return numeric_hard + common_hard

    def forward(self, X):
        expanded = X.repeat_interleave(4, dim=-2)
        return self._joint_repulsion_penalty(expanded)


class _SafeJointExpandedPenalty(
    NominalDuplicatePenaltyMixin,
    _JointExpandedPenaltyBase,
):
    pass


def test_perturbation_replicas_are_not_hard_duplicates() -> None:
    acqf = _SafeExpandedPenalty()
    raw_X = torch.tensor([[[0.2], [0.8]]], dtype=torch.double)

    penalty = acqf.forward(raw_X)

    assert penalty.shape == torch.Size([1, 8])
    assert torch.isfinite(penalty).all()


def test_true_nominal_same_batch_duplicate_remains_excluded() -> None:
    acqf = _SafeExpandedPenalty()
    raw_X = torch.tensor([[[0.5], [0.5]]], dtype=torch.double)

    penalty = acqf.forward(raw_X)

    assert torch.isposinf(penalty).all()


def test_true_nominal_pending_duplicate_remains_excluded() -> None:
    acqf = _SafeExpandedPenalty()
    acqf.X_pending = torch.tensor([[0.5]], dtype=torch.double)

    duplicate = acqf.forward(torch.tensor([[[0.5]]], dtype=torch.double))
    distinct = acqf.forward(torch.tensor([[[0.6]]], dtype=torch.double))

    assert torch.isposinf(duplicate).all()
    assert torch.isfinite(distinct).all()


def test_joint_numeric_hard_penalty_ignores_perturbation_replicas() -> None:
    acqf = _SafeJointExpandedPenalty()

    distinct = acqf.forward(
        torch.tensor([[[0.2], [0.8]]], dtype=torch.double)
    )
    duplicate = acqf.forward(
        torch.tensor([[[0.5], [0.5]]], dtype=torch.double)
    )

    assert torch.isfinite(distinct).all()
    assert torch.isposinf(duplicate).all()


def test_score_based_classification_acquisitions_keep_nominal_duplicate_mixin() -> None:
    from bochan.acquisition.binary.active_learning import qBinaryProbabilityVariance
    from bochan.acquisition.binary.bayesian_optimization import (
        qBinaryProbabilityOfFeasibility,
    )
    from bochan.acquisition.binary.levelset_estimation import (
        qBinaryLatentStraddleAcquisition,
    )
    from bochan.acquisition.multiclass.active_learning import (
        qMulticlassProbabilityVariance,
    )
    from bochan.acquisition.multiclass.bayesian_optimization import (
        qMulticlassProbabilityOfFeasibility,
    )
    from bochan.acquisition.multiclass.levelset_estimation import (
        qMulticlassLatentStraddleAcquisition,
    )
    from bochan.acquisition.ordinal.active_learning import qOrdinalUtilityVariance
    from bochan.acquisition.ordinal.levelset_estimation import (
        qOrdinalLatentStraddleAcquisition,
    )

    classes = [
        qBinaryProbabilityVariance,
        qBinaryProbabilityOfFeasibility,
        qBinaryLatentStraddleAcquisition,
        qMulticlassProbabilityVariance,
        qMulticlassProbabilityOfFeasibility,
        qMulticlassLatentStraddleAcquisition,
        qOrdinalUtilityVariance,
        qOrdinalLatentStraddleAcquisition,
    ]

    for cls in classes:
        assert issubclass(cls, NominalDuplicatePenaltyMixin), cls


def test_standard_classification_bo_does_not_own_duplicate_penalties() -> None:
    from bochan.acquisition.binary.bayesian_optimization import (
        qBinaryExpectedImprovement,
        qBinaryProbabilityOfImprovement,
        qBinaryUpperConfidenceBound,
    )
    from bochan.acquisition.multiclass.bayesian_optimization import (
        qMulticlassExpectedImprovement,
        qMulticlassProbabilityOfImprovement,
        qMulticlassUpperConfidenceBound,
    )
    from bochan.acquisition.ordinal.bayesian_optimization import (
        qOrdinalExpectedImprovement,
        qOrdinalProbabilityOfImprovement,
        qOrdinalUpperConfidenceBound,
    )

    classes = [
        qBinaryExpectedImprovement,
        qBinaryProbabilityOfImprovement,
        qBinaryUpperConfidenceBound,
        qMulticlassExpectedImprovement,
        qMulticlassProbabilityOfImprovement,
        qMulticlassUpperConfidenceBound,
        qOrdinalExpectedImprovement,
        qOrdinalProbabilityOfImprovement,
        qOrdinalUpperConfidenceBound,
    ]

    for cls in classes:
        assert not issubclass(cls, NominalDuplicatePenaltyMixin), cls
