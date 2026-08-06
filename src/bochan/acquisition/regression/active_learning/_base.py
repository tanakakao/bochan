"""Common regression active-learning acquisition base."""

from __future__ import annotations

from collections.abc import Callable

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from ._base_common import OutputReductionType, ReductionType
from ._base_objective import _RegressionObjectiveMixin
from ._base_reference import _RegressionReferenceMixin
from ._base_scoring import _RegressionScoringMixin


class _RegressionActiveLearningBase(
    _RegressionReferenceMixin,
    _RegressionScoringMixin,
    _RegressionObjectiveMixin,
    AcquisitionFunction,
):
    """Base class aligned with classification / ordinal active-learning APIs.

    Args:
        model:
            BoTorch-supported regression model.
        reduction:
            q-batch reduction.  This is intentionally named ``reduction`` to
            match classification / ordinal APIs.
        output_reduction:
            Reduction over output dimension for multi-output regression.
        pending_penalty_weight:
            Weight for avoiding X_pending.
        observed_penalty_weight:
            Weight for avoiding X_observed.
        same_batch_penalty_weight:
            Weight for q-batch diversity penalty.
        exclude_same_batch_duplicates:
            Hard-exclude q-batches containing duplicate candidate points.
        exclude_pending_duplicates:
            Hard-exclude q-batches containing a point already in ``X_pending``.
        exclude_observed_duplicates:
            Hard-exclude q-batches containing a point already in ``X_observed``.
        objective:
            Optional score objective.  Classification / ordinal style score
            objectives receive pointwise scores.  BoTorch MC multi-output
            objectives receive deterministic pseudo-samples.
        n_w:
            Number of input perturbation samples.  If omitted but objective has
            ``n_w``, that value is used.
    """

    def __init__(
        self,
        model,
        *,
        reduction: ReductionType = "mean",
        output_reduction: OutputReductionType = "mean",
        X_pending: Tensor | None = None,
        X_observed: Tensor | None = None,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_penalty: float = 0.0,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = False,
        objective: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        n_w: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(model=model)

        if reduction not in ("mean", "sum", "max", "min"):
            raise ValueError("reduction must be one of 'mean', 'sum', 'max', 'min'.")
        if output_reduction not in ("mean", "sum", "max", "min"):
            raise ValueError("output_reduction must be one of 'mean', 'sum', 'max', 'min'.")

        self.reduction = reduction
        self.output_reduction = output_reduction
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.hard_duplicate_penalty = float(hard_duplicate_penalty)
        self.hard_duplicate_tol = float(hard_duplicate_tol)
        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)
        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)
        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)
        if self.hard_duplicate_tol < 0.0:
            raise ValueError("hard_duplicate_tol must be non-negative.")
        self.objective = objective
        self.eps = float(eps)

        if n_w is None and objective is not None:
            n_w = getattr(objective, "n_w", None)
        self.n_w = None if n_w is None else int(n_w)
        if self.n_w is not None and self.n_w <= 0:
            raise ValueError("n_w must be positive or None.")

        self.X_pending: Tensor | None = None
        self.X_observed: Tensor | None = None
        self.set_X_pending(X_pending)
        self.set_X_observed(X_observed)
