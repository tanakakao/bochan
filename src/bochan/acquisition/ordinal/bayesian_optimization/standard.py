"""BoTorch-style single-output ordinal Bayesian optimization acquisitions."""

from __future__ import annotations

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement as _qExpectedImprovement,
)
from botorch.acquisition.monte_carlo import (
    qProbabilityOfImprovement as _qProbabilityOfImprovement,
)
from botorch.acquisition.monte_carlo import qSimpleRegret as _qSimpleRegret
from botorch.acquisition.monte_carlo import (
    qUpperConfidenceBound as _qUpperConfidenceBound,
)
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from torch import Tensor

from bochan.acquisition._api import Constraints, PosteriorTransformArg, Sampler


def _require_objective(
    objective: MCAcquisitionObjective | None,
) -> MCAcquisitionObjective:
    if objective is None:
        raise ValueError(
            "objective must be provided for ordinal Bayesian optimization. "
            "Map latent ordinal posterior samples to a scalar utility with a "
            "BoTorch MCAcquisitionObjective."
        )
    return objective


class qOrdinalExpectedUtility(_qSimpleRegret):
    """Expected best ordinal utility over a joint q-batch.

    This is BoTorch qSimpleRegret applied after an explicit ordinal utility
    objective, i.e. ``E[max_q utility(Y)]``.
    """

    def __init__(
        self,
        model: Model,
        sampler: Sampler = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransformArg = None,
        X_pending: Tensor | None = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=_require_objective(objective),
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )


class qOrdinalExpectedImprovement(_qExpectedImprovement):
    """Joint qEI in ordinal utility space."""

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        sampler: Sampler = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransformArg = None,
        X_pending: Tensor | None = None,
        constraints: Constraints = None,
        eta: Tensor | float = 1e-3,
    ) -> None:
        super().__init__(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=_require_objective(objective),
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=None if constraints is None else list(constraints),
            eta=eta,
        )


class qOrdinalProbabilityOfImprovement(_qProbabilityOfImprovement):
    """Joint qPI in ordinal utility space."""

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        sampler: Sampler = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransformArg = None,
        X_pending: Tensor | None = None,
        tau: float = 1e-3,
        constraints: Constraints = None,
        eta: Tensor | float = 1e-3,
    ) -> None:
        super().__init__(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=_require_objective(objective),
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            tau=tau,
            constraints=None if constraints is None else list(constraints),
            eta=eta,
        )


class qOrdinalUpperConfidenceBound(_qUpperConfidenceBound):
    """Joint qUCB in ordinal utility space."""

    def __init__(
        self,
        model: Model,
        beta: float | Tensor,
        sampler: Sampler = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransformArg = None,
        X_pending: Tensor | None = None,
    ) -> None:
        super().__init__(
            model=model,
            beta=beta,
            sampler=sampler,
            objective=_require_objective(objective),
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )


__all__ = [
    "qOrdinalExpectedImprovement",
    "qOrdinalExpectedUtility",
    "qOrdinalProbabilityOfImprovement",
    "qOrdinalUpperConfidenceBound",
]
