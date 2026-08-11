"""BoTorch-style single-output binary Bayesian optimization acquisitions."""

from __future__ import annotations

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement as _qExpectedImprovement,
)
from botorch.acquisition.monte_carlo import (
    qProbabilityOfImprovement as _qProbabilityOfImprovement,
)
from botorch.acquisition.monte_carlo import (
    qUpperConfidenceBound as _qUpperConfidenceBound,
)
from botorch.models.model import Model
from torch import Tensor

from bochan.acquisition._api import (
    Constraints,
    MCObjective,
    PosteriorTransformArg,
    Sampler,
)
from bochan.acquisition.binary.epistemic import as_epistemic_probability_model


def _probability_model(model: Model) -> Model:
    """Expose binary epistemic uncertainty in probability space."""

    return as_epistemic_probability_model(model)


class qBinaryExpectedImprovement(_qExpectedImprovement):
    """Joint qEI on positive-class probability.

    ``best_f`` is required and is interpreted in the objective space. The
    constructor intentionally mirrors BoTorch qEI and does not infer baselines,
    switch between pointwise/joint q semantics, or add duplicate penalties.
    """

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        sampler: Sampler = None,
        objective: MCObjective = None,
        posterior_transform: PosteriorTransformArg = None,
        X_pending: Tensor | None = None,
        constraints: Constraints = None,
        eta: Tensor | float = 1e-3,
    ) -> None:
        super().__init__(
            model=_probability_model(model),
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=None if constraints is None else list(constraints),
            eta=eta,
        )


class qBinaryProbabilityOfImprovement(_qProbabilityOfImprovement):
    """Joint qPI on positive-class probability."""

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        sampler: Sampler = None,
        objective: MCObjective = None,
        posterior_transform: PosteriorTransformArg = None,
        X_pending: Tensor | None = None,
        tau: float = 1e-3,
        constraints: Constraints = None,
        eta: Tensor | float = 1e-3,
    ) -> None:
        super().__init__(
            model=_probability_model(model),
            best_f=best_f,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            tau=tau,
            constraints=None if constraints is None else list(constraints),
            eta=eta,
        )


class qBinaryUpperConfidenceBound(_qUpperConfidenceBound):
    """Joint qUCB on positive-class probability."""

    def __init__(
        self,
        model: Model,
        beta: float | Tensor,
        sampler: Sampler = None,
        objective: MCObjective = None,
        posterior_transform: PosteriorTransformArg = None,
        X_pending: Tensor | None = None,
    ) -> None:
        super().__init__(
            model=_probability_model(model),
            beta=beta,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )


__all__ = [
    "qBinaryExpectedImprovement",
    "qBinaryProbabilityOfImprovement",
    "qBinaryUpperConfidenceBound",
]
