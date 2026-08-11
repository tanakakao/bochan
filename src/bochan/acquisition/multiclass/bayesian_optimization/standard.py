"""BoTorch-style single-output multiclass Bayesian optimization acquisitions."""

from __future__ import annotations

from collections.abc import Sequence

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement as _qExpectedImprovement,
)
from botorch.acquisition.monte_carlo import (
    qProbabilityOfImprovement as _qProbabilityOfImprovement,
)
from botorch.acquisition.monte_carlo import (
    qUpperConfidenceBound as _qUpperConfidenceBound,
)
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from torch import Tensor

from bochan.acquisition._api import (
    ClassReduction,
    Constraints,
    PosteriorTransformArg,
    Sampler,
    reduce_class_values,
)


class MulticlassProbabilityObjective(MCAcquisitionObjective):
    """Select target-class probability from multiclass posterior samples.

    Models used with this objective are expected to expose probabilities with
    the class dimension last. The acquisition does not guess whether samples
    are logits and does not apply softmax implicitly.
    """

    def __init__(
        self,
        target_class: int | Sequence[int],
        *,
        class_reduction: ClassReduction = "mean",
    ) -> None:
        super().__init__()
        self.target_class = target_class
        self.class_reduction = class_reduction

    def forward(self, samples: Tensor, X: Tensor | None = None) -> Tensor:
        del X
        if samples.ndim < 1 or samples.shape[-1] < 2:
            raise RuntimeError(
                "Multiclass posterior samples must end in a class dimension C >= 2. "
                f"Got shape={tuple(samples.shape)}."
            )
        if isinstance(self.target_class, int):
            return samples[..., int(self.target_class)]
        indices = [int(index) for index in self.target_class]
        if not indices:
            raise ValueError("target_class must contain at least one class index.")
        return reduce_class_values(samples[..., indices], self.class_reduction)


def _resolve_objective(
    *,
    objective: MCAcquisitionObjective | None,
    target_class: int | Sequence[int] | None,
    class_reduction: ClassReduction,
) -> MCAcquisitionObjective:
    if objective is not None:
        return objective
    if target_class is None:
        raise ValueError("target_class or objective must be provided for multiclass BO.")
    return MulticlassProbabilityObjective(
        target_class=target_class,
        class_reduction=class_reduction,
    )


class qMulticlassExpectedImprovement(_qExpectedImprovement):
    """Joint qEI on a multiclass probability objective."""

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
        *,
        target_class: int | Sequence[int] | None = None,
        class_reduction: ClassReduction = "mean",
    ) -> None:
        super().__init__(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=_resolve_objective(
                objective=objective,
                target_class=target_class,
                class_reduction=class_reduction,
            ),
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=None if constraints is None else list(constraints),
            eta=eta,
        )


class qMulticlassProbabilityOfImprovement(_qProbabilityOfImprovement):
    """Joint qPI on a multiclass probability objective."""

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
        *,
        target_class: int | Sequence[int] | None = None,
        class_reduction: ClassReduction = "mean",
    ) -> None:
        super().__init__(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=_resolve_objective(
                objective=objective,
                target_class=target_class,
                class_reduction=class_reduction,
            ),
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            tau=tau,
            constraints=None if constraints is None else list(constraints),
            eta=eta,
        )


class qMulticlassUpperConfidenceBound(_qUpperConfidenceBound):
    """Joint qUCB on a multiclass probability objective."""

    def __init__(
        self,
        model: Model,
        beta: float | Tensor,
        sampler: Sampler = None,
        objective: MCAcquisitionObjective | None = None,
        posterior_transform: PosteriorTransformArg = None,
        X_pending: Tensor | None = None,
        *,
        target_class: int | Sequence[int] | None = None,
        class_reduction: ClassReduction = "mean",
    ) -> None:
        super().__init__(
            model=model,
            beta=beta,
            sampler=sampler,
            objective=_resolve_objective(
                objective=objective,
                target_class=target_class,
                class_reduction=class_reduction,
            ),
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )


__all__ = [
    "MulticlassProbabilityObjective",
    "qMulticlassExpectedImprovement",
    "qMulticlassProbabilityOfImprovement",
    "qMulticlassUpperConfidenceBound",
]
