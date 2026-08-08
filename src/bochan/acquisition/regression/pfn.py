"""PFNs4BO-native single-point acquisition functions."""

from __future__ import annotations

from typing import Any

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class PFNExpectedImprovement(AcquisitionFunction):
    """Expected improvement computed directly from the PFN bar distribution."""

    def __init__(
        self,
        model: Any,
        *,
        incumbent: Tensor | float | None = None,
        best_f: Tensor | float | None = None,
        maximize: bool = True,
    ) -> None:
        super().__init__(model=model)
        if incumbent is not None and best_f is not None:
            raise ValueError("Specify only one of incumbent and best_f.")
        self.incumbent = best_f if incumbent is None else incumbent
        self.maximize = bool(maximize)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        return self.model.native_acquisition(
            X,
            kind="ei",
            incumbent=self.incumbent,
            maximize=self.maximize,
        )


class PFNProbabilityOfImprovement(AcquisitionFunction):
    """Probability of improvement computed directly from the PFN bar distribution."""

    def __init__(
        self,
        model: Any,
        *,
        incumbent: Tensor | float | None = None,
        best_f: Tensor | float | None = None,
        maximize: bool = True,
    ) -> None:
        super().__init__(model=model)
        if incumbent is not None and best_f is not None:
            raise ValueError("Specify only one of incumbent and best_f.")
        self.incumbent = best_f if incumbent is None else incumbent
        self.maximize = bool(maximize)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        return self.model.native_acquisition(
            X,
            kind="pi",
            incumbent=self.incumbent,
            maximize=self.maximize,
        )


class PFNUpperConfidenceBound(AcquisitionFunction):
    """PFN quantile-UCB using the native bar-distribution inverse CDF."""

    def __init__(
        self,
        model: Any,
        *,
        rest_prob: float = (1.0 - 0.6826894921370859) / 2.0,
        maximize: bool = True,
    ) -> None:
        super().__init__(model=model)
        if not 0.0 < float(rest_prob) < 0.5:
            raise ValueError("rest_prob must be in (0, 0.5).")
        self.rest_prob = float(rest_prob)
        self.maximize = bool(maximize)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        return self.model.native_acquisition(
            X,
            kind="ucb",
            maximize=self.maximize,
            rest_prob=self.rest_prob,
        )


__all__ = [
    "PFNExpectedImprovement",
    "PFNProbabilityOfImprovement",
    "PFNUpperConfidenceBound",
]
