"""Public learning configuration with convenience MLL parameters."""

from __future__ import annotations

from dataclasses import dataclass

from .base import FitConfig as _BaseFitConfig


@dataclass
class FitConfig(_BaseFitConfig):
    """High-level learning configuration.

    ``beta`` is a convenience alias for ``mll_kwargs["beta"]``. It is useful
    for variational models such as DeepGP and DeepKernel classifiers, while
    keeping ``mll_kwargs`` available for advanced configuration.

    When both are specified, an explicit value in ``mll_kwargs`` takes
    precedence.
    """

    beta: float | None = None

    def __post_init__(self) -> None:
        if self.beta is None or "beta" in self.mll_kwargs:
            return
        self.mll_kwargs = {**self.mll_kwargs, "beta": float(self.beta)}
