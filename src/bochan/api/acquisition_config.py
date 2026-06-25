"""Public acquisition configuration with common defaults."""

from __future__ import annotations

from dataclasses import dataclass

from .configs import AcquisitionConfig as _BaseAcquisitionConfig


_DEFAULT_UCB_BETA = 3.0
_UCB_NAMES = {"ucb", "qucb", "upperconfidencebound", "qupperconfidencebound"}


def _normalize_acquisition_name(name: str) -> str:
    return str(name).replace("_", "").replace("-", "").replace(" ", "").lower()


@dataclass
class AcquisitionConfig(_BaseAcquisitionConfig):
    """High-level acquisition configuration.

    UCB aliases use ``beta=3.0`` when ``acqf_kwargs`` does not explicitly
    provide a beta value. Explicit user configuration always takes priority.
    """

    def __post_init__(self) -> None:
        if _normalize_acquisition_name(self.name) not in _UCB_NAMES:
            return
        if "beta" in self.acqf_kwargs:
            return
        self.acqf_kwargs = {**self.acqf_kwargs, "beta": _DEFAULT_UCB_BETA}


__all__ = ["AcquisitionConfig"]
