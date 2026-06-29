"""Public acquisition configuration with common defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .configs import AcquisitionConfig as _BaseAcquisitionConfig


_DEFAULT_UCB_BETA = 3.0
_UCB_NAMES = {"ucb", "qucb", "upperconfidencebound", "qupperconfidencebound"}


def _normalize_acquisition_name(name: str) -> str:
    return str(name).replace("_", "").replace("-", "").replace(" ", "").lower()


@dataclass
class AcquisitionConfig(_BaseAcquisitionConfig):
    """High-level acquisition configuration.

    Args:
        constraints: BoTorch outcome-constraint callables. Each callable receives
            posterior samples with shape ``sample_shape x batch_shape x q x m``
            and must return ``sample_shape x batch_shape x q``. A constraint is
            satisfied when its output is less than or equal to zero.

    Notes:
        ``constraints`` is a first-class acquisition setting, parallel to
        ``objective``. It is forwarded to the acquisition constructor through
        ``acqf_kwargs`` because BoTorch acquisition classes expose it as a direct
        constructor keyword.

        UCB aliases use ``beta=3.0`` when ``acqf_kwargs`` does not explicitly
        provide a beta value. Explicit user configuration always takes priority.
    """

    constraints: list[Any] | None = None

    def __post_init__(self) -> None:
        kwargs = dict(self.acqf_kwargs)

        if "constraints" in kwargs:
            raise ValueError(
                "Pass outcome constraints through AcquisitionConfig.constraints, "
                "not acqf_kwargs['constraints']."
            )
        if self.constraints is not None:
            kwargs["constraints"] = self.constraints

        if (
            _normalize_acquisition_name(self.name) in _UCB_NAMES
            and "beta" not in kwargs
        ):
            kwargs["beta"] = _DEFAULT_UCB_BETA

        self.acqf_kwargs = kwargs


__all__ = ["AcquisitionConfig"]
