"""Acquisition function name resolver for the high-level API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable


_ACQF_ALIASES: dict[str, tuple[str, str]] = {
    # Single-objective Monte Carlo acquisitions
    "qei": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "qexpectedimprovement": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "ei": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "qlogei": ("botorch.acquisition.logei", "qLogExpectedImprovement"),
    "qlogexpectedimprovement": ("botorch.acquisition.logei", "qLogExpectedImprovement"),
    "logei": ("botorch.acquisition.logei", "qLogExpectedImprovement"),
    "qnei": ("botorch.acquisition.monte_carlo", "qNoisyExpectedImprovement"),
    "qnoisyexpectedimprovement": ("botorch.acquisition.monte_carlo", "qNoisyExpectedImprovement"),
    "nei": ("botorch.acquisition.monte_carlo", "qNoisyExpectedImprovement"),
    "qucb": ("botorch.acquisition.monte_carlo", "qUpperConfidenceBound"),
    "qupperconfidencebound": ("botorch.acquisition.monte_carlo", "qUpperConfidenceBound"),
    "ucb": ("botorch.acquisition.monte_carlo", "qUpperConfidenceBound"),
    "qpi": ("botorch.acquisition.monte_carlo", "qProbabilityOfImprovement"),
    "qprobabilityofimprovement": ("botorch.acquisition.monte_carlo", "qProbabilityOfImprovement"),
    "pi": ("botorch.acquisition.monte_carlo", "qProbabilityOfImprovement"),

    # Knowledge-gradient / lookahead
    "qkg": ("botorch.acquisition.knowledge_gradient", "qKnowledgeGradient"),
    "qknowledgegradient": ("botorch.acquisition.knowledge_gradient", "qKnowledgeGradient"),
    "kg": ("botorch.acquisition.knowledge_gradient", "qKnowledgeGradient"),
    "qmultisteplookahead": ("botorch.acquisition.multi_step_lookahead", "qMultiStepLookahead"),
    "multisteplookahead": ("botorch.acquisition.multi_step_lookahead", "qMultiStepLookahead"),

    # Multi-objective hypervolume acquisitions
    "qehvi": ("botorch.acquisition.multi_objective.monte_carlo", "qExpectedHypervolumeImprovement"),
    "qexpectedhypervolumeimprovement": ("botorch.acquisition.multi_objective.monte_carlo", "qExpectedHypervolumeImprovement"),
    "ehvi": ("botorch.acquisition.multi_objective.monte_carlo", "qExpectedHypervolumeImprovement"),
    "qnehvi": ("botorch.acquisition.multi_objective.monte_carlo", "qNoisyExpectedHypervolumeImprovement"),
    "qnoisyexpectedhypervolumeimprovement": ("botorch.acquisition.multi_objective.monte_carlo", "qNoisyExpectedHypervolumeImprovement"),
    "nehvi": ("botorch.acquisition.multi_objective.monte_carlo", "qNoisyExpectedHypervolumeImprovement"),

    # Scalarized multi-objective convenience alias.
    # The acquisition class is qExpectedImprovement; scalarization objective is still configured separately.
    "qnparego": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    "nparego": ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
}


def _normalize_acqf_name(name: str) -> str:
    return str(name).replace("_", "").replace("-", "").replace(" ", "").lower()


def _import_from_path(module_name: str, attr_name: str) -> Any:
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def resolve_acqf_cls(
    name: str,
    acquisition_registry: Mapping[str, Any] | None = None,
) -> type | Callable[..., Any]:
    """Resolve an acquisition class from a string name.

    Args:
        name: Acquisition name, e.g. ``"qEI"`` or ``"qExpectedImprovement"``.
        acquisition_registry: Optional user registry. Values can be classes/functions or
            ``(module_name, attr_name)`` tuples.
    """
    normalized = _normalize_acqf_name(name)

    if acquisition_registry is not None:
        if name in acquisition_registry:
            value = acquisition_registry[name]
        elif normalized in acquisition_registry:
            value = acquisition_registry[normalized]
        else:
            value = None

        if value is not None:
            if isinstance(value, tuple) and len(value) == 2:
                return _import_from_path(value[0], value[1])
            return value

    if normalized not in _ACQF_ALIASES:
        available = sorted(_ACQF_ALIASES)
        raise ValueError(
            f"Unknown acquisition function name: {name!r}. "
            f"Available built-in aliases include: {available}. "
            "For custom acquisitions, pass acquisition_registry."
        )

    module_name, attr_name = _ACQF_ALIASES[normalized]
    return _import_from_path(module_name, attr_name)


__all__ = ["resolve_acqf_cls"]
