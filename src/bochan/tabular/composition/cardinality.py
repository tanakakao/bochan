"""Cardinality helpers for composition-aware Best Subset search."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Mapping

BEST_SUBSET_MIN_K_KWARG = "best_subset_min_k"
BEST_SUBSET_MAX_K_KWARG = "best_subset_max_k"


@dataclass(frozen=True)
class CompositionCardinalityRange:
    """Resolved total and optional active-component cardinality ranges."""

    minimum: int
    maximum: int
    optional_minimum: int
    optional_maximum: int

    @property
    def exact(self) -> bool:
        return self.minimum == self.maximum

    @property
    def optional_exact(self) -> bool:
        return self.optional_minimum == self.optional_maximum

    @property
    def optional_cardinalities(self) -> tuple[int, ...]:
        return tuple(range(self.optional_minimum, self.optional_maximum + 1))


def resolve_composition_cardinality_range(
    config: Mapping[str, Any],
    *,
    required_count: int,
    optional_count: int,
    context: str = "Composition best_subset",
) -> CompositionCardinalityRange:
    """Resolve site-level total cardinality to the optional sparse group.

    ``min_components`` / ``max_components`` count every active element. Elements
    that are required by configuration, positive lower bounds, or non-zero fixed
    values live outside the generic sparse group, so the core Best Subset engine
    receives the residual optional-cardinality range.
    """

    minimum = int(config.get("min_components", 1))
    maximum_raw = config.get("max_components")
    if minimum < 1:
        raise ValueError(f"{context} requires min_components >= 1.")
    if maximum_raw is None:
        raise ValueError(
            f"{context} requires max_components so the support-search range is finite."
        )
    maximum = int(maximum_raw)
    if maximum < minimum:
        raise ValueError(
            f"{context} requires max_components >= min_components."
        )
    if required_count < 0 or optional_count < 0:
        raise ValueError("required_count and optional_count must be non-negative.")
    if required_count > maximum:
        raise ValueError(
            f"{context} requires {required_count} components after required/fixed "
            f"rules, exceeding max_components={maximum}."
        )

    available = required_count + optional_count
    effective_minimum = max(minimum, required_count)
    effective_maximum = min(maximum, available)
    if effective_minimum > effective_maximum:
        raise ValueError(
            f"{context} cannot satisfy the requested component range "
            f"[{minimum}, {maximum}] with {required_count} required and "
            f"{optional_count} optional components."
        )

    return CompositionCardinalityRange(
        minimum=effective_minimum,
        maximum=effective_maximum,
        optional_minimum=effective_minimum - required_count,
        optional_maximum=effective_maximum - required_count,
    )


def apply_optional_cardinality_range(
    optimizer_kwargs: Mapping[str, Any] | None,
    cardinality: CompositionCardinalityRange,
    *,
    context: str = "Composition best_subset",
) -> dict[str, Any]:
    """Attach the composition-owned optional-k range to generic Best Subset."""

    result = dict(optimizer_kwargs or {})
    expected = {
        BEST_SUBSET_MIN_K_KWARG: cardinality.optional_minimum,
        BEST_SUBSET_MAX_K_KWARG: cardinality.optional_maximum,
    }
    for key, value in expected.items():
        if key in result and int(result[key]) != int(value):
            raise ValueError(
                f"{context} derives {key}={value} from min_components/max_components; "
                f"remove the conflicting explicit optimizer value {result[key]!r}."
            )
        result[key] = int(value)
    return result


def support_count(optional_count: int, cardinality: CompositionCardinalityRange) -> int:
    """Return the number of optional supports across the resolved range."""

    return sum(
        comb(optional_count, k)
        for k in cardinality.optional_cardinalities
        if 0 <= k <= optional_count
    )


def require_exact_cardinality_for_steps(
    config: Mapping[str, Any],
    cardinality: CompositionCardinalityRange,
    *,
    context: str = "Composition best_subset",
) -> None:
    """Keep current MILP step projectors on exact-cardinality supports."""

    if config.get("steps") and not cardinality.exact:
        raise ValueError(
            f"{context} with component steps currently requires "
            "min_components == max_components. Remove steps or use an exact "
            "component count."
        )


__all__ = [
    "BEST_SUBSET_MAX_K_KWARG",
    "BEST_SUBSET_MIN_K_KWARG",
    "CompositionCardinalityRange",
    "apply_optional_cardinality_range",
    "require_exact_cardinality_for_steps",
    "resolve_composition_cardinality_range",
    "support_count",
]
