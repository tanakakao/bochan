"""Composition-weighted elemental descriptors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .formula import ATOMIC_NUMBERS, ATOMIC_WEIGHTS
from .simplex import close_compositions

_BUILTIN_PROPERTIES: dict[str, dict[str, float]] = {
    "atomic_number": {element: float(value) for element, value in ATOMIC_NUMBERS.items()},
    "atomic_weight": {element: float(value) for element, value in ATOMIC_WEIGHTS.items()},
}


@dataclass
class CompositionDescriptorCalculator:
    """Calculate Magpie-like weighted statistics from element fractions.

    Args:
        properties: Property names to calculate. Built-ins are ``atomic_number``
            and ``atomic_weight``. Custom property maps can be supplied with
            ``element_properties``.
        statistics: Any of ``mean``, ``std``, ``min``, ``max``, and ``range``.
        include_num_elements: Add the number of non-zero components.
        include_mixing_entropy: Add ``-sum(x * log(x))``.
        missing: ``error``, ``nan``, or ``ignore`` for missing elemental values.
        element_properties: Additional mapping ``property -> element -> value``.
    """

    properties: Sequence[str] = ("atomic_number", "atomic_weight")
    statistics: Sequence[str] = ("mean", "std", "min", "max", "range")
    include_num_elements: bool = True
    include_mixing_entropy: bool = True
    missing: str = "error"
    element_properties: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = {"mean", "std", "min", "max", "range"}
        unknown = set(self.statistics) - allowed
        if unknown:
            raise ValueError(f"Unknown descriptor statistics: {sorted(unknown)!r}.")
        if self.missing not in {"error", "nan", "ignore"}:
            raise ValueError("missing must be one of 'error', 'nan', or 'ignore'.")

    @property
    def property_table(self) -> dict[str, dict[str, float]]:
        """Return built-in and user-supplied elemental property mappings."""

        table = {name: dict(values) for name, values in _BUILTIN_PROPERTIES.items()}
        for name, values in self.element_properties.items():
            table[str(name)] = {str(element): float(value) for element, value in values.items()}
        return table

    def feature_names(self) -> list[str]:
        """Return output descriptor names in deterministic order."""

        names = [
            f"descriptor__{property_name}__{statistic}"
            for property_name in self.properties
            for statistic in self.statistics
        ]
        if self.include_num_elements:
            names.append("descriptor__num_elements")
        if self.include_mixing_entropy:
            names.append("descriptor__mixing_entropy")
        return names

    def transform(self, values: Any, elements: Sequence[str]) -> np.ndarray:
        """Transform element-fraction rows into weighted descriptors."""

        fractions = close_compositions(values)
        if fractions.shape[1] != len(elements):
            raise ValueError("Composition width does not match the number of elements.")
        table = self.property_table
        rows: list[list[float]] = []

        for fraction_row in fractions:
            output: list[float] = []
            for property_name in self.properties:
                if property_name not in table:
                    raise KeyError(f"Unknown elemental property {property_name!r}.")
                property_map = table[property_name]
                property_values = np.asarray([property_map.get(element, np.nan) for element in elements], dtype=float)
                active = fraction_row > 0
                missing_active = active & ~np.isfinite(property_values)
                if missing_active.any() and self.missing == "error":
                    missing_elements = [elements[index] for index in np.flatnonzero(missing_active)]
                    raise KeyError(f"Property {property_name!r} is missing for elements {missing_elements!r}.")
                if missing_active.any() and self.missing == "nan":
                    output.extend([float("nan")] * len(self.statistics))
                    continue

                valid = active & np.isfinite(property_values)
                weights = fraction_row[valid]
                selected = property_values[valid]
                if weights.size == 0:
                    output.extend([float("nan")] * len(self.statistics))
                    continue
                weights = weights / weights.sum()
                mean = float(np.dot(weights, selected))
                variance = float(np.dot(weights, (selected - mean) ** 2))
                statistic_values = {
                    "mean": mean,
                    "std": float(np.sqrt(max(variance, 0.0))),
                    "min": float(selected.min()),
                    "max": float(selected.max()),
                    "range": float(selected.max() - selected.min()),
                }
                output.extend(statistic_values[statistic] for statistic in self.statistics)

            if self.include_num_elements:
                output.append(float(np.count_nonzero(fraction_row > 0)))
            if self.include_mixing_entropy:
                positive = fraction_row[fraction_row > 0]
                output.append(float(-np.sum(positive * np.log(positive))))
            rows.append(output)

        return np.asarray(rows, dtype=float)
