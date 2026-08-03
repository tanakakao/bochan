"""Composition search-space validation and candidate repair."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _project_bounded_simplex(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, total: float) -> np.ndarray:
    if lower.sum() > total + 1e-12 or upper.sum() < total - 1e-12:
        raise ValueError("Bounds do not intersect the requested composition total.")
    left = float(np.min(values - upper))
    right = float(np.max(values - lower))
    for _ in range(200):
        midpoint = 0.5 * (left + right)
        projected = np.clip(values - midpoint, lower, upper)
        if projected.sum() > total:
            left = midpoint
        else:
            right = midpoint
    projected = np.clip(values - 0.5 * (left + right), lower, upper)
    residual = total - projected.sum()
    if abs(residual) > 1e-10:
        if residual > 0:
            room = upper - projected
        else:
            room = projected - lower
        for index in np.argsort(-room):
            adjustment = np.sign(residual) * min(abs(residual), room[index])
            projected[index] += adjustment
            residual -= adjustment
            if abs(residual) <= 1e-10:
                break
    return projected


@dataclass(frozen=True)
class CompositionSearchSpace:
    """Define and repair a bounded, optionally sparse composition simplex."""

    components: Sequence[str]
    total: float = 1.0
    bounds: Mapping[str, Sequence[float]] = field(default_factory=dict)
    steps: Mapping[str, float] = field(default_factory=dict)
    min_active_components: int = 1
    max_active_components: int | None = None
    required_components: Sequence[str] = field(default_factory=tuple)
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if len(components) < 2 or len(set(components)) != len(components):
            raise ValueError("components must contain at least two unique names.")
        if not np.isfinite(self.total) or self.total <= 0:
            raise ValueError("total must be finite and positive.")
        maximum = len(components) if self.max_active_components is None else int(self.max_active_components)
        if not 1 <= self.min_active_components <= maximum <= len(components):
            raise ValueError("Active-component limits are inconsistent with components.")
        unknown_required = set(self.required_components) - set(components)
        if unknown_required:
            raise KeyError(f"Unknown required components: {sorted(unknown_required)!r}.")
        if len(set(self.required_components)) > maximum:
            raise ValueError("required_components exceeds max_active_components.")
        lower, upper = self._bounds_arrays()
        if np.any(lower < 0) or np.any(upper < lower):
            raise ValueError("Each component bound must satisfy 0 <= lower <= upper.")
        if lower.sum() > self.total + self.tolerance or upper.sum() < self.total - self.tolerance:
            raise ValueError("Composition bounds cannot satisfy the requested total.")
        positive_lower = int(np.count_nonzero(lower > self.tolerance))
        if positive_lower > maximum:
            raise ValueError("Positive lower bounds require more active components than allowed.")
        for component, step in self.steps.items():
            if component not in components:
                raise KeyError(f"Unknown step component {component!r}.")
            if not np.isfinite(step) or float(step) <= 0:
                raise ValueError("Composition steps must be finite and positive.")

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(self.components)

    def _bounds_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(len(self.components), dtype=float)
        upper = np.full(len(self.components), float(self.total), dtype=float)
        for index, component in enumerate(self.components):
            if component in self.bounds:
                pair = tuple(self.bounds[component])
                if len(pair) != 2:
                    raise ValueError(f"Bounds for {component!r} must have length 2.")
                lower[index], upper[index] = map(float, pair)
        return lower, upper

    def _as_array(self, candidate: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
        if isinstance(candidate, Mapping):
            unknown = set(candidate) - set(self.components)
            if unknown:
                raise KeyError(f"Unknown composition components: {sorted(unknown)!r}.")
            values = np.asarray([candidate.get(component, 0.0) for component in self.components], dtype=float)
        else:
            values = np.asarray(candidate, dtype=float)
        if values.ndim != 1 or values.shape[0] != len(self.components):
            raise ValueError("Candidate width must match components.")
        if not np.isfinite(values).all():
            raise ValueError("Candidate values must be finite.")
        return values

    def _active_mask(self, raw: np.ndarray, lower: np.ndarray) -> np.ndarray:
        maximum = len(self.components) if self.max_active_components is None else int(self.max_active_components)
        required = {self.component_names.index(component) for component in self.required_components}
        required.update(np.flatnonzero(lower > self.tolerance).tolist())
        ranked = [int(index) for index in np.argsort(-raw)]
        active = set(required)
        for index in ranked:
            if len(active) >= maximum:
                break
            if raw[index] > self.tolerance or len(active) < self.min_active_components:
                active.add(index)
        if len(active) < self.min_active_components:
            for index in ranked:
                active.add(index)
                if len(active) >= self.min_active_components:
                    break
        mask = np.zeros(len(self.components), dtype=bool)
        mask[list(active)] = True
        return mask

    def _quantize(self, projected: np.ndarray, lower: np.ndarray, upper: np.ndarray, active: np.ndarray) -> np.ndarray:
        quantized = projected.copy()
        step_array = np.asarray([float(self.steps.get(component, 0.0)) for component in self.components])
        for index, step in enumerate(step_array):
            if not active[index] or step <= 0:
                continue
            base = lower[index]
            quantized[index] = base + round((quantized[index] - base) / step) * step
        quantized = np.clip(quantized, lower, upper)

        residual = self.total - quantized.sum()
        for _ in range(10000):
            if abs(residual) <= self.tolerance:
                break
            direction = 1.0 if residual > 0 else -1.0
            candidates: list[tuple[float, int, float]] = []
            for index in np.flatnonzero(active):
                step = step_array[index]
                if step <= 0:
                    capacity = upper[index] - quantized[index] if direction > 0 else quantized[index] - lower[index]
                    delta = min(abs(residual), capacity)
                else:
                    capacity = upper[index] - quantized[index] if direction > 0 else quantized[index] - lower[index]
                    can_step = capacity + self.tolerance >= step and abs(residual) + self.tolerance >= step
                    delta = step if can_step else 0.0
                if delta > self.tolerance:
                    score = direction * (projected[index] - quantized[index])
                    candidates.append((score, int(index), float(delta)))
            if not candidates:
                break
            _, index, delta = max(candidates)
            quantized[index] += direction * delta
            residual = self.total - quantized.sum()

        if abs(residual) > self.tolerance:
            raise ValueError(
                "The configured step sizes cannot satisfy the composition total exactly. "
                "Use compatible steps or relax the bounds."
            )
        return quantized

    def repair(self, candidate: Mapping[str, float] | Sequence[float] | np.ndarray) -> dict[str, float]:
        """Repair a raw candidate to bounds, sparsity, steps, and total constraints."""

        raw = np.maximum(self._as_array(candidate), 0.0)
        lower, upper = self._bounds_arrays()
        active = self._active_mask(raw, lower)
        inactive = ~active
        if np.any(lower[inactive] > self.tolerance):
            raise ValueError("An inactive component has a positive lower bound.")
        active_lower = lower.copy()
        active_upper = upper.copy()
        active_lower[inactive] = 0.0
        active_upper[inactive] = 0.0
        for index in np.flatnonzero(active):
            component = self.components[index]
            positive_floor = float(self.steps.get(component, self.tolerance))
            active_lower[index] = max(active_lower[index], positive_floor)
        projected = _project_bounded_simplex(raw, active_lower, active_upper, self.total)
        repaired = self._quantize(projected, active_lower, active_upper, active)
        return {component: float(repaired[index]) for index, component in enumerate(self.components)}

    def validate(self, candidate: Mapping[str, float] | Sequence[float] | np.ndarray) -> list[str]:
        """Return human-readable validation errors without modifying the candidate."""

        values = self._as_array(candidate)
        lower, upper = self._bounds_arrays()
        errors: list[str] = []
        if np.any(values < lower - self.tolerance) or np.any(values > upper + self.tolerance):
            errors.append("One or more component values are outside their bounds.")
        if abs(values.sum() - self.total) > self.tolerance:
            errors.append(f"Component sum must equal {self.total}.")
        active_count = int(np.count_nonzero(values > self.tolerance))
        maximum = len(self.components) if self.max_active_components is None else int(self.max_active_components)
        if not self.min_active_components <= active_count <= maximum:
            errors.append(f"Active component count must be between {self.min_active_components} and {maximum}.")
        for component in self.required_components:
            if values[self.component_names.index(component)] <= self.tolerance:
                errors.append(f"Required component {component!r} is inactive.")
        for index, component in enumerate(self.components):
            step = self.steps.get(component)
            if step is None or values[index] <= self.tolerance:
                continue
            offset = values[index] - lower[index]
            if abs(offset / float(step) - round(offset / float(step))) > self.tolerance:
                errors.append(f"Component {component!r} does not follow step {step}.")
        return errors

    def repair_frame(self, frame: Any) -> Any:
        """Repair composition columns in a pandas DataFrame."""

        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("repair_frame expects a pandas.DataFrame.")
        repaired = frame.copy()
        missing = [component for component in self.components if component not in repaired.columns]
        if missing:
            raise KeyError(f"Missing composition columns: {missing!r}.")
        rows = [self.repair(row) for row in repaired.loc[:, list(self.components)].to_dict(orient="records")]
        repaired.loc[:, list(self.components)] = pd.DataFrame(rows, index=repaired.index)
        return repaired
