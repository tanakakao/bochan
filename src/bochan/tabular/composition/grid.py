"""Exact fixed-support grid projection for composition Best Subset candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import floor
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CompositionGridFinalPostprocess:
    """Project one fixed-support composition block onto its configured step grid.

    Candidate optimization remains continuous. This callable is applied only to
    the final candidate returned for one support, where it solves a tiny MILP that
    minimizes the L1 displacement from the continuous optimum while preserving:

    - the exact active-element support,
    - component bounds,
    - per-element step sizes, and
    - the fixed composition total.

    The optional ``previous`` callback is evaluated first so ordinary process
    rounding/fixed-value handling can remain owned by the caller.
    """

    feature_indices: tuple[int, ...]
    elements: tuple[str, ...]
    total: float
    bounds: tuple[tuple[float, float], ...]
    steps: tuple[float | None, ...]
    exact_k: int
    previous: Callable[[Any], Any] | None = None
    tolerance: float = 1e-8

    @classmethod
    def from_config(
        cls,
        *,
        feature_indices: Sequence[int],
        elements: Sequence[str],
        config: Mapping[str, Any],
        exact_k: int,
        previous: Callable[[Any], Any] | None = None,
    ) -> CompositionGridFinalPostprocess:
        total = float(config["total"])
        bounds = tuple(
            tuple(
                map(
                    float,
                    config["bounds"].get(element, (0.0, total)),
                )
            )
            for element in elements
        )
        raw_steps = config.get("steps") or {}
        steps = tuple(
            None if raw_steps.get(element) is None else float(raw_steps[element])
            for element in elements
        )
        return cls(
            feature_indices=tuple(int(index) for index in feature_indices),
            elements=tuple(str(element) for element in elements),
            total=total,
            bounds=bounds,
            steps=steps,
            exact_k=int(exact_k),
            previous=previous,
        )

    def _project_row(
        self,
        values: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        from scipy.optimize import Bounds, LinearConstraint, milp

        active_indices = np.flatnonzero(active).tolist()
        if len(active_indices) != self.exact_k:
            raise ValueError(
                "Composition grid projection requires the inner optimizer to "
                f"preserve exactly {self.exact_k} active components; got "
                f"{len(active_indices)}."
            )

        n_active = len(active_indices)
        offsets = np.zeros(n_active, dtype=float)
        scales = np.ones(n_active, dtype=float)
        lower = np.zeros(2 * n_active, dtype=float)
        upper = np.full(2 * n_active, np.inf, dtype=float)
        integrality = np.zeros(2 * n_active, dtype=int)
        objective = np.zeros(2 * n_active, dtype=float)

        for local_index, component_index in enumerate(active_indices):
            component_lower, component_upper = self.bounds[component_index]
            step = self.steps[component_index]
            if step is None:
                positive_floor = min(
                    max(10.0 * self.tolerance, component_lower),
                    component_upper,
                )
                lower[local_index] = max(component_lower, positive_floor)
                upper[local_index] = component_upper
            else:
                offsets[local_index] = component_lower
                scales[local_index] = step
                minimum_integer = 0 if component_lower > self.tolerance else 1
                maximum_integer = floor(
                    (component_upper - component_lower) / step
                    + self.tolerance
                )
                if minimum_integer > maximum_integer:
                    raise ValueError(
                        "An active composition component has no positive grid point "
                        f"within its bounds: {self.elements[component_index]!r}."
                    )
                lower[local_index] = minimum_integer
                upper[local_index] = maximum_integer
                integrality[local_index] = 1
            objective[n_active + local_index] = 1.0 / max(self.total, 1.0)

        rows: list[np.ndarray] = []
        lower_constraints: list[float] = []
        upper_constraints: list[float] = []

        for local_index, component_index in enumerate(active_indices):
            target = float(values[component_index])

            row = np.zeros(2 * n_active, dtype=float)
            row[local_index] = scales[local_index]
            row[n_active + local_index] = -1.0
            rows.append(row)
            lower_constraints.append(-np.inf)
            upper_constraints.append(target - offsets[local_index])

            row = np.zeros(2 * n_active, dtype=float)
            row[local_index] = -scales[local_index]
            row[n_active + local_index] = -1.0
            rows.append(row)
            lower_constraints.append(-np.inf)
            upper_constraints.append(-target + offsets[local_index])

        total_row = np.zeros(2 * n_active, dtype=float)
        total_offset = 0.0
        for local_index in range(n_active):
            total_row[local_index] = scales[local_index]
            total_offset += offsets[local_index]
        rows.append(total_row)
        total_rhs = self.total - total_offset
        lower_constraints.append(total_rhs)
        upper_constraints.append(total_rhs)

        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(
                np.vstack(rows),
                np.asarray(lower_constraints, dtype=float),
                np.asarray(upper_constraints, dtype=float),
            ),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            support = [self.elements[index] for index in active_indices]
            raise ValueError(
                "The selected composition support has no feasible point on the "
                f"configured step grid: {support!r}."
            )

        projected = np.zeros(len(self.elements), dtype=float)
        active_values = offsets + scales * np.asarray(
            result.x[:n_active],
            dtype=float,
        )
        active_values[np.abs(active_values) <= self.tolerance] = 0.0
        projected[np.asarray(active_indices, dtype=int)] = active_values
        return projected

    def validate_support(self, active_elements: Sequence[str]) -> None:
        """Raise when one exact support has no feasible point on the step grid."""

        active_set = {str(element) for element in active_elements}
        unknown = active_set - set(self.elements)
        if unknown:
            raise KeyError(
                f"Unknown composition elements in support: {sorted(unknown)!r}."
            )
        if len(active_set) != self.exact_k:
            raise ValueError(
                f"Composition support must contain exactly {self.exact_k} elements."
            )

        active = np.asarray(
            [element in active_set for element in self.elements],
            dtype=bool,
        )
        target = np.zeros(len(self.elements), dtype=float)
        for index in np.flatnonzero(active):
            lower, upper = self.bounds[int(index)]
            target[int(index)] = 0.5 * (lower + upper)
        self._project_row(target, active)

    def __call__(self, candidates: Any) -> Any:
        import torch

        processed = self.previous(candidates) if self.previous is not None else candidates
        if not torch.is_tensor(processed):
            raise TypeError(
                "Composition grid final postprocess requires a torch.Tensor."
            )
        if torch.is_tensor(candidates) and tuple(processed.shape) != tuple(
            candidates.shape
        ):
            raise ValueError(
                "Composition grid final postprocess must preserve candidate shape."
            )

        result = processed.detach().clone()
        if result.ndim == 1:
            work = result.reshape(1, -1)
        else:
            work = result.reshape(-1, result.shape[-1])
        indices = list(self.feature_indices)

        for row in work:
            fractions = (
                row[indices].detach().cpu().numpy().astype(float, copy=True)
            )
            absolute = fractions * self.total
            active = np.abs(absolute) > self.tolerance
            projected = self._project_row(absolute, active)
            row[indices] = torch.as_tensor(
                projected / self.total,
                dtype=row.dtype,
                device=row.device,
            )

        return result


__all__ = ["CompositionGridFinalPostprocess"]
