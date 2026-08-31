"""Fixed-support grid projection for composition Best Subset candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import floor
from typing import Any

import numpy as np

from bochan.api.support.best_subset import InfeasibleBestSubsetSupportError


@dataclass(frozen=True)
class GridLinearConstraint:
    """Linear constraint projected onto the composition grid.

    Composition coefficients multiply the projector's absolute amount variables.
    Non-composition coefficients multiply values that are held fixed at the candidate
    row supplied to the final postprocessor. ``lower`` / ``upper`` use the same scaled
    units as the composition amount expression.
    """

    composition_coefficients: tuple[float, ...]
    process_indices: tuple[int, ...]
    process_coefficients: tuple[float, ...]
    lower: float
    upper: float

    @property
    def is_coupled(self) -> bool:
        """Return whether the constraint also depends on non-composition features."""

        return bool(self.process_indices)


def _constraint_items(values: Any) -> list[Any]:
    if isinstance(values, (str, int, np.integer)):
        return [values]
    if hasattr(values, "detach"):
        return values.detach().cpu().reshape(-1).tolist()
    return list(values)


def _constraint_coefficients(values: Any) -> list[float]:
    if isinstance(values, (int, float, np.integer, np.floating)):
        return [float(values)]
    if hasattr(values, "detach"):
        return [float(value) for value in values.detach().cpu().reshape(-1).tolist()]
    return [float(value) for value in values]


def composition_grid_linear_constraints(
    *,
    equality_constraints: Sequence[tuple[Any, Any, Any]] | None,
    inequality_constraints: Sequence[tuple[Any, Any, Any]] | None,
    feature_names: Sequence[Any],
    composition_feature_names: Sequence[str],
    inequality_sense: str = "ge",
    rhs_scale: float = 1.0,
    context: str = "Composition step-grid best_subset",
) -> tuple[GridLinearConstraint, ...]:
    """Extract constraints that touch composition for the final grid MILP.

    Process-only constraints are ignored because the projector does not modify those
    dimensions. Constraints that mix composition and non-composition features retain
    the non-composition terms as candidate-row constants. ``rhs_scale`` maps
    normalized-fraction constraints to the absolute amount variables used internally
    by the fixed-total projector. The same scale is therefore applied to the RHS and
    all non-composition coefficients, while composition coefficients multiply amounts
    directly.
    """

    scale = float(rhs_scale)
    if scale <= 0.0:
        raise ValueError("rhs_scale must be positive.")
    sense = str(inequality_sense).lower()
    if sense not in {"ge", "le"}:
        raise ValueError("inequality_sense must be 'ge' or 'le'.")

    composition_names = tuple(str(name) for name in composition_feature_names)
    component_index = {name: index for index, name in enumerate(composition_names)}
    feature_values = tuple(feature_names)
    feature_index = {str(name): index for index, name in enumerate(feature_values)}

    def resolve_item(item: Any) -> tuple[str, int]:
        if isinstance(item, str):
            component_position = component_index.get(item)
            if component_position is not None:
                return "composition", int(component_position)
            process_position = feature_index.get(item)
            if process_position is None:
                raise KeyError(
                    f"Unknown linear-constraint feature {item!r} for {context}."
                )
            return "process", int(process_position)
        if isinstance(item, (int, np.integer)):
            position = int(item)
            if position < 0 or position >= len(feature_values):
                raise IndexError(
                    f"Constraint feature index {position} is outside the candidate "
                    f"dimension {len(feature_values)}."
                )
            component_position = component_index.get(str(feature_values[position]))
            if component_position is not None:
                return "composition", int(component_position)
            return "process", position
        raise TypeError(
            f"Constraint feature identifiers for {context} must be names or integer indices."
        )

    def convert(
        constraints: Sequence[tuple[Any, Any, Any]] | None,
        *,
        equality: bool,
    ) -> list[GridLinearConstraint]:
        converted: list[GridLinearConstraint] = []
        for indices, coefficients, rhs in constraints or ():
            items = _constraint_items(indices)
            coeffs = _constraint_coefficients(coefficients)
            if len(items) != len(coeffs):
                raise ValueError(
                    "Constraint indices and coefficients must have matching lengths."
                )

            resolved = [resolve_item(item) for item in items]
            touches_composition = any(kind == "composition" for kind, _ in resolved)
            if not touches_composition:
                continue

            vector = [0.0] * len(composition_names)
            process_by_index: dict[int, float] = {}
            for (kind, index), coefficient in zip(resolved, coeffs, strict=True):
                if kind == "composition":
                    vector[index] += float(coefficient)
                else:
                    process_by_index[index] = process_by_index.get(index, 0.0) + (
                        float(coefficient) * scale
                    )

            scaled_rhs = float(rhs) * scale
            if equality:
                lower = scaled_rhs
                upper = scaled_rhs
            elif sense == "ge":
                lower = scaled_rhs
                upper = np.inf
            else:
                lower = -np.inf
                upper = scaled_rhs

            process_indices = tuple(sorted(process_by_index))
            process_coefficients = tuple(
                process_by_index[index] for index in process_indices
            )
            converted.append(
                GridLinearConstraint(
                    composition_coefficients=tuple(vector),
                    process_indices=process_indices,
                    process_coefficients=process_coefficients,
                    lower=float(lower),
                    upper=float(upper),
                )
            )
        return converted

    return tuple(
        [
            *convert(equality_constraints, equality=True),
            *convert(inequality_constraints, equality=False),
        ]
    )


def merge_composition_grid_linear_constraints(
    *groups: Sequence[GridLinearConstraint],
) -> tuple[GridLinearConstraint, ...]:
    """Merge grid constraints while preserving order and removing duplicates."""

    result: list[GridLinearConstraint] = []
    seen: set[GridLinearConstraint] = set()
    for group in groups:
        for constraint in group:
            normalized = GridLinearConstraint(
                composition_coefficients=tuple(
                    float(value) for value in constraint.composition_coefficients
                ),
                process_indices=tuple(int(value) for value in constraint.process_indices),
                process_coefficients=tuple(
                    float(value) for value in constraint.process_coefficients
                ),
                lower=float(constraint.lower),
                upper=float(constraint.upper),
            )
            if len(normalized.process_indices) != len(normalized.process_coefficients):
                raise ValueError(
                    "Grid linear constraint process indices and coefficients must have "
                    "matching lengths."
                )
            if normalized not in seen:
                result.append(normalized)
                seen.add(normalized)
    return tuple(result)


def _static_grid_linear_constraints(
    constraints: Sequence[GridLinearConstraint],
) -> tuple[GridLinearConstraint, ...]:
    """Return constraints whose feasibility is independent of candidate process values."""

    return tuple(constraint for constraint in constraints if not constraint.is_coupled)


def _project_grid_row(
    values: np.ndarray,
    active: np.ndarray,
    *,
    elements: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    steps: Sequence[float | None],
    minimum_k: int,
    maximum_k: int,
    total_bounds: tuple[float, float],
    tolerance: float,
    linear_constraints: Sequence[GridLinearConstraint] = (),
    constraint_values: np.ndarray | None = None,
) -> np.ndarray:
    """Project one selected support to its nearest feasible amount-grid point.

    Best Subset owns support/cardinality selection. The MILP therefore preserves the
    support it receives and optimizes only the active component values. Variable-
    cardinality search is supported by accepting any selected support whose size lies
    in ``[minimum_k, maximum_k]``. For composition/process coupled constraints,
    non-composition features are held at the supplied candidate-row values and moved
    to the constraint RHS.
    """

    from scipy.optimize import Bounds, LinearConstraint, milp

    active_indices = np.flatnonzero(active).tolist()
    active_count = len(active_indices)
    if active_count < int(minimum_k) or active_count > int(maximum_k):
        raise InfeasibleBestSubsetSupportError(
            "Composition grid projection requires the inner optimizer to preserve an "
            f"active-component count in [{minimum_k}, {maximum_k}]; got {active_count}."
        )

    n_active = active_count
    if n_active == 0:
        raise InfeasibleBestSubsetSupportError(
            "Composition grid projection requires at least one active component."
        )

    offsets = np.zeros(n_active, dtype=float)
    scales = np.ones(n_active, dtype=float)
    lower = np.zeros(2 * n_active, dtype=float)
    upper = np.full(2 * n_active, np.inf, dtype=float)
    integrality = np.zeros(2 * n_active, dtype=int)
    objective = np.zeros(2 * n_active, dtype=float)
    total_scale = max(float(total_bounds[1]), 1.0)

    for local_index, component_index in enumerate(active_indices):
        component_lower, component_upper = bounds[component_index]
        step = steps[component_index]
        if step is None:
            positive_floor = min(
                max(10.0 * tolerance, component_lower),
                component_upper,
            )
            lower[local_index] = max(component_lower, positive_floor)
            upper[local_index] = component_upper
        else:
            offsets[local_index] = component_lower
            scales[local_index] = step
            minimum_integer = 0 if component_lower > tolerance else 1
            maximum_integer = floor(
                (component_upper - component_lower) / step + tolerance
            )
            if minimum_integer > maximum_integer:
                raise InfeasibleBestSubsetSupportError(
                    "An active composition component has no positive grid point "
                    f"within its bounds: {elements[component_index]!r}."
                )
            lower[local_index] = minimum_integer
            upper[local_index] = maximum_integer
            integrality[local_index] = 1
        objective[n_active + local_index] = 1.0 / total_scale

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
    total_lower, total_upper = map(float, total_bounds)
    lower_constraints.append(total_lower - total_offset)
    upper_constraints.append(total_upper - total_offset)

    for constraint in linear_constraints:
        coefficients = constraint.composition_coefficients
        if len(coefficients) != len(elements):
            raise ValueError(
                "Composition grid linear constraint width must match the element count."
            )
        if len(constraint.process_indices) != len(constraint.process_coefficients):
            raise ValueError(
                "Grid linear constraint process indices and coefficients must have "
                "matching lengths."
            )

        process_contribution = 0.0
        if constraint.process_indices:
            if constraint_values is None:
                raise ValueError(
                    "Candidate-row values are required to project a composition/process "
                    "coupled linear constraint."
                )
            for process_index, coefficient in zip(
                constraint.process_indices,
                constraint.process_coefficients,
                strict=True,
            ):
                if process_index < 0 or process_index >= len(constraint_values):
                    raise IndexError(
                        f"Constraint process feature index {process_index} is outside "
                        f"candidate dimension {len(constraint_values)}."
                    )
                process_contribution += float(coefficient) * float(
                    constraint_values[process_index]
                )

        row = np.zeros(2 * n_active, dtype=float)
        offset = 0.0
        for local_index, component_index in enumerate(active_indices):
            coefficient = float(coefficients[component_index])
            row[local_index] = coefficient * scales[local_index]
            offset += coefficient * offsets[local_index]

        adjusted_lower = float(constraint.lower) - process_contribution - offset
        adjusted_upper = float(constraint.upper) - process_contribution - offset
        if np.all(np.abs(row[:n_active]) <= tolerance):
            if adjusted_lower > tolerance or adjusted_upper < -tolerance:
                support = [elements[index] for index in active_indices]
                raise InfeasibleBestSubsetSupportError(
                    "The selected composition support cannot satisfy the configured "
                    f"linear constraints on the step grid: {support!r}."
                )
            continue
        rows.append(row)
        lower_constraints.append(adjusted_lower)
        upper_constraints.append(adjusted_upper)

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
        support = [elements[index] for index in active_indices]
        raise InfeasibleBestSubsetSupportError(
            "The selected composition support has no feasible point on the "
            f"configured step grid and linear constraints: {support!r}."
        )

    projected = np.zeros(len(elements), dtype=float)
    active_values = offsets + scales * np.asarray(result.x[:n_active], dtype=float)
    active_values[np.abs(active_values) <= tolerance] = 0.0
    projected[np.asarray(active_indices, dtype=int)] = active_values
    return projected


def _validate_support(
    active_elements: Sequence[str],
    *,
    elements: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    steps: Sequence[float | None],
    minimum_k: int,
    maximum_k: int,
    total_bounds: tuple[float, float],
    tolerance: float,
    linear_constraints: Sequence[GridLinearConstraint] = (),
) -> None:
    active_set = {str(element) for element in active_elements}
    unknown = active_set - set(elements)
    if unknown:
        raise KeyError(
            f"Unknown composition elements in support: {sorted(unknown)!r}."
        )
    if len(active_set) < int(minimum_k) or len(active_set) > int(maximum_k):
        raise ValueError(
            "Composition support must contain a number of elements in "
            f"[{minimum_k}, {maximum_k}]."
        )

    active = np.asarray([element in active_set for element in elements], dtype=bool)
    target = np.zeros(len(elements), dtype=float)
    for index in np.flatnonzero(active):
        lower, upper = bounds[int(index)]
        target[int(index)] = 0.5 * (lower + upper)
    _project_grid_row(
        target,
        active,
        elements=elements,
        bounds=bounds,
        steps=steps,
        minimum_k=minimum_k,
        maximum_k=maximum_k,
        total_bounds=total_bounds,
        tolerance=tolerance,
        linear_constraints=_static_grid_linear_constraints(linear_constraints),
    )


def _configured_required_forbidden(
    *,
    elements: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    required_elements: Sequence[str],
    forbidden_elements: Sequence[str],
    tolerance: float,
) -> tuple[set[str], set[str]]:
    required = set(str(element) for element in required_elements)
    forbidden = set(str(element) for element in forbidden_elements)
    for index, element in enumerate(elements):
        lower, upper = bounds[index]
        if lower > tolerance:
            required.add(str(element))
        if upper <= tolerance:
            forbidden.add(str(element))
    return required, forbidden


def _has_feasible_support(
    *,
    elements: Sequence[str],
    required_elements: Sequence[str],
    forbidden_elements: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    steps: Sequence[float | None],
    minimum_k: int,
    maximum_k: int,
    total_bounds: tuple[float, float],
    tolerance: float,
    linear_constraints: Sequence[GridLinearConstraint],
) -> bool:
    """Return whether at least one statically feasible support exists in the k-range."""

    element_tuple = tuple(str(element) for element in elements)
    required, forbidden = _configured_required_forbidden(
        elements=element_tuple,
        bounds=bounds,
        required_elements=required_elements,
        forbidden_elements=forbidden_elements,
        tolerance=tolerance,
    )
    if required & forbidden:
        return False

    optional = [
        element
        for element in element_tuple
        if element not in required and element not in forbidden
    ]
    ordered_required = [element for element in element_tuple if element in required]

    effective_minimum = max(int(minimum_k), len(required))
    effective_maximum = min(int(maximum_k), len(required) + len(optional))
    if effective_minimum > effective_maximum:
        return False

    for support_k in range(effective_minimum, effective_maximum + 1):
        choose = support_k - len(required)
        for selected in combinations(optional, choose):
            try:
                _validate_support(
                    [*ordered_required, *selected],
                    elements=element_tuple,
                    bounds=bounds,
                    steps=steps,
                    minimum_k=minimum_k,
                    maximum_k=maximum_k,
                    total_bounds=total_bounds,
                    tolerance=tolerance,
                    linear_constraints=linear_constraints,
                )
            except InfeasibleBestSubsetSupportError:
                continue
            return True
    return False


def _effective_minimum_k(
    *,
    config: Mapping[str, Any],
    elements: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    maximum_k: int,
    tolerance: float = 1e-8,
) -> int:
    configured = int(config.get("min_components", maximum_k))
    required, _ = _configured_required_forbidden(
        elements=elements,
        bounds=bounds,
        required_elements=config.get("required_components") or (),
        forbidden_elements=config.get("forbidden_components") or (),
        tolerance=tolerance,
    )
    return min(int(maximum_k), max(configured, len(required)))


@dataclass(frozen=True)
class CompositionGridFinalPostprocess:
    """Project one fixed-total composition block onto its configured step grid."""

    feature_indices: tuple[int, ...]
    elements: tuple[str, ...]
    total: float
    bounds: tuple[tuple[float, float], ...]
    steps: tuple[float | None, ...]
    exact_k: int
    minimum_k: int | None = None
    required_elements: tuple[str, ...] = ()
    forbidden_elements: tuple[str, ...] = ()
    linear_constraints: tuple[GridLinearConstraint, ...] = ()
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
        linear_constraints: Sequence[GridLinearConstraint] | None = None,
        previous: Callable[[Any], Any] | None = None,
    ) -> CompositionGridFinalPostprocess:
        total = float(config["total"])
        bounds = tuple(
            tuple(map(float, config["bounds"].get(element, (0.0, total))))
            for element in elements
        )
        raw_steps = config.get("steps") or {}
        steps = tuple(
            None if raw_steps.get(element) is None else float(raw_steps[element])
            for element in elements
        )
        maximum_k = int(exact_k)
        minimum_k = _effective_minimum_k(
            config=config,
            elements=elements,
            bounds=bounds,
            maximum_k=maximum_k,
        )
        return cls(
            feature_indices=tuple(int(index) for index in feature_indices),
            elements=tuple(str(element) for element in elements),
            total=total,
            bounds=bounds,
            steps=steps,
            exact_k=maximum_k,
            minimum_k=minimum_k,
            required_elements=tuple(
                str(element) for element in config.get("required_components") or ()
            ),
            forbidden_elements=tuple(
                str(element) for element in config.get("forbidden_components") or ()
            ),
            linear_constraints=tuple(linear_constraints or ()),
            previous=previous,
        )

    @property
    def minimum_cardinality(self) -> int:
        return self.exact_k if self.minimum_k is None else int(self.minimum_k)

    def _project_row(
        self,
        values: np.ndarray,
        active: np.ndarray,
        *,
        constraint_values: np.ndarray | None = None,
    ) -> np.ndarray:
        return _project_grid_row(
            values,
            active,
            elements=self.elements,
            bounds=self.bounds,
            steps=self.steps,
            minimum_k=self.minimum_cardinality,
            maximum_k=self.exact_k,
            total_bounds=(self.total, self.total),
            tolerance=self.tolerance,
            linear_constraints=self.linear_constraints,
            constraint_values=constraint_values,
        )

    def validate_support(self, active_elements: Sequence[str]) -> None:
        """Raise only when the configured cardinality range has no feasible support."""

        try:
            _validate_support(
                active_elements,
                elements=self.elements,
                bounds=self.bounds,
                steps=self.steps,
                minimum_k=self.minimum_cardinality,
                maximum_k=self.exact_k,
                total_bounds=(self.total, self.total),
                tolerance=self.tolerance,
                linear_constraints=self.linear_constraints,
            )
        except InfeasibleBestSubsetSupportError:
            if self.minimum_cardinality != self.exact_k and _has_feasible_support(
                elements=self.elements,
                required_elements=self.required_elements,
                forbidden_elements=self.forbidden_elements,
                bounds=self.bounds,
                steps=self.steps,
                minimum_k=self.minimum_cardinality,
                maximum_k=self.exact_k,
                total_bounds=(self.total, self.total),
                tolerance=self.tolerance,
                linear_constraints=self.linear_constraints,
            ):
                return
            raise

    def __call__(self, candidates: Any) -> Any:
        import torch

        processed = self.previous(candidates) if self.previous is not None else candidates
        if not torch.is_tensor(processed):
            raise TypeError(
                "Composition grid final postprocess requires a torch.Tensor."
            )
        if torch.is_tensor(candidates) and tuple(processed.shape) != tuple(candidates.shape):
            raise ValueError(
                "Composition grid final postprocess must preserve candidate shape."
            )

        result = processed.detach().clone()
        work = (
            result.reshape(1, -1)
            if result.ndim == 1
            else result.reshape(-1, result.shape[-1])
        )
        indices = list(self.feature_indices)

        for row in work:
            candidate_values = row.detach().cpu().numpy().astype(float, copy=True)
            fractions = candidate_values[indices].copy()
            absolute = fractions * self.total
            active = np.abs(absolute) > self.tolerance
            projected = self._project_row(
                absolute,
                active,
                constraint_values=candidate_values,
            )
            row[indices] = torch.as_tensor(
                projected / self.total,
                dtype=row.dtype,
                device=row.device,
            )

        return result


@dataclass(frozen=True)
class CompositionVariableTotalGridFinalPostprocess:
    """Project raw absolute composition amounts onto a variable-total step grid."""

    feature_indices: tuple[int, ...]
    elements: tuple[str, ...]
    total_bounds: tuple[float, float]
    bounds: tuple[tuple[float, float], ...]
    steps: tuple[float | None, ...]
    exact_k: int
    minimum_k: int | None = None
    required_elements: tuple[str, ...] = ()
    forbidden_elements: tuple[str, ...] = ()
    linear_constraints: tuple[GridLinearConstraint, ...] = ()
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
        linear_constraints: Sequence[GridLinearConstraint] | None = None,
        previous: Callable[[Any], Any] | None = None,
    ) -> CompositionVariableTotalGridFinalPostprocess:
        total_pair = tuple(config["total_bounds"])
        if len(total_pair) != 2:
            raise ValueError("total_bounds must contain two values.")
        total_lower, total_upper = map(float, total_pair)
        if total_lower <= 0.0 or total_lower >= total_upper:
            raise ValueError("total_bounds must be positive and increasing.")

        configured_bounds = config.get("bounds") or {}
        bounds = tuple(
            tuple(map(float, configured_bounds.get(element, (0.0, total_upper))))
            for element in elements
        )
        raw_steps = config.get("steps") or {}
        steps = tuple(
            None if raw_steps.get(element) is None else float(raw_steps[element])
            for element in elements
        )
        maximum_k = int(exact_k)
        minimum_k = _effective_minimum_k(
            config=config,
            elements=elements,
            bounds=bounds,
            maximum_k=maximum_k,
        )
        return cls(
            feature_indices=tuple(int(index) for index in feature_indices),
            elements=tuple(str(element) for element in elements),
            total_bounds=(total_lower, total_upper),
            bounds=bounds,
            steps=steps,
            exact_k=maximum_k,
            minimum_k=minimum_k,
            required_elements=tuple(
                str(element) for element in config.get("required_components") or ()
            ),
            forbidden_elements=tuple(
                str(element) for element in config.get("forbidden_components") or ()
            ),
            linear_constraints=tuple(linear_constraints or ()),
            previous=previous,
        )

    @property
    def minimum_cardinality(self) -> int:
        return self.exact_k if self.minimum_k is None else int(self.minimum_k)

    def _project_row(
        self,
        values: np.ndarray,
        active: np.ndarray,
        *,
        constraint_values: np.ndarray | None = None,
    ) -> np.ndarray:
        return _project_grid_row(
            values,
            active,
            elements=self.elements,
            bounds=self.bounds,
            steps=self.steps,
            minimum_k=self.minimum_cardinality,
            maximum_k=self.exact_k,
            total_bounds=self.total_bounds,
            tolerance=self.tolerance,
            linear_constraints=self.linear_constraints,
            constraint_values=constraint_values,
        )

    def validate_support(self, active_elements: Sequence[str]) -> None:
        """Raise only when no configured support in the k-range is grid feasible."""

        try:
            _validate_support(
                active_elements,
                elements=self.elements,
                bounds=self.bounds,
                steps=self.steps,
                minimum_k=self.minimum_cardinality,
                maximum_k=self.exact_k,
                total_bounds=self.total_bounds,
                tolerance=self.tolerance,
                linear_constraints=self.linear_constraints,
            )
        except InfeasibleBestSubsetSupportError:
            if _has_feasible_support(
                elements=self.elements,
                required_elements=self.required_elements,
                forbidden_elements=self.forbidden_elements,
                bounds=self.bounds,
                steps=self.steps,
                minimum_k=self.minimum_cardinality,
                maximum_k=self.exact_k,
                total_bounds=self.total_bounds,
                tolerance=self.tolerance,
                linear_constraints=self.linear_constraints,
            ):
                return
            raise

    def __call__(self, candidates: Any) -> Any:
        import torch

        processed = self.previous(candidates) if self.previous is not None else candidates
        if not torch.is_tensor(processed):
            raise TypeError(
                "Variable-total composition grid final postprocess requires a torch.Tensor."
            )
        if torch.is_tensor(candidates) and tuple(processed.shape) != tuple(candidates.shape):
            raise ValueError(
                "Variable-total composition grid final postprocess must preserve "
                "candidate shape."
            )

        result = processed.detach().clone()
        work = (
            result.reshape(1, -1)
            if result.ndim == 1
            else result.reshape(-1, result.shape[-1])
        )
        indices = list(self.feature_indices)
        total_lower, total_upper = self.total_bounds

        for row in work:
            candidate_values = row.detach().cpu().numpy().astype(float, copy=True)
            amounts = candidate_values[indices].copy()
            active = np.abs(amounts) > self.tolerance
            projected = self._project_row(
                amounts,
                active,
                constraint_values=candidate_values,
            )
            projected_total = float(projected.sum())
            if (
                projected_total < total_lower - self.tolerance
                or projected_total > total_upper + self.tolerance
            ):
                raise RuntimeError(
                    "Variable-total composition grid projection produced a total "
                    "outside total_bounds."
                )
            row[indices] = torch.as_tensor(
                projected,
                dtype=row.dtype,
                device=row.device,
            )

        return result


__all__ = [
    "CompositionGridFinalPostprocess",
    "CompositionVariableTotalGridFinalPostprocess",
    "GridLinearConstraint",
    "composition_grid_linear_constraints",
    "merge_composition_grid_linear_constraints",
]
