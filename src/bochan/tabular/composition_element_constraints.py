"""Element-constraint normalization and mixed-integer composition projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations, islice, product
from math import ceil, floor
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .composition import ATOMIC_WEIGHTS, close_compositions, format_formula

_WEIGHT_NORMALIZATIONS = {"weight_fraction", "weight", "mass_fraction"}
_BASIS_ALIASES = {
    "atomic": "atomic_amount",
    "atomic_amount": "atomic_amount",
    "molar": "atomic_amount",
    "mole": "atomic_amount",
    "weight": "weight_amount",
    "weight_amount": "weight_amount",
    "mass": "weight_amount",
    "mass_amount": "weight_amount",
}


class CompositionElementConstraintResolver:
    """Normalize element constraints and map compatible ones to model coordinates."""

    @staticmethod
    def normalize(
        constraints: Sequence[Any] | None,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(constraints or ()):
            if not isinstance(raw, Mapping):
                raise TypeError(
                    "Each composition element constraint must be a mapping."
                )
            raw_terms = raw.get("terms")
            if not isinstance(raw_terms, Sequence) or isinstance(raw_terms, str):
                raise ValueError(
                    f"Composition element constraint {index} requires 'terms'."
                )
            combined: dict[tuple[str, str], float] = {}
            for term_index, raw_term in enumerate(raw_terms):
                if not isinstance(raw_term, Mapping):
                    raise TypeError(
                        f"Term {term_index} in composition element constraint "
                        f"{index} must be a mapping."
                    )
                site = raw_term.get("site")
                element = raw_term.get("element")
                coefficient = raw_term.get("coefficient", 1.0)
                if site is None or element is None:
                    raise ValueError(
                        f"Term {term_index} in composition element constraint "
                        f"{index} requires site and element."
                    )
                coefficient = float(coefficient)
                if not np.isfinite(coefficient):
                    raise ValueError("Element-constraint coefficients must be finite.")
                key = (str(site), str(element))
                combined[key] = combined.get(key, 0.0) + coefficient

            terms = tuple(
                {
                    "site": site,
                    "element": element,
                    "coefficient": coefficient,
                }
                for (site, element), coefficient in combined.items()
                if abs(coefficient) > 1e-15
            )
            if not terms:
                raise ValueError(
                    f"Composition element constraint {index} has no nonzero terms."
                )

            operator = str(raw.get("operator", raw.get("op", "=")))
            if operator == "==":
                operator = "="
            if operator not in {"=", "<=", ">="}:
                raise ValueError(
                    f"Unknown composition element operator {operator!r}."
                )
            rhs = float(raw.get("rhs", 0.0))
            if not np.isfinite(rhs):
                raise ValueError("Element-constraint rhs must be finite.")
            basis_name = str(raw.get("basis", "atomic_amount")).lower()
            try:
                basis = _BASIS_ALIASES[basis_name]
            except KeyError as exc:
                raise ValueError(
                    "Element-constraint basis must be 'atomic_amount' or "
                    "'weight_amount'."
                ) from exc
            normalized.append(
                {
                    "terms": terms,
                    "operator": operator,
                    "rhs": rhs,
                    "basis": basis,
                }
            )
        return normalized


    @staticmethod
    def component_bounds(
        config: Mapping[str, Any],
        element: str,
        total: float,
    ) -> tuple[float, float]:
        pair = config["bounds"].get(element, (0.0, total))
        lower, upper = map(float, pair)
        return max(0.0, lower), min(float(total), upper)


    @staticmethod
    def basis_scale(
        config: Mapping[str, Any],
        element: str,
        basis: str,
    ) -> float:
        native_is_weight = (
            str(config["normalization"]).lower() in _WEIGHT_NORMALIZATIONS
        )
        if basis == "atomic_amount":
            return 1.0 / ATOMIC_WEIGHTS[element] if native_is_weight else 1.0
        return 1.0 if native_is_weight else ATOMIC_WEIGHTS[element]

    @classmethod
    def named_constraints(
        cls,
        constraints: Sequence[Mapping[str, Any]],
        composition_sites: Mapping[str, Mapping[str, Any]],
        composition_transformers: Mapping[str, Any],
    ) -> list[tuple[Any, ...]]:
        "Translate fixed-total Fraction constraints to named model constraints."

        if not constraints:
            return []
        resolved: list[tuple[Any, ...]] = []
        for constraint in constraints:
            names: list[str] = []
            coefficients: list[float] = []
            compatible = True
            for term in constraint["terms"]:
                config = composition_sites[term["site"]]
                if config.get("variable_total") or str(
                    config["representation"]
                ).lower() not in {"fraction", "fractions"}:
                    compatible = False
                    break
                transformer = composition_transformers.get(term["site"])
                if transformer is None:
                    compatible = False
                    break
                names.append(
                    f"{transformer.prefix}__fraction__{term['element']}"
                )
                coefficients.append(
                    float(term["coefficient"])
                    * cls.basis_scale(
                        config,
                        term["element"],
                        constraint["basis"],
                    )
                    * float(config["total"])
                )
            if compatible:
                resolved.append(
                    (
                        names,
                        coefficients,
                        constraint["operator"],
                        float(constraint["rhs"]),
                    )
                )
        return resolved


class CompositionElementConstraintProjector:
    "Project native composition values onto element constraints using MILP."

    def __init__(
        self,
        *,
        composition_sites: Mapping[str, Mapping[str, Any]],
        composition_element_constraints: Sequence[Mapping[str, Any]],
        composition_transformers: Mapping[str, Any],
        max_supports: int,
    ) -> None:
        self.composition_sites = composition_sites
        self.composition_element_constraints = composition_element_constraints
        self.composition_transformers_ = composition_transformers
        self.composition_constraint_max_supports = int(max_supports)

    @staticmethod
    def _component_bounds(
        config: Mapping[str, Any],
        element: str,
        total: float,
    ) -> tuple[float, float]:
        return CompositionElementConstraintResolver.component_bounds(
            config,
            element,
            total,
        )

    @staticmethod
    def _basis_scale(
        config: Mapping[str, Any],
        element: str,
        basis: str,
    ) -> float:
        return CompositionElementConstraintResolver.basis_scale(
            config,
            element,
            basis,
        )

    def validate(self) -> None:
        "Validate bounds and joint fixed-total feasibility."
        self._validate_element_constraints()

    def project(
        self,
        raw: Mapping[tuple[str, str], float],
        totals: Mapping[str, float],
    ) -> dict[tuple[str, str], float]:
        "Project one native composition onto all configured constraints."
        return self._project_element_values(raw, totals)

    def repair_frame(self, restored: Any) -> Any:
        "Project every row of a restored composition frame."
        return self._repair_element_constraint_frame(restored)

    def _validate_element_constraints(self) -> None:
        if not self.composition_element_constraints:
            return
        if not self.composition_sites:
            raise ValueError(
                "composition_element_constraints requires composition_sites."
            )
        for constraint in self.composition_element_constraints:
            lhs_min = 0.0
            lhs_max = 0.0
            for term in constraint["terms"]:
                site = term["site"]
                if site not in self.composition_sites:
                    raise KeyError(
                        f"Unknown composition site {site!r} in element constraint."
                    )
                config = self.composition_sites[site]
                element = term["element"]
                if element not in config["elements"]:
                    raise KeyError(
                        f"Unknown element {element!r} at composition site {site!r}."
                    )
                if config.get("variable_total"):
                    total_upper = float(config["total_bounds"][1])
                else:
                    total_upper = float(config["total"])
                lower, upper = self._component_bounds(config, element, total_upper)
                scale = self._basis_scale(config, element, constraint["basis"])
                coefficient = float(term["coefficient"]) * scale
                if coefficient >= 0.0:
                    lhs_min += coefficient * lower
                    lhs_max += coefficient * upper
                else:
                    lhs_min += coefficient * upper
                    lhs_max += coefficient * lower

            rhs = float(constraint["rhs"])
            operator = constraint["operator"]
            feasible = (
                lhs_min - 1e-8 <= rhs <= lhs_max + 1e-8
                if operator == "="
                else lhs_min <= rhs + 1e-8
                if operator == "<="
                else lhs_max >= rhs - 1e-8
            )
            if not feasible:
                raise ValueError(
                    "A composition element constraint is infeasible within the "
                    "configured component bounds."
                )

        if not any(
            config.get("variable_total")
            for config in self.composition_sites.values()
        ):
            totals = {
                site: float(config["total"])
                for site, config in self.composition_sites.items()
            }
            raw = {
                (site, element): totals[site] / len(config["elements"])
                for site, config in self.composition_sites.items()
                for element in config["elements"]
            }
            try:
                self._project_element_values(raw, totals)
            except ValueError as exc:
                raise ValueError(
                    "The fixed site totals, component bounds, active-element "
                    "limits, steps, and composition element constraints are "
                    "jointly infeasible."
                ) from exc

    def _constraint_sites(self) -> set[str]:
        return {
            term["site"]
            for constraint in self.composition_element_constraints
            for term in constraint["terms"]
        }

    def _support_options(
        self,
        site: str,
        config: Mapping[str, Any],
        raw: Mapping[tuple[str, str], float],
        total: float,
        *,
        enumerate_alternatives: bool,
    ) -> list[tuple[str, ...]]:
        elements = tuple(config["elements"])
        tolerance = 1e-8
        required = set(config["required_components"])
        activatable: list[str] = []
        for element in elements:
            lower, upper = self._component_bounds(config, element, total)
            if lower > tolerance:
                required.add(element)
            step = config["steps"].get(element)
            if upper > tolerance and (
                step is None or lower > tolerance or upper + tolerance >= float(step)
            ):
                activatable.append(element)

        minimum = int(config["min_components"])
        maximum = int(config["max_components"])
        ranked = sorted(
            activatable,
            key=lambda element: raw.get((site, element), 0.0),
            reverse=True,
        )
        current = {
            element
            for element in activatable
            if raw.get((site, element), 0.0) > tolerance
        }
        current.update(required)
        removable = sorted(
            current - required,
            key=lambda element: raw.get((site, element), 0.0),
        )
        while len(current) > maximum and removable:
            current.remove(removable.pop(0))
        for element in ranked:
            if len(current) >= minimum:
                break
            current.add(element)
        if not required.issubset(current) or not minimum <= len(current) <= maximum:
            raise ValueError(
                f"No valid active-element support is available at site {site!r}."
            )

        options: set[tuple[str, ...]] = {
            tuple(element for element in elements if element in current)
        }
        if enumerate_alternatives:
            optional = [element for element in activatable if element not in required]
            for size in range(max(minimum, len(required)), maximum + 1):
                choose = size - len(required)
                for selected in combinations(optional, choose):
                    support = required | set(selected)
                    options.add(
                        tuple(element for element in elements if element in support)
                    )

        scored = sorted(
            options,
            key=lambda support: sum(
                raw.get((site, element), 0.0) for element in support
            ),
            reverse=True,
        )
        return scored[: min(64, self.composition_constraint_max_supports)]

    def _entry_order(self) -> list[tuple[str, str]]:
        return [
            (site, element)
            for site, config in self.composition_sites.items()
            for element in config["elements"]
        ]

    def _solve_support(
        self,
        raw: Mapping[tuple[str, str], float],
        totals: Mapping[str, float],
        supports: Mapping[str, Sequence[str]],
    ) -> tuple[dict[tuple[str, str], float], float] | None:
        entries = self._entry_order()
        n = len(entries)
        offsets = np.zeros(n, dtype=float)
        scales = np.ones(n, dtype=float)
        lower_variables = np.zeros(2 * n, dtype=float)
        upper_variables = np.full(2 * n, np.inf, dtype=float)
        integrality = np.zeros(2 * n, dtype=int)
        objective = np.zeros(2 * n, dtype=float)

        for index, (site, element) in enumerate(entries):
            config = self.composition_sites[site]
            total = float(totals[site])
            active = element in supports[site]
            if not active:
                upper_variables[index] = 0.0
            else:
                lower, upper = self._component_bounds(config, element, total)
                step = config["steps"].get(element)
                positive_floor = max(10e-8, total * 1e-9)
                effective_lower = max(lower, float(step) if step else positive_floor)
                if effective_lower > upper + 1e-10:
                    return None
                if step is None:
                    lower_variables[index] = effective_lower
                    upper_variables[index] = upper
                else:
                    step = float(step)
                    offsets[index] = lower
                    scales[index] = step
                    lower_variables[index] = max(
                        0,
                        ceil((effective_lower - lower) / step - 1e-10),
                    )
                    upper_variables[index] = floor(
                        (upper - lower) / step + 1e-10
                    )
                    integrality[index] = 1
                    if lower_variables[index] > upper_variables[index]:
                        return None
            objective[n + index] = 1.0 / max(float(totals[site]), 1.0)

        rows: list[np.ndarray] = []
        lower_constraints: list[float] = []
        upper_constraints: list[float] = []

        for index, entry in enumerate(entries):
            raw_value = float(raw.get(entry, 0.0))
            row = np.zeros(2 * n, dtype=float)
            row[index] = scales[index]
            row[n + index] = -1.0
            rows.append(row)
            lower_constraints.append(-np.inf)
            upper_constraints.append(raw_value - offsets[index])

            row = np.zeros(2 * n, dtype=float)
            row[index] = -scales[index]
            row[n + index] = -1.0
            rows.append(row)
            lower_constraints.append(-np.inf)
            upper_constraints.append(-raw_value + offsets[index])

        for site in self.composition_sites:
            row = np.zeros(2 * n, dtype=float)
            constant = 0.0
            for index, (entry_site, _element) in enumerate(entries):
                if entry_site != site:
                    continue
                row[index] = scales[index]
                constant += offsets[index]
            rhs = float(totals[site]) - constant
            rows.append(row)
            lower_constraints.append(rhs)
            upper_constraints.append(rhs)

        entry_indices = {entry: index for index, entry in enumerate(entries)}
        for constraint in self.composition_element_constraints:
            row = np.zeros(2 * n, dtype=float)
            constant = 0.0
            for term in constraint["terms"]:
                entry = (term["site"], term["element"])
                index = entry_indices[entry]
                config = self.composition_sites[term["site"]]
                factor = float(term["coefficient"]) * self._basis_scale(
                    config,
                    term["element"],
                    constraint["basis"],
                )
                row[index] += factor * scales[index]
                constant += factor * offsets[index]
            rhs = float(constraint["rhs"]) - constant
            operator = constraint["operator"]
            rows.append(row)
            if operator == "=":
                lower_constraints.append(rhs)
                upper_constraints.append(rhs)
            elif operator == "<=":
                lower_constraints.append(-np.inf)
                upper_constraints.append(rhs)
            else:
                lower_constraints.append(rhs)
                upper_constraints.append(np.inf)

        matrix = np.vstack(rows)
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower_variables, upper_variables),
            constraints=LinearConstraint(
                matrix,
                np.asarray(lower_constraints, dtype=float),
                np.asarray(upper_constraints, dtype=float),
            ),
            options={"presolve": True, "time_limit": 10.0},
        )
        if not result.success or result.x is None:
            return None
        values = offsets + scales * np.asarray(result.x[:n], dtype=float)
        values[np.abs(values) < 1e-10] = 0.0
        return dict(zip(entries, values, strict=True)), float(result.fun)

    def _project_element_values(
        self,
        raw: Mapping[tuple[str, str], float],
        totals: Mapping[str, float],
    ) -> dict[tuple[str, str], float]:
        constrained_sites = self._constraint_sites()
        site_options: list[tuple[str, list[tuple[str, ...]]]] = []
        for site, config in self.composition_sites.items():
            options = self._support_options(
                site,
                config,
                raw,
                float(totals[site]),
                enumerate_alternatives=site in constrained_sites,
            )
            site_options.append((site, options))

        products = product(*(options for _site, options in site_options))
        best_values: dict[tuple[str, str], float] | None = None
        best_score = np.inf
        for support_tuple in islice(
            products,
            self.composition_constraint_max_supports,
        ):
            supports = {
                site: support
                for (site, _options), support in zip(
                    site_options,
                    support_tuple,
                    strict=True,
                )
            }
            solved = self._solve_support(raw, totals, supports)
            if solved is None:
                continue
            values, score = solved
            if score < best_score:
                best_values = values
                best_score = score
        if best_values is None:
            raise ValueError(
                "No composition satisfies the element constraints together with "
                "the configured totals, bounds, active-element limits, required "
                "elements, and steps."
            )
        return best_values

    def _row_native_values(
        self,
        restored: Any,
        row_index: Any,
    ) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
        raw: dict[tuple[str, str], float] = {}
        totals: dict[str, float] = {}
        for site, config in self.composition_sites.items():
            transformer = self.composition_transformers_[site]
            if config.get("variable_total"):
                total = float(restored.at[row_index, config["total_feature"]])
            else:
                total = float(config["total"])
            totals[site] = total
            for element in config["elements"]:
                if config.get("input_kind") == "element_columns":
                    column = config["element_columns"][element]
                    value = float(restored.at[row_index, column])
                else:
                    column = f"{transformer.prefix}__fraction__{element}"
                    value = float(restored.at[row_index, column]) * total
                raw[(site, element)] = value
        return raw, totals

    def _write_row_native_values(
        self,
        restored: Any,
        row_index: Any,
        values: Mapping[tuple[str, str], float],
    ) -> None:
        for site, config in self.composition_sites.items():
            transformer = self.composition_transformers_[site]
            elements = tuple(config["elements"])
            absolute = np.asarray(
                [values[(site, element)] for element in elements],
                dtype=float,
            )
            total = float(absolute.sum())
            if config.get("variable_total"):
                restored.at[row_index, config["total_feature"]] = total
            fractions = absolute / total
            for index, element in enumerate(elements):
                fraction_column = f"{transformer.prefix}__fraction__{element}"
                if fraction_column in restored.columns:
                    restored.at[row_index, fraction_column] = fractions[index]
                if config.get("input_kind") == "element_columns":
                    output_column = config["element_columns"][element]
                    restored.at[row_index, output_column] = absolute[index]

            if config.get("input_kind") == "formula":
                if (
                    str(config["normalization"]).lower()
                    in _WEIGHT_NORMALIZATIONS
                ):
                    weights = np.asarray(
                        [ATOMIC_WEIGHTS[element] for element in elements],
                        dtype=float,
                    )
                    atomic_fractions = close_compositions(
                        fractions[None, :] / weights[None, :]
                    )[0]
                else:
                    atomic_fractions = close_compositions(fractions[None, :])[0]
                restored.at[row_index, config["column"]] = format_formula(
                    dict(zip(elements, atomic_fractions, strict=True)),
                    order=elements,
                    precision=config["precision"],
                )

    def _repair_element_constraint_frame(self, restored: Any) -> Any:
        repaired = restored.copy()
        for row_index in repaired.index:
            raw, totals = self._row_native_values(repaired, row_index)
            values = self._project_element_values(raw, totals)
            self._write_row_native_values(repaired, row_index, values)
        return repaired


__all__ = [
    "CompositionElementConstraintProjector",
    "CompositionElementConstraintResolver",
]
