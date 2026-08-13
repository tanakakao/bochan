"""Composition element-constraint resolution for tabular optimization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .composition import ATOMIC_WEIGHTS

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
    """Normalize, validate, and translate element-level composition constraints."""

    @staticmethod
    def normalize(constraints: Sequence[Any] | None) -> list[dict[str, Any]]:
        """Normalize user-facing element constraints to one internal representation."""

        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(constraints or ()):
            if not isinstance(raw, Mapping):
                raise TypeError("Each composition element constraint must be a mapping.")
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
        """Resolve native component bounds for one site total."""

        pair = config["bounds"].get(element, (0.0, total))
        lower, upper = map(float, pair)
        return max(0.0, lower), min(float(total), upper)

    @staticmethod
    def basis_scale(
        config: Mapping[str, Any],
        element: str,
        basis: str,
    ) -> float:
        """Convert a native composition amount to the requested constraint basis."""

        native_is_weight = (
            str(config["normalization"]).lower() in _WEIGHT_NORMALIZATIONS
        )
        if basis == "atomic_amount":
            return 1.0 / ATOMIC_WEIGHTS[element] if native_is_weight else 1.0
        return 1.0 if native_is_weight else ATOMIC_WEIGHTS[element]

    @classmethod
    def validate(
        cls,
        constraints: Sequence[Mapping[str, Any]],
        composition_sites: Mapping[str, Mapping[str, Any]],
        *,
        project_values: Callable[
            [Mapping[tuple[str, str], float], Mapping[str, float]],
            Mapping[tuple[str, str], float],
        ]
        | None = None,
    ) -> None:
        """Validate references, individual feasibility, and fixed-total joint feasibility."""

        if not constraints:
            return
        if not composition_sites:
            raise ValueError(
                "composition_element_constraints requires composition_sites."
            )

        for constraint in constraints:
            lhs_min = 0.0
            lhs_max = 0.0
            for term in constraint["terms"]:
                site = term["site"]
                if site not in composition_sites:
                    raise KeyError(
                        f"Unknown composition site {site!r} in element constraint."
                    )
                config = composition_sites[site]
                element = term["element"]
                if element not in config["elements"]:
                    raise KeyError(
                        f"Unknown element {element!r} at composition site {site!r}."
                    )
                if config.get("variable_total"):
                    total_upper = float(config["total_bounds"][1])
                else:
                    total_upper = float(config["total"])
                lower, upper = cls.component_bounds(config, element, total_upper)
                scale = cls.basis_scale(config, element, constraint["basis"])
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

        if project_values is None or any(
            config.get("variable_total") for config in composition_sites.values()
        ):
            return

        totals = {
            site: float(config["total"])
            for site, config in composition_sites.items()
        }
        raw = {
            (site, element): totals[site] / len(config["elements"])
            for site, config in composition_sites.items()
            for element in config["elements"]
        }
        try:
            project_values(raw, totals)
        except ValueError as exc:
            raise ValueError(
                "The fixed site totals, component bounds, active-element limits, "
                "steps, and composition element constraints are jointly infeasible."
            ) from exc

    @classmethod
    def named_constraints(
        cls,
        constraints: Sequence[Mapping[str, Any]],
        composition_sites: Mapping[str, Mapping[str, Any]],
        composition_transformers: Mapping[str, Any],
    ) -> list[tuple[Any, ...]]:
        """Translate compatible fixed-fraction constraints to model-feature names."""

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


__all__ = ["CompositionElementConstraintResolver"]
