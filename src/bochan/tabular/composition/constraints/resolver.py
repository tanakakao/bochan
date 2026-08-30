"""Composition element-constraint normalization and model-coordinate resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bochan.composition import ATOMIC_WEIGHTS

_WEIGHT_NORMALIZATIONS = {"weight_fraction", "weight", "mass_fraction", "wt%"}
_LOG_RATIO_REPRESENTATIONS = {"clr", "alr", "ilr"}
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
    """Normalize element constraints and map compatible ones to decision features."""

    @staticmethod
    def normalize(
        constraints: Sequence[Any] | None,
    ) -> list[dict[str, Any]]:
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
                        f"Term {term_index} in composition element constraint {index} must be a mapping."
                    )
                site = raw_term.get("site")
                element = raw_term.get("element")
                coefficient = raw_term.get("coefficient", 1.0)
                if site is None or element is None:
                    raise ValueError(
                        f"Term {term_index} in composition element constraint {index} requires site and element."
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
                raise ValueError(f"Unknown composition element operator {operator!r}.")
            rhs = float(raw.get("rhs", 0.0))
            if not np.isfinite(rhs):
                raise ValueError("Element-constraint rhs must be finite.")
            basis_name = str(raw.get("basis", "atomic_amount")).lower()
            try:
                basis = _BASIS_ALIASES[basis_name]
            except KeyError as exc:
                raise ValueError(
                    "Element-constraint basis must be 'atomic_amount' or 'weight_amount'."
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
    def native_basis(config: Mapping[str, Any]) -> str:
        """Return the basis used by native composition values."""

        if config.get("input_kind") == "element_columns":
            return str(
                config.get("input_basis") or config.get("normalization", "atomic_fraction")
            ).lower()
        return str(config["normalization"]).lower()

    @classmethod
    def basis_scale(
        cls,
        config: Mapping[str, Any],
        element: str,
        basis: str,
    ) -> float:
        native_is_weight = cls.native_basis(config) in _WEIGHT_NORMALIZATIONS
        if basis == "atomic_amount":
            return 1.0 / ATOMIC_WEIGHTS[element] if native_is_weight else 1.0
        return 1.0 if native_is_weight else ATOMIC_WEIGHTS[element]

    @staticmethod
    def _uses_fraction_decision_features(config: Mapping[str, Any]) -> bool:
        representation = str(config["representation"]).lower()
        if representation in {"fraction", "fractions"}:
            return True
        return (
            representation in _LOG_RATIO_REPRESENTATIONS
            and str(config.get("support_selection", "repair")).lower()
            == "best_subset"
        )

    @classmethod
    def named_constraints(
        cls,
        constraints: Sequence[Mapping[str, Any]],
        composition_sites: Mapping[str, Mapping[str, Any]],
        composition_transformers: Mapping[str, Any],
    ) -> list[tuple[Any, ...]]:
        """Translate constraints to fraction decision features when available.

        Fraction representations expose these names directly. CLR/ALR/ILR
        best-subset search exposes the same names in its synthetic raw-fraction
        decision space, so element constraints can be optimized jointly with the
        selected support instead of being imposed by a post-hoc repair.
        """

        if not constraints:
            return []
        resolved: list[tuple[Any, ...]] = []
        for constraint in constraints:
            names: list[str] = []
            coefficients: list[float] = []
            compatible = True
            for term in constraint["terms"]:
                config = composition_sites[term["site"]]
                if config.get("variable_total") or not cls._uses_fraction_decision_features(
                    config
                ):
                    compatible = False
                    break
                if (
                    config.get("input_kind") == "element_columns"
                    and cls.native_basis(config) in _WEIGHT_NORMALIZATIONS
                ):
                    compatible = False
                    break
                transformer = composition_transformers.get(term["site"])
                if transformer is None:
                    compatible = False
                    break
                names.append(f"{transformer.prefix}__fraction__{term['element']}")
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
