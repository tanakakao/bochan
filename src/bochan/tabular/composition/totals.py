"""Composition-site total-constraint resolution for tabular optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from bochan.api import OptimizeConfig


class CompositionTotalConstraintResolver:
    """Normalize, validate, and translate coupled composition-site totals."""

    @staticmethod
    def normalize(constraints: Sequence[Any] | None) -> list[dict[str, Any]]:
        """Normalize user-facing total constraints to one internal representation."""

        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(constraints or ()):
            if isinstance(raw, Mapping):
                sites = tuple(str(site) for site in raw.get("sites", ()))
                coefficients = raw.get("coefficients")
                operator = str(raw.get("operator", raw.get("op", "=")))
                rhs = raw.get("total", raw.get("rhs"))
            else:
                values = tuple(raw)
                if len(values) == 3:
                    sites, operator, rhs = values
                    coefficients = None
                elif len(values) == 4:
                    sites, coefficients, operator, rhs = values
                else:
                    raise ValueError(
                        "Each composition total constraint must be a mapping or "
                        "a 3/4-item tuple."
                    )
                sites = tuple(str(site) for site in sites)

            if not sites:
                raise ValueError(
                    f"Composition total constraint {index} requires at least one site."
                )
            if coefficients is None:
                resolved_coefficients = tuple(1.0 for _ in sites)
            else:
                resolved_coefficients = tuple(float(value) for value in coefficients)
            if len(resolved_coefficients) != len(sites):
                raise ValueError(
                    f"Composition total constraint {index} requires one coefficient "
                    "per site."
                )
            if operator not in {"=", "==", "<=", ">="}:
                raise ValueError(f"Unknown composition total operator {operator!r}.")
            if rhs is None or not np.isfinite(float(rhs)):
                raise ValueError(
                    f"Composition total constraint {index} requires a finite total/rhs."
                )
            normalized.append(
                {
                    "sites": sites,
                    "coefficients": resolved_coefficients,
                    "operator": "=" if operator == "==" else operator,
                    "rhs": float(rhs),
                }
            )
        return normalized

    @staticmethod
    def validate(
        constraints: Sequence[Mapping[str, Any]],
        composition_sites: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Validate site references and feasibility against configured total ranges."""

        available = set(composition_sites)
        for constraint in constraints:
            unknown = set(constraint["sites"]) - available
            if unknown:
                raise KeyError(
                    "Unknown sites in composition_total_constraints: "
                    f"{sorted(unknown)!r}."
                )

            lhs_min = 0.0
            lhs_max = 0.0
            for site, coefficient in zip(
                constraint["sites"],
                constraint["coefficients"],
                strict=True,
            ):
                config = composition_sites[site]
                if config["variable_total"]:
                    lower, upper = config["total_bounds"]
                else:
                    lower = upper = float(config["total"])
                if coefficient >= 0.0:
                    lhs_min += coefficient * lower
                    lhs_max += coefficient * upper
                else:
                    lhs_min += coefficient * upper
                    lhs_max += coefficient * lower

            operator = constraint["operator"]
            rhs = constraint["rhs"]
            feasible = (
                lhs_min - 1e-8 <= rhs <= lhs_max + 1e-8
                if operator == "="
                else lhs_min <= rhs + 1e-8
                if operator == "<="
                else lhs_max >= rhs - 1e-8
            )
            if not feasible:
                raise ValueError(
                    "A composition total constraint is infeasible within the "
                    "configured fixed totals and total_bounds."
                )

    @staticmethod
    def named_constraints(
        constraints: Sequence[Mapping[str, Any]],
        composition_sites: Mapping[str, Mapping[str, Any]],
    ) -> list[tuple[Any, ...]]:
        """Translate site-level constraints to named model-feature constraints."""

        resolved: list[tuple[Any, ...]] = []
        for constraint in constraints:
            names: list[str] = []
            coefficients: list[float] = []
            rhs = float(constraint["rhs"])
            for site, coefficient in zip(
                constraint["sites"],
                constraint["coefficients"],
                strict=True,
            ):
                config = composition_sites[site]
                if config["variable_total"]:
                    names.append(config["total_feature"])
                    coefficients.append(float(coefficient))
                else:
                    rhs -= float(coefficient) * float(config["total"])
            if names:
                resolved.append((names, coefficients, constraint["operator"], rhs))
        return resolved

    @staticmethod
    def merge_optimize_config(
        opt_config: OptimizeConfig | Mapping[str, Any] | None,
        constraints: Sequence[tuple[Any, ...]],
    ) -> OptimizeConfig | Mapping[str, Any] | None:
        """Merge named total constraints into mapping or dataclass optimize configs."""

        if not constraints:
            return opt_config
        if opt_config is None:
            return {"constraints": list(constraints)}
        if isinstance(opt_config, Mapping):
            payload = dict(opt_config)
            existing = list(payload.get("constraints") or ())
            payload["constraints"] = [*existing, *constraints]
            return payload

        equalities = list(opt_config.equality_constraints or ())
        inequalities = list(opt_config.inequality_constraints or ())
        for columns, coefficients, operator, rhs in constraints:
            if operator == "=":
                equalities.append((columns, coefficients, rhs))
            elif operator == ">=":
                inequalities.append((columns, coefficients, rhs))
            else:
                inequalities.append(
                    (columns, [-float(value) for value in coefficients], -float(rhs))
                )
        return replace(
            opt_config,
            equality_constraints=equalities,
            inequality_constraints=inequalities,
        )


__all__ = ["CompositionTotalConstraintResolver"]
