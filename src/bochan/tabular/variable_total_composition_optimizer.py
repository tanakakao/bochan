"""Variable-total support for multi-site composition optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from bochan.api import OptimizeConfig

from .composition import (
    ATOMIC_WEIGHTS,
    CompositionSearchSpace,
    close_compositions,
    format_formula,
    parse_formula,
)
from .element_column_composition_optimizer import (
    TabularBayesianOptimizer as _ElementColumnTabularBayesianOptimizer,
)


class TabularBayesianOptimizer(_ElementColumnTabularBayesianOptimizer):
    """Support bounded and coupled composition-site totals.

    A site with ``total_bounds=(lower, upper)`` receives an additional numeric
    model feature named ``<prefix>__total``. For element-column inputs, this
    feature is learned from the row-wise sum of the site's element columns.

    ``composition_total_constraints`` can couple site totals using the same
    named linear-constraint convention as the tabular optimizer.
    """

    def __init__(
        self,
        model_config: Any | None = None,
        fit_config: Any | None = None,
        *,
        composition_sites: Mapping[str, Mapping[str, Any]] | None = None,
        composition_total_constraints: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.composition_total_constraints = self._normalize_total_constraints(
            composition_total_constraints
        )
        super().__init__(
            model_config=model_config,
            fit_config=fit_config,
            composition_sites=composition_sites,
            **kwargs,
        )
        self._validate_total_constraints()

    @staticmethod
    def _normalize_composition_sites(
        sites: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        if not sites:
            return {}

        prepared: dict[str, dict[str, Any]] = {}
        variable_settings: dict[str, tuple[tuple[float, float], str | None]] = {}
        for raw_name, raw_config in sites.items():
            name = str(raw_name)
            if not isinstance(raw_config, Mapping):
                raise TypeError(f"Composition site {name!r} must be a mapping.")
            config = dict(raw_config)
            total_bounds = config.pop("total_bounds", None)
            total_feature = config.pop("total_feature", None)

            if total_bounds is None:
                prepared[name] = config
                continue
            if "total" in config:
                raise ValueError(
                    f"Composition site {name!r} must specify either 'total' or "
                    "'total_bounds', not both."
                )
            pair = tuple(total_bounds)
            if len(pair) != 2:
                raise ValueError(
                    f"total_bounds for site {name!r} must contain two values."
                )
            lower, upper = map(float, pair)
            if (
                not np.isfinite([lower, upper]).all()
                or lower <= 0.0
                or lower >= upper
            ):
                raise ValueError(
                    f"total_bounds for site {name!r} must be finite, positive, "
                    "and increasing."
                )
            config["total"] = 0.5 * (lower + upper)
            prepared[name] = config
            variable_settings[name] = ((lower, upper), total_feature)

        normalized = (
            _ElementColumnTabularBayesianOptimizer._normalize_composition_sites(
                prepared
            )
        )
        for name, config in normalized.items():
            setting = variable_settings.get(name)
            if setting is None:
                config["variable_total"] = False
                config["total_bounds"] = None
                config["total_feature"] = None
                continue

            total_bounds, total_feature = setting
            prefix = str(config["prefix"] or name)
            config["variable_total"] = True
            config["total_bounds"] = total_bounds
            config["total_feature"] = (
                str(total_feature) if total_feature is not None else f"{prefix}__total"
            )

            lower_sum = sum(float(pair[0]) for pair in config["bounds"].values())
            if lower_sum > total_bounds[0] + 1e-12:
                raise ValueError(
                    f"Lower component bounds at site {name!r} require a total "
                    f"of at least {lower_sum}, so the lower end of "
                    f"total_bounds={total_bounds!r} is infeasible."
                )
            if all(element in config["bounds"] for element in config["elements"]):
                upper_sum = sum(
                    float(config["bounds"][element][1])
                    for element in config["elements"]
                )
                if upper_sum < total_bounds[1] - 1e-12:
                    raise ValueError(
                        f"Upper component bounds at site {name!r} allow a total "
                        f"of at most {upper_sum}, so the upper end of "
                        f"total_bounds={total_bounds!r} is infeasible."
                    )

        total_features = [
            config["total_feature"]
            for config in normalized.values()
            if config["variable_total"]
        ]
        if len(set(total_features)) != len(total_features):
            raise ValueError("Each variable composition total needs a unique feature name.")
        return normalized

    @staticmethod
    def _normalize_total_constraints(
        constraints: Sequence[Any] | None,
    ) -> list[dict[str, Any]]:
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

    def _validate_total_constraints(self) -> None:
        available = set(self.composition_sites)
        for constraint in self.composition_total_constraints:
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
                config = self.composition_sites[site]
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
    def _make_site_search_space(
        config: Mapping[str, Any],
    ) -> CompositionSearchSpace | None:
        if config.get("variable_total"):
            return None
        return _ElementColumnTabularBayesianOptimizer._make_site_search_space(config)

    @staticmethod
    def _formula_site_totals(
        formulas: Any,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        totals: list[float] = []
        weight_basis = str(config["normalization"]).lower() in {
            "weight_fraction",
            "weight",
            "mass_fraction",
        }
        for formula in formulas:
            parsed = parse_formula(str(formula))
            if weight_basis:
                total = sum(
                    float(amount) * ATOMIC_WEIGHTS[element]
                    for element, amount in parsed.items()
                )
            else:
                total = sum(float(amount) for amount in parsed.values())
            totals.append(total)
        return np.asarray(totals, dtype=float)

    def _site_totals_from_frame(
        self,
        data: Any,
        site_name: str,
        config: Mapping[str, Any],
    ) -> np.ndarray | None:
        total_feature = config["total_feature"]
        if total_feature in data.columns:
            values = data.loc[:, total_feature].to_numpy(dtype=float)
        elif config.get("input_kind") == "element_columns":
            source_columns = self._site_source_columns(config)
            if not all(column in data.columns for column in source_columns):
                return None
            values = self._numeric_site_values(data, site_name, config).sum(axis=1)
        else:
            column = config["column"]
            if column not in data.columns:
                return None
            values = self._formula_site_totals(data.loc[:, column], config)

        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(
                f"Composition totals for site {site_name!r} must be finite and positive."
            )
        return values

    def _prepare_multi_site_frame(
        self,
        data: Any,
        *,
        fit_transformers: bool,
    ) -> Any:
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError("composition_sites requires pandas DataFrame input.")

        totals: dict[str, np.ndarray] = {}
        for site_name, config in self.composition_sites.items():
            if not config.get("variable_total"):
                continue
            values = self._site_totals_from_frame(data, site_name, config)
            if values is not None:
                totals[site_name] = values

        transformed = super()._prepare_multi_site_frame(
            data,
            fit_transformers=fit_transformers,
        )
        for site_name, values in totals.items():
            feature = self.composition_sites[site_name]["total_feature"]
            transformed.loc[:, feature] = values
        return transformed

    def _replace_multi_site_input_cols(
        self,
        input_cols: Sequence[Any] | None,
    ) -> list[Any] | None:
        if input_cols is None:
            return None
        source_to_site = {
            str(column): site_name
            for site_name, config in self.composition_sites.items()
            for column in self._site_source_columns(config)
        }
        inserted: set[str] = set()
        resolved: list[Any] = []
        for column in input_cols:
            site_name = source_to_site.get(str(column))
            if site_name is None:
                resolved.append(column)
                continue
            if site_name in inserted:
                continue
            resolved.extend(
                self.composition_transformers_[site_name].feature_names_ or ()
            )
            config = self.composition_sites[site_name]
            if config.get("variable_total"):
                resolved.append(config["total_feature"])
            inserted.add(site_name)
        return resolved

    def _expanded_multi_site_bounds(self, bounds: Any, transformed: Any) -> Any:
        expanded = super()._expanded_multi_site_bounds(bounds, transformed)
        for site_name, config in self.composition_sites.items():
            if not config.get("variable_total"):
                continue
            expanded[config["total_feature"]] = list(config["total_bounds"])
            representation = str(config["representation"]).lower()
            if representation not in {"none", "fractions"}:
                continue
            transformer = self.composition_transformers_[site_name]
            elements = transformer._require_fitted()
            for feature_name in transformer._representation_names(elements):
                expanded[feature_name] = [0.0, 1.0]
        return expanded

    def _named_total_constraints(self) -> list[tuple[Any, ...]]:
        resolved: list[tuple[Any, ...]] = []
        for constraint in self.composition_total_constraints:
            names: list[str] = []
            coefficients: list[float] = []
            rhs = float(constraint["rhs"])
            for site, coefficient in zip(
                constraint["sites"],
                constraint["coefficients"],
                strict=True,
            ):
                config = self.composition_sites[site]
                if config["variable_total"]:
                    names.append(config["total_feature"])
                    coefficients.append(float(coefficient))
                else:
                    rhs -= float(coefficient) * float(config["total"])
            if names:
                resolved.append((names, coefficients, constraint["operator"], rhs))
        return resolved

    @staticmethod
    def _merge_total_constraints(
        opt_config: OptimizeConfig | Mapping[str, Any] | None,
        constraints: Sequence[tuple[Any, ...]],
    ) -> OptimizeConfig | Mapping[str, Any] | None:
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

    def candidate(
        self,
        acq_config: Any | None = None,
        opt_config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        named_constraints = self._named_total_constraints()
        opt_config = self._merge_total_constraints(opt_config, named_constraints)
        return super().candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            **kwargs,
        )

    @staticmethod
    def _dynamic_search_space(
        config: Mapping[str, Any],
        total: float,
    ) -> CompositionSearchSpace:
        return CompositionSearchSpace(
            components=config["elements"],
            total=float(total),
            bounds=config["bounds"],
            steps=config["steps"],
            min_active_components=config["min_components"],
            max_active_components=config["max_components"],
            required_components=config["required_components"],
        )

    def inverse_compositions(
        self,
        data: Any,
        *,
        repair: bool = True,
        keep_coordinates: bool = False,
    ) -> Any:
        import pandas as pd

        restored = super().inverse_compositions(
            data,
            repair=repair,
            keep_coordinates=keep_coordinates,
        )
        if not self.multi_site_composition_enabled:
            return restored

        candidates = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        for site_name, config in self.composition_sites.items():
            if not config.get("variable_total"):
                continue
            total_feature = config["total_feature"]
            if total_feature not in candidates:
                raise KeyError(
                    f"Missing total feature {total_feature!r} for site {site_name!r}."
                )

            lower, upper = config["total_bounds"]
            totals = np.clip(
                candidates.loc[:, total_feature].to_numpy(dtype=float),
                lower,
                upper,
            )
            transformer = self.composition_transformers_[site_name]
            elements = transformer._require_fitted()
            representation_names = transformer._representation_names(elements)
            missing = [
                name for name in representation_names if name not in candidates.columns
            ]
            if missing:
                raise KeyError(
                    f"Missing model-space columns for site {site_name!r}: {missing!r}."
                )

            simplex_transform = transformer.simplex_transform_
            assert simplex_transform is not None
            basis_fractions = simplex_transform.inverse_transform(
                candidates.loc[:, representation_names].to_numpy(dtype=float),
                n_components=len(elements),
            )

            rows: list[dict[str, float]] = []
            for basis_row, total in zip(basis_fractions, totals, strict=True):
                composition = {
                    element: float(value) * float(total)
                    for element, value in zip(elements, basis_row, strict=True)
                }
                if repair:
                    composition = self._dynamic_search_space(
                        config,
                        float(total),
                    ).repair(composition)
                rows.append(composition)

            absolute = np.asarray(
                [[row[element] for element in elements] for row in rows],
                dtype=float,
            )
            actual_totals = absolute.sum(axis=1)
            restored.loc[:, total_feature] = actual_totals

            if config.get("input_kind") == "element_columns":
                for index, element in enumerate(elements):
                    output_column = config["element_columns"][element]
                    restored.loc[:, output_column] = absolute[:, index]
                continue

            normalized_basis = absolute / actual_totals[:, None]
            if str(config["normalization"]).lower() in {
                "weight_fraction",
                "weight",
                "mass_fraction",
            }:
                weights = np.asarray(
                    [ATOMIC_WEIGHTS[element] for element in elements],
                    dtype=float,
                )
                atomic_fractions = close_compositions(normalized_basis / weights)
            else:
                atomic_fractions = close_compositions(normalized_basis)
            restored.loc[:, config["column"]] = [
                format_formula(
                    dict(zip(elements, row, strict=True)),
                    order=elements,
                    precision=config["precision"],
                )
                for row in atomic_fractions
            ]
            for index, element in enumerate(elements):
                fraction_column = f"{transformer.prefix}__fraction__{element}"
                if fraction_column in restored:
                    restored.loc[:, fraction_column] = normalized_basis[:, index]
        return restored


__all__ = ["TabularBayesianOptimizer"]
