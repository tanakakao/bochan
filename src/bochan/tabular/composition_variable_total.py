"""Variable-total composition transforms for tabular optimization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .composition import (
    ATOMIC_WEIGHTS,
    CompositionSearchSpace,
    close_compositions,
    format_formula,
    parse_formula,
)


class CompositionVariableTotalTransformer:
    """Own variable-total site normalization and model/native transforms."""

    @staticmethod
    def normalize_sites(
        sites: Mapping[str, Mapping[str, Any]] | None,
        *,
        base_normalizer: Callable[[Mapping[str, Mapping[str, Any]] | None], dict[str, dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        """Normalize ``total_bounds`` and add one total feature per variable site."""

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

        normalized = base_normalizer(prepared)
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
    def make_site_search_space(
        config: Mapping[str, Any],
        *,
        base_factory: Callable[[Mapping[str, Any]], CompositionSearchSpace | None],
    ) -> CompositionSearchSpace | None:
        """Return no fixed-total search space for a variable-total site."""

        if config.get("variable_total"):
            return None
        return base_factory(config)

    @staticmethod
    def formula_site_totals(
        formulas: Any,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        """Compute native site totals from formula values."""

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

    def site_totals_from_frame(
        self,
        owner: Any,
        data: Any,
        site_name: str,
        config: Mapping[str, Any],
    ) -> np.ndarray | None:
        """Extract one variable site's total feature from native tabular input."""

        total_feature = config["total_feature"]
        if total_feature in data.columns:
            values = data.loc[:, total_feature].to_numpy(dtype=float)
        elif config.get("input_kind") == "element_columns":
            source_columns = owner._site_source_columns(config)
            if not all(column in data.columns for column in source_columns):
                return None
            values = owner._numeric_site_values(data, site_name, config).sum(axis=1)
        else:
            column = config["column"]
            if column not in data.columns:
                return None
            values = self.formula_site_totals(data.loc[:, column], config)

        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(
                f"Composition totals for site {site_name!r} must be finite and positive."
            )
        return values

    def collect_totals(self, owner: Any, data: Any) -> dict[str, np.ndarray]:
        """Collect available variable-total values before composition transformation."""

        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError("composition_sites requires pandas DataFrame input.")

        totals: dict[str, np.ndarray] = {}
        for site_name, config in owner.composition_sites.items():
            if not config.get("variable_total"):
                continue
            values = self.site_totals_from_frame(owner, data, site_name, config)
            if values is not None:
                totals[site_name] = values
        return totals

    @staticmethod
    def inject_totals(
        owner: Any,
        transformed: Any,
        totals: Mapping[str, np.ndarray],
    ) -> Any:
        """Append collected total features to model-space data."""

        for site_name, values in totals.items():
            feature = owner.composition_sites[site_name]["total_feature"]
            transformed.loc[:, feature] = values
        return transformed

    @staticmethod
    def replace_input_cols(
        owner: Any,
        input_cols: Sequence[Any] | None,
    ) -> list[Any] | None:
        """Replace native composition inputs with coordinates plus total features."""

        if input_cols is None:
            return None
        source_to_site = {
            str(column): site_name
            for site_name, config in owner.composition_sites.items()
            for column in owner._site_source_columns(config)
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
            resolved.extend(owner.composition_transformers_[site_name].feature_names_ or ())
            config = owner.composition_sites[site_name]
            if config.get("variable_total"):
                resolved.append(config["total_feature"])
            inserted.add(site_name)
        return resolved

    @staticmethod
    def complete_bounds(owner: Any, expanded: Any) -> Any:
        """Add total and fraction-coordinate bounds for variable-total sites."""

        for site_name, config in owner.composition_sites.items():
            if not config.get("variable_total"):
                continue
            expanded[config["total_feature"]] = list(config["total_bounds"])
            representation = str(config["representation"]).lower()
            if representation not in {"none", "fractions"}:
                continue
            transformer = owner.composition_transformers_[site_name]
            elements = transformer._require_fitted()
            for feature_name in transformer._representation_names(elements):
                expanded[feature_name] = [0.0, 1.0]
        return expanded

    @staticmethod
    def dynamic_search_space(
        config: Mapping[str, Any],
        total: float,
    ) -> CompositionSearchSpace:
        """Build a native composition search space for one concrete total."""

        return CompositionSearchSpace(
            components=config["elements"],
            total=float(total),
            bounds=config["bounds"],
            steps=config["steps"],
            min_active_components=config["min_components"],
            max_active_components=config["max_components"],
            required_components=config["required_components"],
        )

    def restore(
        self,
        owner: Any,
        data: Any,
        restored: Any,
        *,
        repair: bool,
    ) -> Any:
        """Restore variable-total model coordinates to native compositions."""

        import pandas as pd

        if not owner.multi_site_composition_enabled:
            return restored

        candidates = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        for site_name, config in owner.composition_sites.items():
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
            transformer = owner.composition_transformers_[site_name]
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
                    composition = self.dynamic_search_space(
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


__all__ = ["CompositionVariableTotalTransformer"]
