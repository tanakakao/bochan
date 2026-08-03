"""Element-column input support for multi-site composition optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .composition import ATOMIC_WEIGHTS, format_formula
from .multi_site_composition_optimizer import (
    TabularBayesianOptimizer as _FormulaMultiSiteTabularBayesianOptimizer,
)


_INPUT_BASIS_ALIASES = {
    "atomic": "atomic_fraction",
    "atomic_fraction": "atomic_fraction",
    "at_fraction": "atomic_fraction",
    "at%": "atomic_fraction",
    "molar": "atomic_fraction",
    "mole_fraction": "atomic_fraction",
    "weight": "weight_fraction",
    "weight_fraction": "weight_fraction",
    "mass_fraction": "weight_fraction",
    "wt%": "weight_fraction",
    "raw": "none",
    "amount": "none",
    "stoichiometric": "none",
    "none": "none",
}


class TabularBayesianOptimizer(_FormulaMultiSiteTabularBayesianOptimizer):
    """Support formula columns or element-value columns at each site.

    ``element_columns`` maps element symbols to existing numeric DataFrame
    columns, for example ``{"Fe": "Fe", "Ti": "Ti"}``. The values are
    closed row-wise before Fraction, CLR, ALR, or ILR transformation.
    """

    @staticmethod
    def _normalize_composition_sites(
        sites: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        if not sites:
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        source_columns: list[str] = []
        for raw_name, raw_config in sites.items():
            name = str(raw_name)
            if not isinstance(raw_config, Mapping):
                raise TypeError(f"Composition site {name!r} must be a mapping.")
            config = dict(raw_config)

            if "input_basis" in config:
                if "normalization" in config:
                    raise ValueError(
                        f"Composition site {name!r} supplies both 'input_basis' "
                        "and 'normalization'."
                    )
                basis = str(config.pop("input_basis")).lower()
                if basis not in _INPUT_BASIS_ALIASES:
                    raise ValueError(
                        f"Unknown input_basis {basis!r} for site {name!r}."
                    )
                config["normalization"] = _INPUT_BASIS_ALIASES[basis]

            element_columns = config.pop("element_columns", None)
            formula_column = config.get("column")
            if element_columns is None:
                resolved = (
                    _FormulaMultiSiteTabularBayesianOptimizer
                    ._normalize_composition_sites({name: config})[name]
                )
                resolved["input_kind"] = "formula"
                resolved["element_columns"] = None
                normalized[name] = resolved
                source_columns.append(str(resolved["column"]))
                continue

            if formula_column is not None:
                raise ValueError(
                    f"Composition site {name!r} must specify exactly one of "
                    "'column' or 'element_columns'."
                )
            if not isinstance(element_columns, Mapping) or not element_columns:
                raise ValueError(
                    f"Composition site {name!r} requires a non-empty "
                    "element_columns mapping."
                )
            mapped_columns = {
                str(element): column for element, column in element_columns.items()
            }
            if len(set(map(str, mapped_columns.values()))) != len(mapped_columns):
                raise ValueError(
                    f"Composition site {name!r} must use unique element columns."
                )

            configured_elements = config.get("elements")
            if configured_elements is None:
                elements = tuple(mapped_columns)
            else:
                elements = tuple(dict.fromkeys(configured_elements))
                if set(elements) != set(mapped_columns):
                    raise ValueError(
                        f"Composition site {name!r} elements must match the "
                        "element_columns keys."
                    )
                mapped_columns = {
                    element: mapped_columns[element] for element in elements
                }
            if len(elements) < 2:
                raise ValueError(
                    f"Composition site {name!r} requires at least two elements."
                )

            internal_column = f"__bochan_{name}_composition_formula__"
            config["column"] = internal_column
            config["elements"] = elements
            resolved = (
                _FormulaMultiSiteTabularBayesianOptimizer
                ._normalize_composition_sites({name: config})[name]
            )
            resolved["input_kind"] = "element_columns"
            resolved["element_columns"] = mapped_columns
            normalized[name] = resolved
            source_columns.extend(str(column) for column in mapped_columns.values())

        if len(set(source_columns)) != len(source_columns):
            raise ValueError(
                "Formula and element input columns must be unique across "
                "composition sites."
            )
        return normalized

    @staticmethod
    def _site_source_columns(config: Mapping[str, Any]) -> list[Any]:
        if config.get("input_kind") == "element_columns":
            return list(config["element_columns"].values())
        return [config["column"]]

    @staticmethod
    def _numeric_site_values(
        frame: Any,
        site_name: str,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        columns = list(config["element_columns"].values())
        values = frame.loc[:, columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Element columns for site {site_name!r} must contain "
                "finite values."
            )
        if np.any(values < 0):
            raise ValueError(
                f"Element columns for site {site_name!r} must be non-negative."
            )
        if np.any(values.sum(axis=1) <= 0):
            raise ValueError(
                f"Every row at site {site_name!r} must have a positive total."
            )
        return values

    def _with_internal_formula_columns(self, data: Any) -> Any:
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError("composition_sites requires pandas DataFrame input.")
        prepared = data.copy()
        for site_name, config in self.composition_sites.items():
            if config.get("input_kind") != "element_columns":
                continue
            source_columns = self._site_source_columns(config)
            missing = [column for column in source_columns if column not in prepared]
            transformer = self.composition_transformers_.get(site_name)
            model_names = (
                set(transformer.feature_names_ or ()) if transformer else set()
            )
            if missing:
                if model_names and model_names.issubset(prepared.columns):
                    continue
                raise KeyError(
                    f"Missing element columns for site {site_name!r}: {missing!r}."
                )

            values = self._numeric_site_values(prepared, site_name, config)
            values = values / values.sum(axis=1, keepdims=True)
            if str(config["normalization"]).lower() == "weight_fraction":
                weights = np.asarray(
                    [ATOMIC_WEIGHTS[element] for element in config["elements"]],
                    dtype=float,
                )
                values = values / weights
            prepared[config["column"]] = [
                format_formula(
                    dict(zip(config["elements"], row, strict=True)),
                    order=config["elements"],
                    precision=15,
                    omit_one=False,
                    zero_tolerance=0.0,
                )
                for row in values
            ]
        return prepared

    def _prepare_multi_site_frame(
        self,
        data: Any,
        *,
        fit_transformers: bool,
    ) -> Any:
        prepared = self._with_internal_formula_columns(data)
        transformed = super()._prepare_multi_site_frame(
            prepared,
            fit_transformers=fit_transformers,
        )
        drop_columns = [
            column
            for config in self.composition_sites.values()
            if config.get("input_kind") == "element_columns"
            for column in self._site_source_columns(config)
            if column in transformed
        ]
        return transformed.drop(columns=drop_columns, errors="ignore")

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
            elif site_name not in inserted:
                resolved.extend(
                    self.composition_transformers_[site_name].feature_names_ or ()
                )
                inserted.add(site_name)
        return resolved

    def _expanded_multi_site_bounds(self, bounds: Any, transformed: Any) -> Any:
        expanded = super()._expanded_multi_site_bounds(bounds, transformed)
        for config in self.composition_sites.values():
            if config.get("input_kind") != "element_columns":
                continue
            for column in self._site_source_columns(config):
                expanded.pop(column, None)
                expanded.pop(str(column), None)
        return expanded

    def fit(
        self,
        data: Any | None = None,
        y: Any | None = None,
        *,
        input_cols: Sequence[Any] | None = None,
        categorical_cols: Sequence[Any] | None = None,
        bounds: Any = None,
        **kwargs: Any,
    ) -> "TabularBayesianOptimizer":
        if not self.multi_site_composition_enabled:
            return super().fit(
                data=data,
                y=y,
                input_cols=input_cols,
                categorical_cols=categorical_cols,
                bounds=bounds,
                **kwargs,
            )
        source_categorical = (
            list(categorical_cols)
            if categorical_cols is not None
            else list(self.data_config.categorical_cols or ())
        )
        source_columns = {
            str(column)
            for config in self.composition_sites.values()
            for column in self._site_source_columns(config)
        }
        resolved_categorical = [
            column
            for column in source_categorical
            if str(column) not in source_columns
        ]
        return super().fit(
            data=data,
            y=y,
            input_cols=input_cols,
            categorical_cols=resolved_categorical,
            bounds=bounds,
            **kwargs,
        )

    def inverse_compositions(
        self,
        data: Any,
        *,
        repair: bool = True,
        keep_coordinates: bool = False,
    ) -> Any:
        restored = super().inverse_compositions(
            data,
            repair=repair,
            keep_coordinates=keep_coordinates,
        )
        if not self.multi_site_composition_enabled:
            return restored

        for site_name, config in self.composition_sites.items():
            if config.get("input_kind") != "element_columns":
                continue
            transformer = self.composition_transformers_[site_name]
            fraction_columns: list[str] = []
            for element, output_column in config["element_columns"].items():
                fraction_column = f"{transformer.prefix}__fraction__{element}"
                restored[output_column] = (
                    restored[fraction_column] * float(config["total"])
                )
                fraction_columns.append(fraction_column)
            restored = restored.drop(
                columns=[config["column"], *fraction_columns],
                errors="ignore",
            )
        return restored
