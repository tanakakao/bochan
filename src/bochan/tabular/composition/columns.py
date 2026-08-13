"""Element-column input transforms for multi-site composition optimization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .formula import ATOMIC_WEIGHTS, format_formula
from .simplex import close_compositions

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
_WEIGHT_BASES = {"weight_fraction", "weight", "mass_fraction", "wt%"}


class CompositionElementColumnTransform:
    """Convert element-value columns to and from internal formula coordinates."""

    @staticmethod
    def normalize_sites(
        sites: Mapping[str, Mapping[str, Any]] | None,
        *,
        base_normalizer: Callable[
            [Mapping[str, Mapping[str, Any]] | None], dict[str, dict[str, Any]]
        ],
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
                        f"Composition site {name!r} supplies both 'input_basis' and 'normalization'."
                    )
                basis = str(config.pop("input_basis")).lower()
                if basis not in _INPUT_BASIS_ALIASES:
                    raise ValueError(f"Unknown input_basis {basis!r} for site {name!r}.")
                config["normalization"] = _INPUT_BASIS_ALIASES[basis]

            element_columns = config.pop("element_columns", None)
            formula_column = config.get("column")
            if element_columns is None:
                resolved = base_normalizer({name: config})[name]
                resolved["input_kind"] = "formula"
                resolved["input_basis"] = None
                resolved["element_columns"] = None
                normalized[name] = resolved
                source_columns.append(str(resolved["column"]))
                continue

            if formula_column is not None:
                raise ValueError(
                    f"Composition site {name!r} must specify exactly one of 'column' or 'element_columns'."
                )
            if not isinstance(element_columns, Mapping) or not element_columns:
                raise ValueError(
                    f"Composition site {name!r} requires a non-empty element_columns mapping."
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
                        f"Composition site {name!r} elements must match the element_columns keys."
                    )
                mapped_columns = {
                    element: mapped_columns[element] for element in elements
                }
            if len(elements) < 2:
                raise ValueError(
                    f"Composition site {name!r} requires at least two elements."
                )

            config["column"] = f"__bochan_{name}_composition_formula__"
            config["elements"] = elements
            resolved = base_normalizer({name: config})[name]
            native_basis = str(resolved["normalization"]).lower()
            resolved["input_kind"] = "element_columns"
            resolved["input_basis"] = _INPUT_BASIS_ALIASES.get(
                native_basis,
                native_basis,
            )
            # Element-column values are converted to atomic fractions before the
            # formula transformer. Keep the model coordinate basis atomic and
            # retain the source basis separately in ``input_basis``.
            resolved["normalization"] = "atomic_fraction"
            resolved["element_columns"] = mapped_columns
            normalized[name] = resolved
            source_columns.extend(
                str(column) for column in mapped_columns.values()
            )

        if len(set(source_columns)) != len(source_columns):
            raise ValueError(
                "Formula and element input columns must be unique across composition sites."
            )
        return normalized

    @staticmethod
    def source_columns(config: Mapping[str, Any]) -> list[Any]:
        if config.get("input_kind") == "element_columns":
            return list(config["element_columns"].values())
        return [config["column"]]

    @staticmethod
    def numeric_site_values(
        frame: Any,
        site_name: str,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        columns = list(config["element_columns"].values())
        values = frame.loc[:, columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Element columns for site {site_name!r} must contain finite values."
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

    @staticmethod
    def _input_basis(config: Mapping[str, Any]) -> str:
        return str(
            config.get("input_basis") or config.get("normalization", "atomic_fraction")
        ).lower()

    @classmethod
    def to_atomic_fractions(
        cls,
        values: Any,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        """Convert native element-column values to atomic fractions."""

        fractions = close_compositions(np.asarray(values, dtype=float))
        if cls._input_basis(config) in _WEIGHT_BASES:
            weights = np.asarray(
                [ATOMIC_WEIGHTS[element] for element in config["elements"]],
                dtype=float,
            )
            fractions = close_compositions(fractions / weights)
        return fractions

    @classmethod
    def from_atomic_fractions(
        cls,
        values: Any,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        """Convert atomic fractions to the native element-column basis."""

        fractions = close_compositions(np.asarray(values, dtype=float))
        if cls._input_basis(config) in _WEIGHT_BASES:
            weights = np.asarray(
                [ATOMIC_WEIGHTS[element] for element in config["elements"]],
                dtype=float,
            )
            fractions = close_compositions(fractions * weights)
        return fractions

    @classmethod
    def with_internal_formula_columns(
        cls,
        data: Any,
        *,
        composition_sites: Mapping[str, Mapping[str, Any]],
        composition_transformers: Mapping[str, Any],
    ) -> Any:
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError("composition_sites requires pandas DataFrame input.")
        prepared = data.copy()
        for site_name, config in composition_sites.items():
            if config.get("input_kind") != "element_columns":
                continue
            source_columns = cls.source_columns(config)
            missing = [column for column in source_columns if column not in prepared]
            transformer = composition_transformers.get(site_name)
            model_names = (
                set(transformer.feature_names_ or ()) if transformer else set()
            )
            if missing:
                if model_names and model_names.issubset(prepared.columns):
                    continue
                raise KeyError(
                    f"Missing element columns for site {site_name!r}: {missing!r}."
                )

            values = cls.numeric_site_values(prepared, site_name, config)
            atomic_fractions = cls.to_atomic_fractions(values, config)
            prepared[config["column"]] = [
                format_formula(
                    dict(zip(config["elements"], row, strict=True)),
                    order=config["elements"],
                    precision=15,
                    omit_one=False,
                    zero_tolerance=0.0,
                )
                for row in atomic_fractions
            ]
        return prepared

    @classmethod
    def prepare_multi_site_frame(
        cls,
        data: Any,
        *,
        fit_transformers: bool,
        composition_sites: Mapping[str, Mapping[str, Any]],
        composition_transformers: Mapping[str, Any],
        base_prepare: Callable[..., Any],
    ) -> Any:
        prepared = cls.with_internal_formula_columns(
            data,
            composition_sites=composition_sites,
            composition_transformers=composition_transformers,
        )
        transformed = base_prepare(prepared, fit_transformers=fit_transformers)
        drop_columns = [
            column
            for config in composition_sites.values()
            if config.get("input_kind") == "element_columns"
            for column in cls.source_columns(config)
            if column in transformed
        ]
        return transformed.drop(columns=drop_columns, errors="ignore")

    @classmethod
    def replace_input_cols(
        cls,
        input_cols: Sequence[Any] | None,
        *,
        composition_sites: Mapping[str, Mapping[str, Any]],
        composition_transformers: Mapping[str, Any],
    ) -> list[Any] | None:
        if input_cols is None:
            return None
        source_to_site = {
            str(column): site_name
            for site_name, config in composition_sites.items()
            for column in cls.source_columns(config)
        }
        inserted: set[str] = set()
        resolved: list[Any] = []
        for column in input_cols:
            site_name = source_to_site.get(str(column))
            if site_name is None:
                resolved.append(column)
            elif site_name not in inserted:
                resolved.extend(
                    composition_transformers[site_name].feature_names_ or ()
                )
                inserted.add(site_name)
        return resolved

    @classmethod
    def clean_bounds(
        cls,
        expanded: Any,
        *,
        composition_sites: Mapping[str, Mapping[str, Any]],
    ) -> Any:
        for config in composition_sites.values():
            if config.get("input_kind") != "element_columns":
                continue
            for column in cls.source_columns(config):
                expanded.pop(column, None)
                expanded.pop(str(column), None)
        return expanded

    @classmethod
    def resolve_categorical_cols(
        cls,
        categorical_cols: Sequence[Any] | None,
        *,
        default_categorical_cols: Sequence[Any],
        composition_sites: Mapping[str, Mapping[str, Any]],
    ) -> list[Any]:
        source_categorical = (
            list(categorical_cols)
            if categorical_cols is not None
            else list(default_categorical_cols)
        )
        source_columns = {
            str(column)
            for config in composition_sites.values()
            for column in cls.source_columns(config)
        }
        return [
            column
            for column in source_categorical
            if str(column) not in source_columns
        ]

    @classmethod
    def inverse_compositions(
        cls,
        restored: Any,
        *,
        composition_sites: Mapping[str, Mapping[str, Any]],
        composition_transformers: Mapping[str, Any],
        enabled: bool,
    ) -> Any:
        if not enabled:
            return restored

        for site_name, config in composition_sites.items():
            if config.get("input_kind") != "element_columns":
                continue
            transformer = composition_transformers[site_name]
            elements = tuple(config["elements"])
            fraction_columns = [
                f"{transformer.prefix}__fraction__{element}"
                for element in elements
            ]
            atomic_fractions = restored.loc[:, fraction_columns].to_numpy(
                dtype=float
            )
            native_fractions = cls.from_atomic_fractions(
                atomic_fractions,
                config,
            )
            total = float(config["total"])
            for index, element in enumerate(elements):
                restored[config["element_columns"][element]] = (
                    native_fractions[:, index] * total
                )
            restored = restored.drop(
                columns=[config["column"], *fraction_columns],
                errors="ignore",
            )
        return restored


__all__ = ["CompositionElementColumnTransform"]
