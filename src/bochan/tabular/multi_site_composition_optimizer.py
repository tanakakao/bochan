"""Multi-site composition support for the public tabular optimizer API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .composition import (
    ATOMIC_WEIGHTS,
    CompositionDescriptorCalculator,
    CompositionSearchSpace,
    CompositionTransformer,
    close_compositions,
    format_formula,
)
from .composition_optimizer import (
    TabularBayesianOptimizer as _SingleSiteTabularBayesianOptimizer,
)
from .optimizer_api import TabularBayesianOptimizer as _CoreTabularBayesianOptimizer


_SITE_DEFAULTS: dict[str, Any] = {
    "normalization": "atomic_fraction",
    "representation": "ilr",
    "reference_element": None,
    "pseudocount": 1e-12,
    "include_descriptors": False,
    "descriptor_properties": ("atomic_number", "atomic_weight"),
    "descriptor_statistics": ("mean", "std", "min", "max", "range"),
    "element_properties": {},
    "prefix": None,
    "precision": 6,
    "total": 1.0,
    "bounds": {},
    "steps": {},
    "min_components": 1,
    "max_components": None,
    "required_components": (),
    "coordinate_bounds": (-8.0, 8.0),
}

_SITE_ALIASES = {
    "min_active_components": "min_components",
    "max_active_components": "max_components",
    "required_elements": "required_components",
}


class TabularBayesianOptimizer(_SingleSiteTabularBayesianOptimizer):
    """Tabular optimizer supporting one or more independent composition sites.

    ``composition_sites`` maps site names to direct settings. Each site has its
    own formula column, element pool, active-component limits, required
    elements, bounds, steps, representation, and inverse repair.

    The legacy single-composition arguments remain supported when
    ``composition_sites`` is omitted.
    """

    def __init__(
        self,
        model_config: Any | None = None,
        fit_config: Any | None = None,
        *,
        composition_sites: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        if composition_sites and (
            kwargs.get("composition_col") is not None
            or kwargs.get("formula_col") is not None
        ):
            raise ValueError(
                "Specify composition_sites or the legacy composition_col/formula_col, "
                "not both."
            )
        self.composition_sites = self._normalize_composition_sites(composition_sites)
        self.composition_transformers_: dict[str, CompositionTransformer] = {}
        self.composition_search_spaces_: dict[str, CompositionSearchSpace] = {}
        super().__init__(
            model_config=model_config,
            fit_config=fit_config,
            **kwargs,
        )

    @staticmethod
    def _normalize_composition_sites(
        sites: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        if not sites:
            return {}
        normalized: dict[str, dict[str, Any]] = {}
        allowed = {"column", "elements", *_SITE_DEFAULTS}
        for raw_name, raw_config in sites.items():
            name = str(raw_name)
            if not isinstance(raw_config, Mapping):
                raise TypeError(f"Composition site {name!r} must be a mapping.")
            config = dict(raw_config)
            for alias, canonical in _SITE_ALIASES.items():
                if alias in config:
                    if canonical in config:
                        raise ValueError(
                            f"Composition site {name!r} supplies both {alias!r} "
                            f"and {canonical!r}."
                        )
                    config[canonical] = config.pop(alias)
            unknown = set(config) - allowed
            if unknown:
                raise KeyError(
                    f"Unknown composition-site settings for {name!r}: "
                    f"{sorted(unknown)!r}."
                )
            if "column" not in config:
                raise ValueError(f"Composition site {name!r} requires a column.")
            if "elements" not in config or not config["elements"]:
                raise ValueError(
                    f"Composition site {name!r} requires one or more candidate elements."
                )
            resolved = dict(_SITE_DEFAULTS)
            resolved.update(config)
            resolved["elements"] = tuple(resolved["elements"])
            resolved["descriptor_properties"] = tuple(
                resolved["descriptor_properties"]
            )
            resolved["descriptor_statistics"] = tuple(
                resolved["descriptor_statistics"]
            )
            resolved["element_properties"] = dict(
                resolved["element_properties"] or {}
            )
            resolved["bounds"] = dict(resolved["bounds"] or {})
            resolved["steps"] = dict(resolved["steps"] or {})
            resolved["required_components"] = tuple(
                resolved["required_components"] or ()
            )
            resolved["min_components"] = int(resolved["min_components"])
            if resolved["max_components"] is not None:
                resolved["max_components"] = int(resolved["max_components"])
            resolved["total"] = float(resolved["total"])
            resolved["precision"] = int(resolved["precision"])
            resolved["pseudocount"] = float(resolved["pseudocount"])
            resolved["include_descriptors"] = bool(
                resolved["include_descriptors"]
            )
            resolved["prefix"] = resolved["prefix"] or name
            normalized[name] = resolved
        columns = [config["column"] for config in normalized.values()]
        if len(set(columns)) != len(columns):
            raise ValueError("Each composition site must use a unique formula column.")
        return normalized

    @property
    def multi_site_composition_enabled(self) -> bool:
        return bool(getattr(self, "composition_sites", {}))

    @property
    def composition_enabled(self) -> bool:
        return self.multi_site_composition_enabled or super().composition_enabled

    def _make_site_transformer(
        self,
        site_name: str,
        config: Mapping[str, Any],
        formulas: Any,
    ) -> CompositionTransformer:
        calculator = CompositionDescriptorCalculator(
            properties=config["descriptor_properties"],
            statistics=config["descriptor_statistics"],
            element_properties=config["element_properties"],
        )
        transformer = CompositionTransformer(
            elements=config["elements"],
            normalization=config["normalization"],
            representation=config["representation"],
            reference_element=config["reference_element"],
            pseudocount=config["pseudocount"],
            include_descriptors=config["include_descriptors"],
            descriptor_calculator=calculator,
            prefix=config["prefix"] or site_name,
            precision=config["precision"],
        )
        transformer.fit(formulas)
        return transformer

    @staticmethod
    def _make_site_search_space(
        config: Mapping[str, Any],
    ) -> CompositionSearchSpace:
        return CompositionSearchSpace(
            components=config["elements"],
            total=config["total"],
            bounds=config["bounds"],
            steps=config["steps"],
            min_active_components=config["min_components"],
            max_active_components=config["max_components"],
            required_components=config["required_components"],
        )

    def _prepare_multi_site_frame(
        self,
        data: Any,
        *,
        fit_transformers: bool,
    ) -> Any:
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError("composition_sites requires pandas DataFrame input.")
        transformed = data.copy()
        for site_name, config in self.composition_sites.items():
            column = config["column"]
            transformer = self.composition_transformers_.get(site_name)
            if column not in transformed.columns:
                model_names = set(transformer.feature_names_ or ()) if transformer else set()
                if model_names and model_names.issubset(transformed.columns):
                    continue
                raise KeyError(
                    f"Unknown composition column {column!r} for site {site_name!r}."
                )
            if fit_transformers or transformer is None:
                transformer = self._make_site_transformer(
                    site_name,
                    config,
                    transformed.loc[:, column],
                )
                self.composition_transformers_[site_name] = transformer
                self.composition_search_spaces_[site_name] = (
                    self._make_site_search_space(config)
                )
            transformed = transformer.transform_frame(
                transformed,
                column,
                drop_formula=True,
            )
        return transformed

    def _replace_multi_site_input_cols(
        self,
        input_cols: Sequence[Any] | None,
    ) -> list[Any] | None:
        if input_cols is None:
            return None
        column_to_features = {
            config["column"]: list(
                self.composition_transformers_[site_name].feature_names_ or ()
            )
            for site_name, config in self.composition_sites.items()
        }
        resolved: list[Any] = []
        for column in input_cols:
            replacement = next(
                (
                    features
                    for source, features in column_to_features.items()
                    if column == source or str(column) == str(source)
                ),
                None,
            )
            if replacement is None:
                resolved.append(column)
            else:
                resolved.extend(replacement)
        return resolved

    @staticmethod
    def _site_coordinate_bound(
        configured: Any,
        name: str,
        index: int,
    ) -> tuple[float, float]:
        if isinstance(configured, Mapping):
            value = configured.get(name, configured.get(index))
            if value is None:
                raise KeyError(
                    f"No composition coordinate bound was supplied for {name!r}."
                )
        else:
            value = configured
        pair = tuple(value)
        if len(pair) != 2 or float(pair[0]) >= float(pair[1]):
            raise ValueError(
                "Each composition coordinate bound must be an increasing pair."
            )
        return float(pair[0]), float(pair[1])

    def _expanded_multi_site_bounds(self, bounds: Any, transformed: Any) -> Any:
        if bounds is not None and not isinstance(bounds, Mapping):
            raise TypeError(
                "Multi-site composition optimization requires bounds to be a "
                "column mapping."
            )
        expanded = dict(bounds or {})
        for site_name, config in self.composition_sites.items():
            column = config["column"]
            expanded.pop(column, None)
            expanded.pop(str(column), None)
            transformer = self.composition_transformers_[site_name]
            elements = transformer._require_fitted()
            representation_names = transformer._representation_names(elements)
            representation = str(config["representation"]).lower()
            for index, feature_name in enumerate(representation_names):
                if feature_name in expanded:
                    continue
                if representation in {"none", "fractions"}:
                    component = elements[index]
                    pair = config["bounds"].get(
                        component,
                        (0.0, config["total"]),
                    )
                    expanded[feature_name] = [
                        float(pair[0]) / config["total"],
                        float(pair[1]) / config["total"],
                    ]
                else:
                    expanded[feature_name] = list(
                        self._site_coordinate_bound(
                            config["coordinate_bounds"],
                            feature_name,
                            index,
                        )
                    )
            descriptor_names = [
                value
                for value in (transformer.feature_names_ or ())
                if value not in representation_names
            ]
            for feature_name in descriptor_names:
                if feature_name in expanded:
                    continue
                lower = float(transformed[feature_name].min())
                upper = float(transformed[feature_name].max())
                margin = max((upper - lower) * 0.05, 1e-9)
                expanded[feature_name] = [lower - margin, upper + margin]
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
        if data is None:
            data = self.data
        if data is None:
            raise ValueError(
                "No data was supplied. Pass data to fit(...) or use from_csv(...)."
            )
        source_input_cols = (
            input_cols if input_cols is not None else self.data_config.input_cols
        )
        source_categorical = (
            categorical_cols
            if categorical_cols is not None
            else self.data_config.categorical_cols
        )
        source_bounds = bounds if bounds is not None else self.data_config.bounds
        transformed = self._prepare_multi_site_frame(
            data,
            fit_transformers=True,
        )
        resolved_input_cols = self._replace_multi_site_input_cols(source_input_cols)
        site_columns = {config["column"] for config in self.composition_sites.values()}
        resolved_categorical = [
            column
            for column in source_categorical
            if not any(
                column == site_column or str(column) == str(site_column)
                for site_column in site_columns
            )
        ]
        resolved_bounds = self._expanded_multi_site_bounds(source_bounds, transformed)
        return _CoreTabularBayesianOptimizer.fit(
            self,
            data=transformed,
            y=y,
            input_cols=resolved_input_cols,
            categorical_cols=resolved_categorical,
            bounds=resolved_bounds,
            **kwargs,
        )

    def transform_compositions(self, data: Any) -> Any:
        if not self.multi_site_composition_enabled:
            return super().transform_compositions(data)
        if not self.composition_transformers_:
            raise RuntimeError("Call fit() before transform_compositions().")
        return self._prepare_multi_site_frame(data, fit_transformers=False)

    def inverse_compositions(
        self,
        data: Any,
        *,
        repair: bool = True,
        keep_coordinates: bool = False,
    ) -> Any:
        if not self.multi_site_composition_enabled:
            return super().inverse_compositions(
                data,
                repair=repair,
                keep_coordinates=keep_coordinates,
            )
        import pandas as pd

        if not self.composition_transformers_:
            raise RuntimeError("Call fit() before inverse_compositions().")
        candidates = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        output_frames: list[Any] = []
        all_representation_names: list[str] = []
        for site_name, config in self.composition_sites.items():
            transformer = self.composition_transformers_[site_name]
            elements = transformer._require_fitted()
            representation_names = transformer._representation_names(elements)
            all_representation_names.extend(representation_names)
            missing = [name for name in representation_names if name not in candidates]
            if missing:
                raise KeyError(
                    f"Missing model-space columns for site {site_name!r}: {missing!r}."
                )
            array = candidates.loc[:, representation_names].to_numpy(dtype=float)
            simplex_transform = transformer.simplex_transform_
            assert simplex_transform is not None
            basis_fractions = simplex_transform.inverse_transform(
                array,
                n_components=len(elements),
            )
            rows: list[dict[str, float]] = []
            search_space = self.composition_search_spaces_.get(site_name)
            for basis_row in basis_fractions:
                scaled = {
                    element: float(value) * config["total"]
                    for element, value in zip(elements, basis_row, strict=True)
                }
                if repair and search_space is not None:
                    scaled = search_space.repair(scaled)
                rows.append(scaled)
            normalized_basis = np.asarray(
                [
                    [row[element] / config["total"] for element in elements]
                    for row in rows
                ],
                dtype=float,
            )
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
            formula = pd.Series(
                [
                    format_formula(
                        dict(zip(elements, row, strict=True)),
                        order=elements,
                        precision=config["precision"],
                    )
                    for row in atomic_fractions
                ],
                index=candidates.index,
                name=config["column"],
            )
            fractions = pd.DataFrame(
                normalized_basis,
                columns=elements,
                index=candidates.index,
            ).add_prefix(f"{transformer.prefix}__fraction__")
            output_frames.extend([formula, fractions])
        drop_columns = [] if keep_coordinates else all_representation_names
        passthrough = candidates.drop(columns=drop_columns, errors="ignore")
        output_frames.append(passthrough)
        return pd.concat(output_frames, axis=1)

    def candidate(
        self,
        acq_config: Any | None = None,
        opt_config: Any | None = None,
        *,
        return_dataframe: bool = True,
        return_result: bool = False,
        return_composition: bool = True,
        keep_composition_coordinates: bool = False,
        **kwargs: Any,
    ) -> Any:
        if not self.multi_site_composition_enabled:
            return super().candidate(
                acq_config=acq_config,
                opt_config=opt_config,
                return_dataframe=return_dataframe,
                return_result=return_result,
                return_composition=return_composition,
                keep_composition_coordinates=keep_composition_coordinates,
                **kwargs,
            )
        descriptor_sites = [
            name
            for name, config in self.composition_sites.items()
            if config["include_descriptors"]
        ]
        if descriptor_sites:
            raise ValueError(
                "Composition descriptors are supported for fit/predict, but cannot "
                "be optimized independently. Disable descriptors for candidate "
                f"generation at sites {descriptor_sites!r}."
            )
        result = _CoreTabularBayesianOptimizer.candidate(
            self,
            acq_config=acq_config,
            opt_config=opt_config,
            return_dataframe=return_dataframe,
            return_result=return_result,
            **kwargs,
        )
        if return_result or not return_dataframe or not return_composition:
            return result
        candidates, acq_value = result
        restored = self.inverse_compositions(
            candidates,
            repair=True,
            keep_coordinates=keep_composition_coordinates,
        )
        return restored, acq_value

    def predict(self, data: Any, **kwargs: Any) -> Any:
        if not self.multi_site_composition_enabled:
            return super().predict(data, **kwargs)
        transformed = self._prepare_multi_site_frame(
            data,
            fit_transformers=False,
        )
        return _CoreTabularBayesianOptimizer.predict(self, transformed, **kwargs)

    def feature_importance(
        self,
        data: Any | None = None,
        y: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self.multi_site_composition_enabled:
            return super().feature_importance(data=data, y=y, **kwargs)
        if data is not None:
            data = self._prepare_multi_site_frame(
                data,
                fit_transformers=False,
            )
        return _CoreTabularBayesianOptimizer.feature_importance(
            self,
            data=data,
            y=y,
            **kwargs,
        )
