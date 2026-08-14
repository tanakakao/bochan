"""Stateful composition adapter used by the tabular optimizer facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from bochan.composition import (
    CompositionDescriptorCalculator,
    CompositionSearchSpace,
    CompositionTransformer,
    format_formula,
)

from .bounds import CompositionBoundsResolver
from .columns import CompositionElementColumnTransform
from .config import normalize_composition_sites
from .transformer import transform_composition_frame
from .variable_total import CompositionVariableTotalTransform


class CompositionAdapter:
    """Own composition transform state without depending on an optimizer class."""

    def __init__(
        self,
        sites: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.element_columns = CompositionElementColumnTransform()
        self.variable_total = CompositionVariableTotalTransform()
        self.bounds_resolver = CompositionBoundsResolver()
        self.sites = self.variable_total.normalize_sites(
            sites,
            base_normalizer=lambda prepared: self.element_columns.normalize_sites(
                prepared,
                base_normalizer=normalize_composition_sites,
            ),
        )
        self.transformers: dict[str, CompositionTransformer] = {}
        self.search_spaces: dict[str, CompositionSearchSpace] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.sites)

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
    def _make_base_search_space(
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

    def _make_site_search_space(
        self,
        config: Mapping[str, Any],
    ) -> CompositionSearchSpace | None:
        return self.variable_total.make_site_search_space(
            config,
            base_factory=self._make_base_search_space,
        )

    def _prepare_base_frame(
        self,
        data: Any,
        *,
        fit_transformers: bool,
    ) -> Any:
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError("composition_sites requires pandas DataFrame input.")

        transformed = data.copy()
        for site_name, config in self.sites.items():
            column = config["column"]
            transformer = self.transformers.get(site_name)
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
                self.transformers[site_name] = transformer
                search_space = self._make_site_search_space(config)
                if search_space is None:
                    self.search_spaces.pop(site_name, None)
                else:
                    self.search_spaces[site_name] = search_space
            transformed = transform_composition_frame(
                transformer,
                transformed,
                column,
                drop_formula=True,
            )
        return transformed

    def _prepare_element_frame(
        self,
        data: Any,
        *,
        fit_transformers: bool,
    ) -> Any:
        return self.element_columns.prepare_multi_site_frame(
            data,
            fit_transformers=fit_transformers,
            composition_sites=self.sites,
            composition_transformers=self.transformers,
            base_prepare=self._prepare_base_frame,
        )

    def prepare_frame(
        self,
        data: Any,
        *,
        fit_transformers: bool,
    ) -> Any:
        """Transform all configured composition inputs into model coordinates."""

        if not self.enabled:
            return data
        return self.variable_total.prepare_multi_site_frame(
            data,
            fit_transformers=fit_transformers,
            composition_sites=self.sites,
            base_prepare=self._prepare_element_frame,
            site_source_columns=self.element_columns.source_columns,
            numeric_site_values=self.element_columns.numeric_site_values,
        )

    def replace_input_cols(
        self,
        input_cols: Sequence[Any] | None,
    ) -> list[Any] | None:
        return self.variable_total.replace_input_cols(
            input_cols,
            composition_sites=self.sites,
            composition_transformers=self.transformers,
            site_source_columns=self.element_columns.source_columns,
        )

    @staticmethod
    def _coordinate_bound(
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

    def _expanded_base_bounds(self, bounds: Any, transformed: Any) -> Any:
        if bounds is not None and not isinstance(bounds, Mapping):
            raise TypeError(
                "Composition optimization requires bounds to be a column mapping."
            )
        expanded = dict(bounds or {})
        for site_name, config in self.sites.items():
            column = config["column"]
            expanded.pop(column, None)
            expanded.pop(str(column), None)
            transformer = self.transformers[site_name]
            elements = transformer.fitted_elements
            representation_names = transformer.representation_feature_names_
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
                        self._coordinate_bound(
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

    def expanded_bounds(self, bounds: Any, transformed: Any) -> Any:
        expanded = self._expanded_base_bounds(bounds, transformed)
        expanded = self.element_columns.clean_bounds(
            expanded,
            composition_sites=self.sites,
        )
        expanded = self.variable_total.complete_bounds(
            expanded,
            composition_sites=self.sites,
            composition_transformers=self.transformers,
        )
        return self.bounds_resolver.complete(expanded, transformed)

    def resolve_categorical_cols(
        self,
        categorical_cols: Sequence[Any] | None,
        *,
        default_categorical_cols: Sequence[Any],
    ) -> list[Any]:
        return self.element_columns.resolve_categorical_cols(
            categorical_cols,
            default_categorical_cols=default_categorical_cols,
            composition_sites=self.sites,
        )

    def _inverse_base(
        self,
        data: Any,
        *,
        repair: bool,
        keep_coordinates: bool,
    ) -> Any:
        import pandas as pd

        if not self.transformers:
            raise RuntimeError("Call fit() before inverse_compositions().")
        candidates = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        output_frames: list[Any] = []
        all_representation_names: list[str] = []
        for site_name, config in self.sites.items():
            transformer = self.transformers[site_name]
            elements = transformer.fitted_elements
            representation_names = transformer.representation_feature_names_
            all_representation_names.extend(representation_names)
            missing = [
                name for name in representation_names if name not in candidates
            ]
            if missing:
                raise KeyError(
                    f"Missing model-space columns for site {site_name!r}: {missing!r}."
                )

            array = candidates.loc[:, representation_names].to_numpy(dtype=float)
            basis_fractions = transformer.inverse_transform_fractions(array)
            rows: list[dict[str, float]] = []
            search_space = self.search_spaces.get(site_name)
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
            atomic_fractions = transformer.basis_to_atomic_fractions(normalized_basis)

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
        passthrough = candidates.drop(
            columns=drop_columns,
            errors="ignore",
        )
        output_frames.append(passthrough)
        return pd.concat(output_frames, axis=1)

    def inverse(
        self,
        data: Any,
        *,
        repair: bool = True,
        keep_coordinates: bool = False,
    ) -> Any:
        """Restore original formula/element inputs from model coordinates."""

        if not self.enabled:
            return data
        restored = self._inverse_base(
            data,
            repair=repair,
            keep_coordinates=keep_coordinates,
        )
        restored = self.element_columns.inverse_compositions(
            restored,
            composition_sites=self.sites,
            composition_transformers=self.transformers,
            enabled=self.enabled,
        )
        return self.variable_total.inverse_compositions(
            data,
            restored,
            composition_sites=self.sites,
            composition_transformers=self.transformers,
            multi_site_composition_enabled=self.enabled,
            repair=repair,
        )

    def descriptor_sites(self) -> list[str]:
        return [
            name
            for name, config in self.sites.items()
            if config["include_descriptors"]
        ]


__all__ = ["CompositionAdapter"]