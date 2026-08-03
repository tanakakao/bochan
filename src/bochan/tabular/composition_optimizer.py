"""Direct composition arguments for :class:`TabularBayesianOptimizer`."""
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
from .optimizer_api import TabularBayesianOptimizer as _TabularBayesianOptimizer


class TabularBayesianOptimizer(_TabularBayesianOptimizer):
    """Tabular optimizer with automatic chemical-formula preprocessing.

    Composition handling is enabled with direct keyword arguments such as
    ``composition_col="formula"`` and ``composition_representation="ilr"``.
    No composition-specific config object is required.
    """

    def __init__(
        self,
        model_config: Any | None = None,
        fit_config: Any | None = None,
        *,
        composition_col: Any | None = None,
        formula_col: Any | None = None,
        composition_elements: Sequence[str] | None = None,
        composition_normalization: str = "atomic_fraction",
        composition_representation: str = "ilr",
        composition_reference_element: str | None = None,
        composition_pseudocount: float = 1e-12,
        composition_include_descriptors: bool = False,
        composition_descriptor_properties: Sequence[str] = (
            "atomic_number",
            "atomic_weight",
        ),
        composition_descriptor_statistics: Sequence[str] = (
            "mean",
            "std",
            "min",
            "max",
            "range",
        ),
        composition_element_properties: Mapping[str, Mapping[str, float]] | None = None,
        composition_prefix: str | None = None,
        composition_precision: int = 6,
        composition_total: float = 1.0,
        composition_bounds: Mapping[str, Sequence[float]] | None = None,
        composition_steps: Mapping[str, float] | None = None,
        composition_min_components: int = 1,
        composition_max_components: int | None = None,
        composition_required_components: Sequence[str] | None = None,
        composition_coordinate_bounds: Sequence[float] | Mapping[Any, Sequence[float]] = (
            -8.0,
            8.0,
        ),
        **kwargs: Any,
    ) -> None:
        if (
            composition_col is not None
            and formula_col is not None
            and composition_col != formula_col
        ):
            raise ValueError(
                "Specify either composition_col or formula_col, not conflicting values."
            )
        self.composition_col = (
            composition_col if composition_col is not None else formula_col
        )
        self.composition_elements = (
            None if composition_elements is None else tuple(composition_elements)
        )
        self.composition_normalization = str(composition_normalization)
        self.composition_representation = str(composition_representation)
        self.composition_reference_element = composition_reference_element
        self.composition_pseudocount = float(composition_pseudocount)
        self.composition_include_descriptors = bool(composition_include_descriptors)
        self.composition_descriptor_properties = tuple(
            composition_descriptor_properties
        )
        self.composition_descriptor_statistics = tuple(
            composition_descriptor_statistics
        )
        self.composition_element_properties = dict(
            composition_element_properties or {}
        )
        self.composition_prefix = (
            None if composition_prefix is None else str(composition_prefix)
        )
        self.composition_precision = int(composition_precision)
        self.composition_total = float(composition_total)
        self.composition_bounds = dict(composition_bounds or {})
        self.composition_steps = dict(composition_steps or {})
        self.composition_min_components = int(composition_min_components)
        self.composition_max_components = composition_max_components
        self.composition_required_components = tuple(
            composition_required_components or ()
        )
        self.composition_coordinate_bounds = composition_coordinate_bounds
        self.composition_transformer_: CompositionTransformer | None = None
        self.composition_search_space_: CompositionSearchSpace | None = None
        super().__init__(model_config=model_config, fit_config=fit_config, **kwargs)

    @property
    def composition_enabled(self) -> bool:
        return getattr(self, "composition_col", None) is not None

    def _make_composition_transformer(self, formulas: Any) -> CompositionTransformer:
        calculator = CompositionDescriptorCalculator(
            properties=self.composition_descriptor_properties,
            statistics=self.composition_descriptor_statistics,
            element_properties=self.composition_element_properties,
        )
        transformer = CompositionTransformer(
            elements=self.composition_elements,
            normalization=self.composition_normalization,
            representation=self.composition_representation,
            reference_element=self.composition_reference_element,
            pseudocount=self.composition_pseudocount,
            include_descriptors=self.composition_include_descriptors,
            descriptor_calculator=calculator,
            prefix=self.composition_prefix or str(self.composition_col),
            precision=self.composition_precision,
        )
        transformer.fit(formulas)
        return transformer

    def _make_composition_search_space(
        self,
        elements: Sequence[str],
    ) -> CompositionSearchSpace:
        return CompositionSearchSpace(
            components=elements,
            total=self.composition_total,
            bounds=self.composition_bounds,
            steps=self.composition_steps,
            min_active_components=self.composition_min_components,
            max_active_components=self.composition_max_components,
            required_components=self.composition_required_components,
        )

    def _replace_composition_input_col(
        self,
        input_cols: Sequence[Any] | None,
    ) -> list[Any] | None:
        if input_cols is None or self.composition_transformer_ is None:
            return None if input_cols is None else list(input_cols)
        replacement = list(self.composition_transformer_.feature_names_ or ())
        resolved: list[Any] = []
        for column in input_cols:
            if column == self.composition_col or str(column) == str(
                self.composition_col
            ):
                resolved.extend(replacement)
            else:
                resolved.append(column)
        return resolved

    def _coordinate_bound(self, name: str, index: int) -> tuple[float, float]:
        configured = self.composition_coordinate_bounds
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
                "composition_coordinate_bounds must contain increasing "
                "[lower, upper] pairs."
            )
        return float(pair[0]), float(pair[1])

    def _expanded_bounds(self, bounds: Any, transformed: Any) -> Any:
        if self.composition_transformer_ is None:
            return bounds
        if bounds is not None and not isinstance(bounds, Mapping):
            raise TypeError(
                "Composition-enabled tabular optimization requires bounds to be "
                "a column mapping."
            )
        expanded = dict(bounds or {})
        expanded.pop(self.composition_col, None)
        expanded.pop(str(self.composition_col), None)
        elements = self.composition_transformer_._require_fitted()
        representation_names = self.composition_transformer_._representation_names(
            elements
        )
        representation = self.composition_representation.lower()
        for index, name in enumerate(representation_names):
            if name in expanded:
                continue
            if representation in {"none", "fractions"}:
                component = elements[index]
                raw_pair = self.composition_bounds.get(
                    component,
                    (0.0, self.composition_total),
                )
                expanded[name] = [
                    float(raw_pair[0]) / self.composition_total,
                    float(raw_pair[1]) / self.composition_total,
                ]
            else:
                expanded[name] = list(self._coordinate_bound(name, index))
        descriptor_names = [
            value
            for value in (self.composition_transformer_.feature_names_ or ())
            if value not in representation_names
        ]
        for name in descriptor_names:
            if name in expanded:
                continue
            lower = float(transformed[name].min())
            upper = float(transformed[name].max())
            margin = max((upper - lower) * 0.05, 1e-9)
            expanded[name] = [lower - margin, upper + margin]
        return expanded

    def _prepare_composition_frame(
        self,
        data: Any,
        *,
        fit_transformer: bool,
    ) -> Any:
        if not self.composition_enabled:
            return data
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError("composition_col requires pandas DataFrame input.")
        if self.composition_col not in data.columns:
            model_names = (
                set(self.composition_transformer_.feature_names_ or ())
                if self.composition_transformer_
                else set()
            )
            if model_names and model_names.issubset(data.columns):
                return data
            raise KeyError(f"Unknown composition column {self.composition_col!r}.")
        if fit_transformer or self.composition_transformer_ is None:
            self.composition_transformer_ = self._make_composition_transformer(
                data.loc[:, self.composition_col]
            )
            elements = self.composition_transformer_._require_fitted()
            self.composition_search_space_ = self._make_composition_search_space(
                elements
            )
        return self.composition_transformer_.transform_frame(
            data,
            self.composition_col,
            drop_formula=True,
        )

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
        if data is None:
            data = self.data
        if data is None:
            raise ValueError(
                "No data was supplied. Pass data to fit(...) or use from_csv(...)."
            )
        if not self.composition_enabled:
            return super().fit(
                data=data,
                y=y,
                input_cols=input_cols,
                categorical_cols=categorical_cols,
                bounds=bounds,
                **kwargs,
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
        transformed = self._prepare_composition_frame(
            data,
            fit_transformer=True,
        )
        resolved_input_cols = self._replace_composition_input_col(source_input_cols)
        resolved_categorical = [
            column
            for column in source_categorical
            if column != self.composition_col
            and str(column) != str(self.composition_col)
        ]
        resolved_bounds = self._expanded_bounds(source_bounds, transformed)
        return super().fit(
            data=transformed,
            y=y,
            input_cols=resolved_input_cols,
            categorical_cols=resolved_categorical,
            bounds=resolved_bounds,
            **kwargs,
        )

    def transform_compositions(self, data: Any) -> Any:
        """Convert a raw formula DataFrame into the fitted model-space frame."""
        if self.composition_transformer_ is None:
            raise RuntimeError("Call fit() before transform_compositions().")
        return self._prepare_composition_frame(data, fit_transformer=False)

    def inverse_compositions(
        self,
        data: Any,
        *,
        repair: bool = True,
        keep_coordinates: bool = False,
    ) -> Any:
        """Convert model coordinates into formula and fraction columns."""
        import pandas as pd

        if self.composition_transformer_ is None:
            raise RuntimeError("Call fit() before inverse_compositions().")
        elements = self.composition_transformer_._require_fitted()
        representation_names = self.composition_transformer_._representation_names(
            elements
        )
        candidates = (
            data
            if isinstance(data, pd.DataFrame)
            else pd.DataFrame(data, columns=representation_names)
        )
        array = candidates.loc[:, representation_names].to_numpy(dtype=float)
        simplex_transform = self.composition_transformer_.simplex_transform_
        assert simplex_transform is not None
        basis_fractions = simplex_transform.inverse_transform(
            array,
            n_components=len(elements),
        )
        rows: list[dict[str, float]] = []
        for basis_row in basis_fractions:
            scaled = {
                element: float(value) * self.composition_total
                for element, value in zip(elements, basis_row, strict=True)
            }
            if repair and self.composition_search_space_ is not None:
                scaled = self.composition_search_space_.repair(scaled)
            rows.append(scaled)
        normalized_basis = np.asarray(
            [
                [row[element] / self.composition_total for element in elements]
                for row in rows
            ],
            dtype=float,
        )
        if self.composition_normalization.lower() in {
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
                    precision=self.composition_precision,
                )
                for row in atomic_fractions
            ],
            index=candidates.index,
            name=self.composition_col,
        )
        normalized = pd.DataFrame(
            normalized_basis,
            columns=elements,
            index=candidates.index,
        ).add_prefix(f"{self.composition_transformer_.prefix}__fraction__")
        drop_columns = [] if keep_coordinates else representation_names
        passthrough = candidates.drop(columns=drop_columns, errors="ignore")
        return pd.concat([formula, normalized, passthrough], axis=1)

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
        if self.composition_enabled and self.composition_include_descriptors:
            raise ValueError(
                "composition_include_descriptors=True is supported for fit/predict, "
                "but descriptors cannot be optimized independently. Disable "
                "descriptors for candidate generation."
            )
        result = super().candidate(
            acq_config=acq_config,
            opt_config=opt_config,
            return_dataframe=return_dataframe,
            return_result=return_result,
            **kwargs,
        )
        if (
            return_result
            or not return_dataframe
            or not return_composition
            or not self.composition_enabled
        ):
            return result
        candidates, acq_value = result
        restored = self.inverse_compositions(
            candidates,
            repair=True,
            keep_coordinates=keep_composition_coordinates,
        )
        return restored, acq_value

    def predict(self, data: Any, **kwargs: Any) -> Any:
        if self.composition_enabled:
            data = self._prepare_composition_frame(
                data,
                fit_transformer=False,
            )
        return super().predict(data, **kwargs)

    def feature_importance(
        self,
        data: Any | None = None,
        y: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if data is not None and self.composition_enabled:
            data = self._prepare_composition_frame(
                data,
                fit_transformer=False,
            )
        return super().feature_importance(data=data, y=y, **kwargs)
