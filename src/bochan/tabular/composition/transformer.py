"""Pandas-oriented composition transformer for bochan tabular models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .descriptors import CompositionDescriptorCalculator
from .formula import ATOMIC_NUMBERS, format_formula, normalize_composition, parse_formula
from .search_space import CompositionSearchSpace
from .simplex import SimplexTransform, close_compositions


@dataclass(frozen=True)
class CompositionColumnConfig:
    """Configuration for one chemical-formula column."""

    column: str
    elements: Sequence[str] | None = None
    normalization: str = "atomic_fraction"
    representation: str = "fractions"
    reference_element: str | None = None
    pseudocount: float = 1e-12
    include_descriptors: bool = False
    descriptor_properties: Sequence[str] = ("atomic_number", "atomic_weight")
    descriptor_statistics: Sequence[str] = ("mean", "std", "min", "max", "range")
    element_properties: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    prefix: str | None = None
    retain_formula: bool = False
    precision: int = 6


class CompositionTransformer:
    """Convert formula strings to numeric composition features and back.

    This transformer deliberately stays outside the GP model. The transformed
    DataFrame can be passed to the existing bochan tabular API unchanged.
    """

    def __init__(
        self,
        *,
        elements: Sequence[str] | None = None,
        normalization: str = "atomic_fraction",
        representation: str = "fractions",
        reference_element: str | None = None,
        pseudocount: float = 1e-12,
        include_descriptors: bool = False,
        descriptor_calculator: CompositionDescriptorCalculator | None = None,
        prefix: str = "composition",
        precision: int = 6,
    ) -> None:
        self.elements = None if elements is None else tuple(elements)
        self.normalization = normalization
        self.representation = representation
        self.reference_element = reference_element
        self.pseudocount = float(pseudocount)
        self.include_descriptors = bool(include_descriptors)
        self.descriptor_calculator = descriptor_calculator or CompositionDescriptorCalculator()
        self.prefix = str(prefix)
        self.precision = int(precision)
        self.elements_: tuple[str, ...] | None = None
        self.feature_names_: tuple[str, ...] | None = None
        self.simplex_transform_: SimplexTransform | None = None

    @classmethod
    def from_config(cls, config: CompositionColumnConfig) -> "CompositionTransformer":
        """Construct a transformer from a column config."""

        calculator = CompositionDescriptorCalculator(
            properties=config.descriptor_properties,
            statistics=config.descriptor_statistics,
            element_properties=config.element_properties,
        )
        return cls(
            elements=config.elements,
            normalization=config.normalization,
            representation=config.representation,
            reference_element=config.reference_element,
            pseudocount=config.pseudocount,
            include_descriptors=config.include_descriptors,
            descriptor_calculator=calculator,
            prefix=config.prefix or config.column,
            precision=config.precision,
        )

    @staticmethod
    def _values(formulas: Any) -> tuple[list[str], Any]:
        try:
            import pandas as pd
        except ImportError:
            pd = None
        if pd is not None and isinstance(formulas, pd.Series):
            return formulas.astype(str).tolist(), formulas.index
        if isinstance(formulas, str):
            return [formulas], None
        return [str(value) for value in formulas], getattr(formulas, "index", None)

    def _require_fitted(self) -> tuple[str, ...]:
        if self.elements_ is None or self.feature_names_ is None or self.simplex_transform_ is None:
            raise RuntimeError("CompositionTransformer must be fitted before transform/inverse_transform.")
        return self.elements_

    def _reference_index(self, elements: Sequence[str]) -> int | None:
        if self.representation.lower() != "alr":
            return None
        if self.reference_element is None:
            return len(elements) - 1
        if self.reference_element not in elements:
            raise KeyError(f"reference_element {self.reference_element!r} is not included in elements.")
        return list(elements).index(self.reference_element)

    def _representation_names(self, elements: Sequence[str]) -> list[str]:
        representation = self.representation.lower()
        if representation in {"none", "fractions"}:
            return [f"{self.prefix}__fraction__{element}" for element in elements]
        if representation == "clr":
            return [f"{self.prefix}__clr__{element}" for element in elements]
        if representation == "alr":
            reference_index = self._reference_index(elements)
            assert reference_index is not None
            reference = elements[reference_index]
            return [
                f"{self.prefix}__alr__{element}_over_{reference}"
                for index, element in enumerate(elements)
                if index != reference_index
            ]
        if representation == "ilr":
            return [f"{self.prefix}__ilr__{index + 1}" for index in range(len(elements) - 1)]
        raise ValueError("representation must be one of 'fractions', 'clr', 'alr', or 'ilr'.")

    def fit(self, formulas: Any) -> "CompositionTransformer":
        """Learn the deterministic element vocabulary from formula values."""

        values, _ = self._values(formulas)
        parsed = [parse_formula(value) for value in values]
        if self.elements is None:
            discovered = {element for composition in parsed for element in composition}
            elements = sorted(discovered, key=lambda element: ATOMIC_NUMBERS[element])
        else:
            elements = list(dict.fromkeys(self.elements))
        if len(elements) < 2:
            raise ValueError("At least two elements are required for composition modelling.")
        unknown = sorted({element for composition in parsed for element in composition} - set(elements))
        if unknown:
            raise ValueError(f"Formulas contain elements outside the configured vocabulary: {unknown!r}.")

        self.elements_ = tuple(elements)
        self.simplex_transform_ = SimplexTransform(
            method=self.representation,
            pseudocount=self.pseudocount,
            reference_index=self._reference_index(elements),
        )
        feature_names = self._representation_names(elements)
        if self.include_descriptors:
            feature_names.extend(f"{self.prefix}__{name}" for name in self.descriptor_calculator.feature_names())
        self.feature_names_ = tuple(feature_names)
        return self

    def _raw_matrix(self, formulas: Any) -> tuple[np.ndarray, np.ndarray, Any]:
        elements = self._require_fitted()
        values, index = self._values(formulas)
        raw_rows: list[list[float]] = []
        normalized_rows: list[list[float]] = []
        for formula in values:
            parsed = parse_formula(formula)
            unknown = sorted(set(parsed) - set(elements))
            if unknown:
                raise ValueError(f"Formula {formula!r} contains unknown elements: {unknown!r}.")
            raw_rows.append([float(parsed.get(element, 0.0)) for element in elements])
            normalized = normalize_composition(parsed, mode=self.normalization)
            normalized_rows.append([float(normalized.get(element, 0.0)) for element in elements])
        return np.asarray(raw_rows), np.asarray(normalized_rows), index

    def transform(self, formulas: Any) -> Any:
        """Transform formula values into a numeric pandas DataFrame."""

        import pandas as pd

        elements = self._require_fitted()
        raw, normalized, index = self._raw_matrix(formulas)
        assert self.simplex_transform_ is not None
        is_normalized = self.normalization.lower() not in {"none", "raw", "stoichiometric"}
        representation_input = normalized if is_normalized else raw
        transformed = self.simplex_transform_.transform(representation_input)
        arrays = [transformed]
        if self.include_descriptors:
            atomic_fractions = close_compositions(raw)
            arrays.append(self.descriptor_calculator.transform(atomic_fractions, elements))
        output = np.concatenate(arrays, axis=1) if len(arrays) > 1 else arrays[0]
        return pd.DataFrame(output, columns=list(self.feature_names_ or ()), index=index)

    def fit_transform(self, formulas: Any) -> Any:
        """Fit and transform formula values."""

        return self.fit(formulas).transform(formulas)

    def inverse_transform(self, values: Any) -> Any:
        """Convert transformed composition coordinates to canonical formulas."""

        import pandas as pd

        elements = self._require_fitted()
        representation_names = self._representation_names(elements)
        if isinstance(values, pd.DataFrame):
            missing = [name for name in representation_names if name not in values.columns]
            if missing:
                raise KeyError(f"Missing composition feature columns: {missing!r}.")
            array = values.loc[:, representation_names].to_numpy(dtype=float)
            index = values.index
        else:
            array = np.asarray(values, dtype=float)
            index = None
        if array.ndim == 1:
            array = array.reshape(1, -1)
        assert self.simplex_transform_ is not None
        fractions = self.simplex_transform_.inverse_transform(array, n_components=len(elements))
        formulas = [
            format_formula(dict(zip(elements, row, strict=True)), order=elements, precision=self.precision)
            for row in fractions
        ]
        return pd.Series(formulas, index=index, name=f"{self.prefix}__formula")

    def transform_frame(self, frame: Any, formula_column: str, *, drop_formula: bool = True) -> Any:
        """Replace one formula column with numeric composition features."""

        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("transform_frame expects a pandas.DataFrame.")
        if formula_column not in frame.columns:
            raise KeyError(f"Unknown formula column {formula_column!r}.")
        transformed = self.transform(frame.loc[:, formula_column])
        base = frame.drop(columns=[formula_column]) if drop_formula else frame.copy()
        return pd.concat([base, transformed], axis=1)


class CompositionTabularPreprocessor:
    """Bridge a formula column to bochan's existing tabular DataFrame API."""

    def __init__(
        self,
        config: CompositionColumnConfig,
        *,
        search_space: CompositionSearchSpace | None = None,
    ) -> None:
        self.config = config
        self.transformer = CompositionTransformer.from_config(config)
        self.search_space = search_space

    def fit(self, frame: Any) -> "CompositionTabularPreprocessor":
        """Fit the formula vocabulary from a DataFrame."""

        if self.config.column not in frame.columns:
            raise KeyError(f"Unknown formula column {self.config.column!r}.")
        self.transformer.fit(frame.loc[:, self.config.column])
        return self

    def transform(self, frame: Any) -> Any:
        """Return a model-ready DataFrame for the existing tabular API."""

        return self.transformer.transform_frame(
            frame,
            self.config.column,
            drop_formula=not self.config.retain_formula,
        )

    def fit_transform(self, frame: Any) -> Any:
        """Fit and transform a tabular DataFrame."""

        return self.fit(frame).transform(frame)

    def inverse_candidates(self, candidates: Any, *, repair: bool = True) -> Any:
        """Recover formula strings from model-space candidate coordinates.

        When a search space is configured, inverse-transformed fractions are
        repaired before formatting and returned alongside the numeric fractions.
        """

        import pandas as pd

        elements = self.transformer._require_fitted()
        representation_names = self.transformer._representation_names(elements)
        if not isinstance(candidates, pd.DataFrame):
            candidates = pd.DataFrame(candidates, columns=representation_names)
        array = candidates.loc[:, representation_names].to_numpy(dtype=float)
        assert self.transformer.simplex_transform_ is not None
        fractions = self.transformer.simplex_transform_.inverse_transform(array, n_components=len(elements))

        fraction_rows: list[dict[str, float]] = []
        for row in fractions:
            composition = dict(zip(elements, row, strict=True))
            if repair and self.search_space is not None:
                composition = self.search_space.repair(composition)
            fraction_rows.append(composition)
        fraction_frame = pd.DataFrame(fraction_rows, index=candidates.index)
        formula_series = pd.Series(
            [format_formula(row, order=elements, precision=self.config.precision) for row in fraction_rows],
            index=candidates.index,
            name=self.config.column,
        )
        passthrough = candidates.drop(columns=representation_names, errors="ignore")
        fraction_frame = fraction_frame.add_prefix(f"{self.transformer.prefix}__fraction__")
        return pd.concat([formula_series, fraction_frame, passthrough], axis=1)
