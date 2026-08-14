"""Composition-domain transforms from formula strings to numeric coordinates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .descriptors import CompositionDescriptorCalculator
from .formula import (
    ATOMIC_NUMBERS,
    ATOMIC_WEIGHTS,
    format_formula,
    normalize_composition,
    parse_formula,
)
from .simplex import SimplexTransform, close_compositions


class CompositionTransformer:
    """Convert chemical formulas to numeric composition coordinates and back.

    The domain transformer is deliberately independent of pandas and bochan's
    tabular optimizer. Input may be any iterable of formula-like values; numeric
    outputs are NumPy arrays and inverse outputs are plain Python strings.
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
        self.descriptor_calculator = (
            descriptor_calculator or CompositionDescriptorCalculator()
        )
        self.prefix = str(prefix)
        self.precision = int(precision)
        self.elements_: tuple[str, ...] | None = None
        self.feature_names_: tuple[str, ...] | None = None
        self.simplex_transform_: SimplexTransform | None = None

    @staticmethod
    def _values(formulas: Any) -> list[str]:
        if isinstance(formulas, str):
            return [formulas]
        try:
            return [str(value) for value in formulas]
        except TypeError as exc:
            raise TypeError("formulas must be a string or iterable of formula values.") from exc

    def _require_fitted(self) -> tuple[str, ...]:
        if (
            self.elements_ is None
            or self.feature_names_ is None
            or self.simplex_transform_ is None
        ):
            raise RuntimeError(
                "CompositionTransformer must be fitted before transform/inverse_transform."
            )
        return self.elements_

    @property
    def fitted_elements(self) -> tuple[str, ...]:
        """Return the fitted element vocabulary."""

        return self._require_fitted()

    def _reference_index(self, elements: Sequence[str]) -> int | None:
        if self.representation.lower() != "alr":
            return None
        if self.reference_element is None:
            return len(elements) - 1
        if self.reference_element not in elements:
            raise KeyError(
                f"reference_element {self.reference_element!r} is not included in elements."
            )
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
            return [
                f"{self.prefix}__ilr__{index + 1}"
                for index in range(len(elements) - 1)
            ]
        raise ValueError(
            "representation must be one of 'fractions', 'clr', 'alr', or 'ilr'."
        )

    @property
    def representation_feature_names_(self) -> tuple[str, ...]:
        """Return fitted model-coordinate feature names."""

        elements = self._require_fitted()
        return tuple(self._representation_names(elements))

    def fit(self, formulas: Any) -> "CompositionTransformer":
        """Learn a deterministic element vocabulary from formula values."""

        values = self._values(formulas)
        if not values:
            raise ValueError("formulas must contain at least one value.")
        parsed = [parse_formula(value) for value in values]
        if self.elements is None:
            discovered = {
                element for composition in parsed for element in composition
            }
            elements = sorted(
                discovered,
                key=lambda element: ATOMIC_NUMBERS[element],
            )
        else:
            elements = list(dict.fromkeys(self.elements))
        if len(elements) < 2:
            raise ValueError(
                "At least two elements are required for composition modelling."
            )
        unknown = sorted(
            {element for composition in parsed for element in composition}
            - set(elements)
        )
        if unknown:
            raise ValueError(
                "Formulas contain elements outside the configured vocabulary: "
                f"{unknown!r}."
            )

        self.elements_ = tuple(elements)
        self.simplex_transform_ = SimplexTransform(
            method=self.representation,
            pseudocount=self.pseudocount,
            reference_index=self._reference_index(elements),
        )
        feature_names = self._representation_names(elements)
        if self.include_descriptors:
            feature_names.extend(
                f"{self.prefix}__{name}"
                for name in self.descriptor_calculator.feature_names()
            )
        self.feature_names_ = tuple(feature_names)
        return self

    def _raw_matrix(self, formulas: Any) -> tuple[np.ndarray, np.ndarray]:
        elements = self._require_fitted()
        values = self._values(formulas)
        raw_rows: list[list[float]] = []
        normalized_rows: list[list[float]] = []
        for formula in values:
            parsed = parse_formula(formula)
            unknown = sorted(set(parsed) - set(elements))
            if unknown:
                raise ValueError(
                    f"Formula {formula!r} contains unknown elements: {unknown!r}."
                )
            raw_rows.append(
                [float(parsed.get(element, 0.0)) for element in elements]
            )
            normalized = normalize_composition(
                parsed,
                mode=self.normalization,
            )
            normalized_rows.append(
                [float(normalized.get(element, 0.0)) for element in elements]
            )
        return np.asarray(raw_rows), np.asarray(normalized_rows)

    def basis_to_atomic_fractions(self, fractions: Any) -> np.ndarray:
        """Convert values in the configured composition basis to atomic fractions."""

        elements = self._require_fitted()
        array = np.asarray(fractions, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if (
            array.ndim != 2
            or array.shape[1] != len(elements)
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                "Composition fractions must be a finite 2D array matching the fitted elements."
            )
        mode = self.normalization.lower()
        if mode in {"weight_fraction", "weight", "mass_fraction"}:
            weights = np.asarray(
                [ATOMIC_WEIGHTS[element] for element in elements],
                dtype=float,
            )
            return close_compositions(array / weights)
        return close_compositions(array)

    def transform(self, formulas: Any) -> np.ndarray:
        """Transform formulas into a numeric NumPy feature matrix."""

        elements = self._require_fitted()
        raw, normalized = self._raw_matrix(formulas)
        assert self.simplex_transform_ is not None
        is_normalized = self.normalization.lower() not in {
            "none",
            "raw",
            "stoichiometric",
        }
        representation_input = normalized if is_normalized else raw
        transformed = self.simplex_transform_.transform(representation_input)
        arrays = [transformed]
        if self.include_descriptors:
            atomic_fractions = close_compositions(raw)
            arrays.append(
                self.descriptor_calculator.transform(
                    atomic_fractions,
                    elements,
                )
            )
        return (
            np.concatenate(arrays, axis=1)
            if len(arrays) > 1
            else arrays[0]
        )

    def fit_transform(self, formulas: Any) -> np.ndarray:
        """Fit the transformer and return numeric composition features."""

        return self.fit(formulas).transform(formulas)

    def inverse_transform_fractions(self, values: Any) -> np.ndarray:
        """Convert model coordinates back to fractions in the configured basis."""

        elements = self._require_fitted()
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("Composition coordinates must be a finite 2D array.")

        representation_width = len(self.representation_feature_names_)
        full_width = len(self.feature_names_ or ())
        if array.shape[1] == full_width and full_width > representation_width:
            array = array[:, :representation_width]
        elif array.shape[1] != representation_width:
            raise ValueError(
                "Composition coordinate width does not match the fitted representation."
            )

        assert self.simplex_transform_ is not None
        return self.simplex_transform_.inverse_transform(
            array,
            n_components=len(elements),
        )

    def inverse_transform(self, values: Any) -> list[str]:
        """Convert model coordinates back to canonical formula strings."""

        elements = self._require_fitted()
        fractions = self.inverse_transform_fractions(values)
        atomic_fractions = self.basis_to_atomic_fractions(fractions)
        return [
            format_formula(
                dict(zip(elements, row, strict=True)),
                order=elements,
                precision=self.precision,
            )
            for row in atomic_fractions
        ]


__all__ = ["CompositionTransformer"]