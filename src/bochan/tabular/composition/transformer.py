"""Pandas integration for the core composition domain transformer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bochan.composition import (
    CompositionDescriptorCalculator,
    CompositionSearchSpace,
    CompositionTransformer,
    format_formula,
)


@dataclass(frozen=True)
class CompositionColumnConfig:
    """Configuration for one chemical-formula DataFrame column."""

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


def make_composition_transformer(
    config: CompositionColumnConfig,
) -> CompositionTransformer:
    """Construct a core transformer from tabular column configuration."""

    calculator = CompositionDescriptorCalculator(
        properties=config.descriptor_properties,
        statistics=config.descriptor_statistics,
        element_properties=config.element_properties,
    )
    return CompositionTransformer(
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


def transform_composition_frame(
    transformer: CompositionTransformer,
    frame: Any,
    formula_column: str,
    *,
    drop_formula: bool = True,
) -> Any:
    """Apply a fitted core transformer to one DataFrame formula column."""

    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("transform_composition_frame expects a pandas.DataFrame.")
    if formula_column not in frame.columns:
        raise KeyError(f"Unknown formula column {formula_column!r}.")
    values = transformer.transform(frame.loc[:, formula_column])
    transformed = pd.DataFrame(
        values,
        columns=list(transformer.feature_names_ or ()),
        index=frame.index,
    )
    base = frame.drop(columns=[formula_column]) if drop_formula else frame.copy()
    return pd.concat([base, transformed], axis=1)


class CompositionTabularPreprocessor:
    """Bridge one formula column to bochan's tabular DataFrame API."""

    def __init__(
        self,
        config: CompositionColumnConfig,
        *,
        search_space: CompositionSearchSpace | None = None,
    ) -> None:
        self.config = config
        self.transformer = make_composition_transformer(config)
        self.search_space = search_space

    def fit(self, frame: Any) -> "CompositionTabularPreprocessor":
        """Fit the formula vocabulary from a DataFrame."""

        if self.config.column not in frame.columns:
            raise KeyError(f"Unknown formula column {self.config.column!r}.")
        self.transformer.fit(frame.loc[:, self.config.column])
        return self

    def transform(self, frame: Any) -> Any:
        """Return a model-ready DataFrame for the existing tabular API."""

        return transform_composition_frame(
            self.transformer,
            frame,
            self.config.column,
            drop_formula=not self.config.retain_formula,
        )

    def fit_transform(self, frame: Any) -> Any:
        """Fit and transform a tabular DataFrame."""

        return self.fit(frame).transform(frame)

    def inverse_candidates(
        self,
        candidates: Any,
        *,
        repair: bool = True,
    ) -> Any:
        """Recover formula strings and native fractions from model coordinates."""

        import pandas as pd

        elements = self.transformer.fitted_elements
        representation_names = self.transformer.representation_feature_names_
        if not isinstance(candidates, pd.DataFrame):
            candidates = pd.DataFrame(candidates, columns=representation_names)
        array = candidates.loc[:, representation_names].to_numpy(dtype=float)
        fractions = self.transformer.inverse_transform_fractions(array)

        fraction_rows: list[dict[str, float]] = []
        for row in fractions:
            composition = dict(zip(elements, row, strict=True))
            if repair and self.search_space is not None:
                composition = self.search_space.repair(composition)
            fraction_rows.append(composition)

        fraction_array = np.asarray(
            [
                [row[element] for element in elements]
                for row in fraction_rows
            ],
            dtype=float,
        )
        atomic_fractions = self.transformer.basis_to_atomic_fractions(fraction_array)
        formula_series = pd.Series(
            [
                format_formula(
                    dict(zip(elements, row, strict=True)),
                    order=elements,
                    precision=self.config.precision,
                )
                for row in atomic_fractions
            ],
            index=candidates.index,
            name=self.config.column,
        )
        fraction_frame = pd.DataFrame(
            fraction_rows,
            index=candidates.index,
        ).add_prefix(f"{self.transformer.prefix}__fraction__")
        passthrough = candidates.drop(
            columns=representation_names,
            errors="ignore",
        )
        return pd.concat(
            [formula_series, fraction_frame, passthrough],
            axis=1,
        )


__all__ = [
    "CompositionColumnConfig",
    "CompositionTabularPreprocessor",
]