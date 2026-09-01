"""Tabular crystal-structure catalog and structure-index conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bochan.structure import ALIGNNGraphBuilder


def _as_list(value: Any | None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _mapping_value(mapping: Mapping[Any, Any] | None, key: Any) -> Any | None:
    if mapping is None:
        return None
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key))


class StructureTabularAdapter:
    """Own a structure catalog and its canonical discrete model coordinate.

    User-facing structure IDs are encoded through the existing tabular category
    map machinery. The resulting integer coordinate indexes an ordered raw
    ``structures`` bank. Backends that need another representation, such as
    ALIGNN graphs, derive and cache it from that same canonical ordering.
    Candidate generation must enumerate the structure index rather than relax it
    into a continuous variable.
    """

    def __init__(
        self,
        *,
        column: Any | None = None,
        catalog: Mapping[Any, Any] | None = None,
        graph_builder: ALIGNNGraphBuilder | Any | None = None,
    ) -> None:
        if column is None:
            if catalog:
                raise ValueError("structure_catalog requires structure_col.")
            self.column = None
            self.catalog: dict[Any, Any] = {}
        else:
            if catalog is None or not isinstance(catalog, Mapping) or not catalog:
                raise ValueError("structure_col requires a non-empty structure_catalog mapping.")
            self.column = column
            self.catalog = dict(catalog)
        if graph_builder is not None and not callable(getattr(graph_builder, "build_many", None)):
            raise TypeError("structure_graph_builder must expose build_many(structures).")
        self.graph_builder = graph_builder
        self._ids = tuple(self.catalog)
        self._id_to_index = {value: index for index, value in enumerate(self._ids)}
        self._structures = tuple(self.catalog[value] for value in self._ids)
        self._structure_graphs: tuple[Any, ...] | None = None

    @property
    def enabled(self) -> bool:
        return self.column is not None

    @property
    def structure_ids(self) -> tuple[Any, ...]:
        return self._ids

    @property
    def structures(self) -> tuple[Any, ...]:
        """Return raw structures in the canonical structure-index order."""

        if not self.enabled:
            raise RuntimeError("No tabular structure catalog is configured.")
        return self._structures

    @property
    def num_structures(self) -> int:
        return len(self._ids)

    @property
    def category_map(self) -> dict[Any, int]:
        """Return the canonical user-ID -> structure-index mapping."""

        return dict(self._id_to_index)

    @property
    def structure_graphs(self) -> tuple[Any, ...]:
        """Return the ALIGNN graph bank derived from the canonical raw structures."""

        if not self.enabled:
            raise RuntimeError("No tabular structure catalog is configured.")
        if self._structure_graphs is None:
            builder = self.graph_builder or ALIGNNGraphBuilder()
            graphs = builder.build_many(self.structures)
            if not isinstance(graphs, Sequence) or isinstance(graphs, (str, bytes)):
                raise TypeError("structure_graph_builder.build_many() must return a sequence.")
            if len(graphs) != self.num_structures:
                raise ValueError(
                    "The structure graph bank must contain one graph per catalog entry: "
                    f"{len(graphs)} != {self.num_structures}."
                )
            self._structure_graphs = tuple(graphs)
        return self._structure_graphs

    def replace_input_cols(self, input_cols: Sequence[Any] | Any | None) -> list[Any] | None:
        """Place the structure selector first, matching structure-model contracts."""

        if input_cols is None:
            return None
        values = _as_list(input_cols)
        if not self.enabled:
            return values
        if self.column not in values:
            raise ValueError(
                f"structure_col={self.column!r} must be included in input_cols for structure-aware models."
            )
        return [self.column, *(value for value in values if value != self.column)]

    def resolve_categorical_cols(self, categorical_cols: Sequence[Any] | Any | None) -> list[Any]:
        """Use tabular category encoding for the user-facing structure ID."""

        values = _as_list(categorical_cols)
        if self.enabled and self.column not in values:
            values.insert(0, self.column)
        return values

    def merge_category_maps(self, category_maps: Any) -> dict[Any, dict[Any, int]]:
        """Inject the catalog order as an explicit category map."""

        if not self.enabled:
            return dict(category_maps or {})
        if category_maps is not None and not isinstance(category_maps, Mapping):
            raise TypeError("category_maps must be a mapping when structure_col is configured.")
        merged = dict(category_maps or {})
        existing = merged.get(self.column, merged.get(str(self.column)))
        if existing is not None and dict(existing) != self.category_map:
            raise ValueError(
                "The category map for structure_col conflicts with structure_catalog order."
            )
        merged[self.column] = self.category_map
        return merged

    def expanded_bounds(self, bounds: Any) -> Any:
        """Add the full structure-index range to column-addressed bounds."""

        if not self.enabled:
            return bounds
        if bounds is not None and not isinstance(bounds, Mapping):
            raise TypeError("structure_col requires bounds to be a column mapping when supplied.")
        expanded = dict(bounds or {})
        expanded[self.column] = [0.0, float(self.num_structures - 1)]
        return expanded

    def complete_categorical_bounds(
        self,
        bounds: Any,
        data: Any,
        *,
        categorical_cols: Sequence[Any] | Any | None,
        category_maps: Mapping[Any, Mapping[Any, int]] | None = None,
    ) -> Any:
        """Fill missing process-category bounds using the eventual encoded values.

        String/object categories are label encoded to ``0..K-1`` by the tabular
        data layer unless an explicit ``category_maps`` entry is supplied. Numeric
        categorical columns are preserved, so their observed numeric min/max are
        used instead. Continuous-process bounds are deliberately not inferred here.
        """

        if not self.enabled or not isinstance(bounds, Mapping):
            return bounds
        completed = dict(bounds)
        process_categorical = [
            column for column in _as_list(categorical_cols) if column != self.column
        ]
        if not process_categorical:
            return completed

        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError(
                "pandas is required to infer categorical process bounds for tabular structure models."
            ) from error
        if not isinstance(data, pd.DataFrame):
            missing = [
                column
                for column in process_categorical
                if column not in completed and str(column) not in completed
            ]
            if missing:
                raise ValueError(
                    "Categorical process bounds could not be inferred from non-DataFrame input; "
                    f"provide bounds or category_maps for {missing!r}."
                )
            return completed

        for column in process_categorical:
            if column in completed or str(column) in completed:
                continue
            explicit = _mapping_value(category_maps, column)
            if explicit is not None:
                codes = [int(value) for value in explicit.values()]
                if not codes:
                    raise ValueError(f"category_maps[{column!r}] must not be empty.")
                completed[column] = [float(min(codes)), float(max(codes))]
                continue
            if column not in data.columns:
                raise KeyError(f"Unknown categorical process column {column!r}.")
            values = data.loc[:, column].dropna()
            if values.empty:
                raise ValueError(
                    f"Cannot infer categorical bounds for {column!r}: no non-missing values."
                )
            if pd.api.types.is_numeric_dtype(values):
                completed[column] = [float(values.min()), float(values.max())]
            else:
                completed[column] = [0.0, float(values.nunique() - 1)]
        return completed

    def fixed_features_list(
        self,
        structure_ids: Sequence[Any] | Any | None = None,
        *,
        feature_index: int = 0,
    ) -> list[dict[int, float]]:
        """Return mixed-optimizer assignments for selected structure IDs."""

        if not self.enabled:
            raise RuntimeError("No tabular structure catalog is configured.")
        selected = self._ids if structure_ids is None else tuple(_as_list(structure_ids))
        if not selected:
            raise ValueError("structure_ids must contain at least one structure ID.")
        indices: list[int] = []
        for value in selected:
            if value not in self._id_to_index:
                raise KeyError(f"Unknown structure ID {value!r}.")
            index = self._id_to_index[value]
            if index not in indices:
                indices.append(index)
        return [{int(feature_index): float(index)} for index in indices]


__all__ = ["StructureTabularAdapter"]
