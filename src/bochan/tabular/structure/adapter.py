"""Tabular crystal-structure catalog and structure-index conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bochan.structure import ALIGNNGraphBuilder


class StructureTabularAdapter:
    """Own the tabular structure catalog and its discrete model coordinate.

    A user-facing structure identifier is converted to an integer index into an
    ordered ``structure_graphs`` bank. The index is a discrete selector, not a
    continuous optimization variable. Candidate generation should enumerate it
    through ``OptimizeConfig.fixed_features_list``.
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
        self._structure_graphs: tuple[Any, ...] | None = None

    @property
    def enabled(self) -> bool:
        return self.column is not None

    @property
    def structure_ids(self) -> tuple[Any, ...]:
        return self._ids

    @property
    def num_structures(self) -> int:
        return len(self._ids)

    @property
    def structure_graphs(self) -> tuple[Any, ...]:
        if not self.enabled:
            raise RuntimeError("No tabular structure catalog is configured.")
        if self._structure_graphs is None:
            builder = self.graph_builder or ALIGNNGraphBuilder()
            graphs = builder.build_many(tuple(self.catalog[value] for value in self._ids))
            if not isinstance(graphs, Sequence) or isinstance(graphs, (str, bytes)):
                raise TypeError("structure_graph_builder.build_many() must return a sequence.")
            if len(graphs) != self.num_structures:
                raise ValueError(
                    "The structure graph bank must contain one graph per catalog entry: "
                    f"{len(graphs)} != {self.num_structures}."
                )
            self._structure_graphs = tuple(graphs)
        return self._structure_graphs

    def prepare_frame(self, data: Any) -> Any:
        """Replace the configured structure-ID column by integer model indices."""

        if not self.enabled:
            return data
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError("structure_col requires pandas DataFrame input.")
        if self.column not in data.columns:
            raise KeyError(f"Unknown structure column {self.column!r}.")
        transformed = data.copy()
        mapped = transformed.loc[:, self.column].map(self._id_to_index)
        missing_mask = mapped.isna()
        if bool(missing_mask.any()):
            unknown = transformed.loc[missing_mask, self.column].drop_duplicates().tolist()
            raise KeyError(
                "Structure IDs are not present in structure_catalog: "
                f"{unknown!r}."
            )
        transformed.loc[:, self.column] = mapped.astype(float)
        return transformed

    def replace_input_cols(self, input_cols: Sequence[Any] | None) -> list[Any] | None:
        """Place the structure selector first, matching the ALIGNN model contract."""

        if not self.enabled or input_cols is None:
            return None if input_cols is None else list(input_cols)
        values = list(input_cols)
        if self.column not in values:
            raise ValueError(
                f"structure_col={self.column!r} must be included in input_cols for ALIGNN models."
            )
        return [self.column, *(value for value in values if value != self.column)]

    def resolve_categorical_cols(self, categorical_cols: Sequence[Any] | None) -> list[Any]:
        """Keep the structure selector outside generic categorical encodings."""

        values = list(categorical_cols or ())
        if self.enabled and self.column in values:
            raise ValueError(
                "structure_col is a discrete ALIGNN structure selector and must not be listed "
                "in categorical_cols; candidate generation enumerates it explicitly."
            )
        return values

    def expanded_bounds(self, bounds: Any) -> Any:
        """Add the full structure-index range to column-addressed bounds."""

        if not self.enabled:
            return bounds
        if bounds is not None and not isinstance(bounds, Mapping):
            raise TypeError("structure_col requires bounds to be a column mapping when supplied.")
        expanded = dict(bounds or {})
        expanded[self.column] = [0.0, float(self.num_structures - 1)]
        return expanded

    def fixed_features_list(
        self,
        structure_ids: Sequence[Any] | None = None,
        *,
        feature_index: int = 0,
    ) -> list[dict[int, float]]:
        """Return mixed-optimizer assignments for selected structure IDs."""

        if not self.enabled:
            raise RuntimeError("No tabular structure catalog is configured.")
        selected = self._ids if structure_ids is None else tuple(structure_ids)
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

    def inverse(self, data: Any) -> Any:
        """Restore structure IDs from the integer model coordinate."""

        if not self.enabled:
            return data
        import pandas as pd

        frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if self.column not in frame.columns:
            raise KeyError(f"Missing structure model column {self.column!r}.")
        values = frame.loc[:, self.column].astype(float)
        rounded = values.round()
        if not ((values - rounded).abs() <= 1e-6).all():
            raise ValueError("ALIGNN structure candidates must be integer-valued.")
        indices = rounded.astype(int)
        if ((indices < 0) | (indices >= self.num_structures)).any():
            raise ValueError(
                f"ALIGNN structure indices must be in [0, {self.num_structures - 1}]."
            )
        restored = frame.copy()
        restored.loc[:, self.column] = [self._ids[index] for index in indices.tolist()]
        return restored


__all__ = ["StructureTabularAdapter"]
