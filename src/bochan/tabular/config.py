'''Tabular configuration objects for pandas / numpy friendly APIs.'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

ColumnKey = str | int


@dataclass
class TabularDataConfig:
    '''Configuration for converting tabular data into bochan tensors.'''

    input_cols: Sequence[ColumnKey] | None = None
    target_cols: Sequence[ColumnKey] | ColumnKey | None = None
    categorical_cols: Sequence[ColumnKey] = field(default_factory=list)
    target_categorical_cols: Sequence[ColumnKey] | None = None

    bounds: Any | Mapping[ColumnKey, Sequence[float]] | None = None
    dtype: Any | None = None
    device: Any | None = None

    dropna: bool = True
    encode_categories: bool = True
    category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None
    target_category_maps: Mapping[ColumnKey, Mapping[Any, int]] | None = None
    return_original_categories: bool = True
