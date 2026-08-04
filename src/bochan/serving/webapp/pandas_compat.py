"""Pandas compatibility helpers used by the Web application."""

from __future__ import annotations

from functools import wraps
from typing import Any

_INSTALLED = False


def _categorical_columns(data: Any, config: Any, pd: Any) -> list[Any]:
    """Return columns that may be label-encoded by the tabular converter."""

    columns = list(config.categorical_cols or ())
    target_columns = config.target_cols
    if target_columns is None:
        targets: list[Any] = []
    elif isinstance(target_columns, (str, bytes)):
        targets = [target_columns]
    else:
        try:
            targets = list(target_columns)
        except TypeError:
            targets = [target_columns]

    if config.target_categorical_cols is None:
        columns.extend(
            column
            for column in targets
            if column in data.columns
            and not pd.api.types.is_numeric_dtype(data.loc[:, column])
        )
    else:
        columns.extend(list(config.target_categorical_cols or ()))

    return list(dict.fromkeys(columns))


def _object_backed_categories(data: Any, config: Any, pd: Any) -> Any:
    """Copy extension-string category columns to object-backed columns.

    Pandas 3 rejects assigning integer label codes into a ``StringDtype``
    column through ``.loc``. The shared converter intentionally performs that
    assignment in-place, so Web requests first provide an object-backed copy.
    """

    if not isinstance(data, pd.DataFrame):
        return data

    converted = data.copy()
    for column in _categorical_columns(converted, config, pd):
        if column not in converted.columns:
            continue
        values = converted.loc[:, column]
        if pd.api.types.is_string_dtype(values.dtype) and not pd.api.types.is_object_dtype(values.dtype):
            converted[column] = values.astype(object)
    return converted


def install_pandas_string_category_compat() -> None:
    """Install a Web-scoped compatibility wrapper around DataFrame conversion."""

    global _INSTALLED
    if _INSTALLED:
        return

    import pandas as pd

    from bochan.tabular import converter

    original = converter.dataframe_to_tensors

    @wraps(original)
    def dataframe_to_tensors_compat(data: Any, config: Any) -> Any:
        compatible = _object_backed_categories(data, config, pd)
        return original(compatible, config)

    converter.dataframe_to_tensors = dataframe_to_tensors_compat
    _INSTALLED = True


__all__ = ["install_pandas_string_category_compat"]
