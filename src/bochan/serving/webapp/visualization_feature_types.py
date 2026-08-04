"""Source-column feature type handling for Web visualization controls."""

from __future__ import annotations

from typing import Any


def _source_numeric_features(session: Any) -> list[str]:
    """Return numeric source columns without reusing transformed feature indices.

    Composition columns expand into multiple model coordinates, so categorical
    dimensions from the fitted tensor dataset no longer align positionally with
    the original Web feature columns. Prefer source column dtypes and named
    category maps, and retain the positional behavior only as a fallback when
    the source frame does not contain a requested column.
    """

    import pandas as pd

    dataset = getattr(session.tabular_optimizer, "dataset", None)
    category_maps = dict(getattr(dataset, "category_maps", None) or {})
    categorical_names = {str(name) for name in category_maps}
    cat_dims = set(int(value) for value in (getattr(dataset, "cat_dims", None) or []))
    data_columns = set(getattr(session.data, "columns", []))

    numeric: list[str] = []
    for index, name in enumerate(session.feature_columns):
        if str(name) in categorical_names:
            continue

        if name in data_columns:
            dtype = session.data[name].dtype
            if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                numeric.append(name)
            continue

        if index not in cat_dims:
            numeric.append(name)
    return numeric


def install_visualization_feature_type_compat() -> None:
    """Install source-aware feature typing before the FastAPI app is imported."""

    from . import visualization_sessions

    if getattr(visualization_sessions, "_source_feature_type_compat", False):
        return
    visualization_sessions._numeric_features = _source_numeric_features
    visualization_sessions._source_feature_type_compat = True


__all__ = ["install_visualization_feature_type_compat"]
