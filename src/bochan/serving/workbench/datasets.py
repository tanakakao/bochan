"""Dataset loading, profiling, and in-memory storage for the Web workbench."""

from __future__ import annotations

import base64
import io
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DatasetRecord:
    """Loaded tabular dataset kept in memory by a workbench process."""

    dataset_id: str
    name: str
    data: Any
    source_type: str
    profile: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetStore:
    """Simple in-memory dataset store for the Web workbench."""

    def __init__(self) -> None:
        self._records: dict[str, DatasetRecord] = {}

    def add(self, record: DatasetRecord) -> None:
        self._records[record.dataset_id] = record

    def get(self, dataset_id: str) -> DatasetRecord:
        try:
            return self._records[dataset_id]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset_id: {dataset_id}") from exc

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "dataset_id": record.dataset_id,
                "name": record.name,
                "source_type": record.source_type,
                "n_rows": record.profile["n_rows"],
                "n_columns": record.profile["n_columns"],
            }
            for record in self._records.values()
        ]


def decode_base64_payload(value: str) -> bytes:
    """Decode browser FileReader base64 payloads.

    The UI may send either a raw base64 string or a data URL. Both are accepted.
    """

    payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    return base64.b64decode(payload)


def load_dataframe_from_payload(
    *,
    source_type: Literal["csv", "excel", "sqlite", "duckdb"],
    content_base64: str | None = None,
    name: str | None = None,
    encoding: str = "utf-8-sig",
    sep: str | None = None,
    sheet_name: str | int | None = 0,
    sql: str | None = None,
    database_path: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a pandas DataFrame from a browser or local database payload."""

    import pandas as pd

    metadata: dict[str, Any] = {"name": name, "source_type": source_type}
    if source_type == "csv":
        if not content_base64:
            raise ValueError("content_base64 is required for CSV input.")
        raw = decode_base64_payload(content_base64)
        text = raw.decode(encoding)
        read_kwargs: dict[str, Any] = {}
        if sep:
            read_kwargs["sep"] = sep
        else:
            read_kwargs["sep"] = None
            read_kwargs["engine"] = "python"
        return pd.read_csv(io.StringIO(text), **read_kwargs), metadata

    if source_type == "excel":
        if not content_base64:
            raise ValueError("content_base64 is required for Excel input.")
        raw = decode_base64_payload(content_base64)
        return pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name), metadata

    if source_type == "sqlite":
        if not database_path:
            raise ValueError("database_path is required for SQLite input.")
        if not sql:
            raise ValueError("sql is required for SQLite input.")
        with sqlite3.connect(database_path) as conn:
            return pd.read_sql_query(sql, conn), {**metadata, "database_path": database_path, "sql": sql}

    if source_type == "duckdb":
        if not database_path:
            raise ValueError("database_path is required for DuckDB input.")
        if not sql:
            raise ValueError("sql is required for DuckDB input.")
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("Install DuckDB support to use DuckDB data sources.") from exc
        with duckdb.connect(database_path) as conn:
            return conn.execute(sql).df(), {**metadata, "database_path": database_path, "sql": sql}

    raise ValueError(f"Unsupported source_type: {source_type}")


def build_dataset_record(
    *,
    data: Any,
    name: str,
    source_type: str,
    metadata: dict[str, Any] | None = None,
) -> DatasetRecord:
    """Create a dataset record from a pandas DataFrame."""

    dataset_id = str(uuid.uuid4())
    profile = profile_dataframe(data)
    return DatasetRecord(
        dataset_id=dataset_id,
        name=name,
        data=data,
        source_type=source_type,
        profile=profile,
        metadata=metadata or {},
    )


def dataframe_preview(data: Any, *, limit: int = 100) -> list[dict[str, Any]]:
    """Return a JSON-friendly preview of a DataFrame."""

    import pandas as pd

    preview = data.head(max(0, int(limit))).copy()
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")


def profile_dataframe(data: Any) -> dict[str, Any]:
    """Build a compact UI profile for a DataFrame."""

    import pandas as pd

    columns: list[dict[str, Any]] = []
    for name in data.columns:
        series = data[name]
        missing_count = int(series.isna().sum())
        non_null = series.dropna()
        unique_count = int(non_null.nunique(dropna=True))
        kind = infer_column_kind(series)
        column: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "kind": kind,
            "missing_count": missing_count,
            "missing_rate": float(missing_count / max(len(series), 1)),
            "unique_count": unique_count,
        }
        if pd.api.types.is_numeric_dtype(series):
            column.update(
                {
                    "min": _safe_float(non_null.min()) if len(non_null) else None,
                    "max": _safe_float(non_null.max()) if len(non_null) else None,
                    "mean": _safe_float(non_null.mean()) if len(non_null) else None,
                    "std": _safe_float(non_null.std()) if len(non_null) else None,
                }
            )
        else:
            values = [str(value) for value in non_null.astype(str).unique()[:30]]
            column["values"] = values
        columns.append(column)

    return {
        "n_rows": int(len(data)),
        "n_columns": int(len(data.columns)),
        "columns": columns,
    }


def infer_column_kind(series: Any) -> str:
    """Infer a UI-friendly column kind."""

    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    non_null = series.dropna()
    unique_count = int(non_null.nunique(dropna=True))
    if unique_count <= max(20, int(0.2 * max(len(non_null), 1))):
        return "categorical"
    return "string"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


__all__ = [
    "DatasetRecord",
    "DatasetStore",
    "build_dataset_record",
    "dataframe_preview",
    "decode_base64_payload",
    "infer_column_kind",
    "load_dataframe_from_payload",
    "profile_dataframe",
]
