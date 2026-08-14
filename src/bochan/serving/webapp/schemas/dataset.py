"""Dataset request schemas for the Web API."""

from typing import Literal

from ._base import WebSchema


class DatasetLoadRequest(WebSchema):
    """Browser-uploaded tabular dataset encoded as base64."""

    source_type: Literal["csv", "excel"] = "csv"
    name: str | None = None
    content_base64: str
    encoding: str = "utf-8-sig"
    sep: str | None = None
    sheet_name: str | int | None = 0


__all__ = ["DatasetLoadRequest"]
