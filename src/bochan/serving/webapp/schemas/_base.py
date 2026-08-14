"""Shared schema configuration for the Web API."""

from pydantic import BaseModel, ConfigDict


class WebSchema(BaseModel):
    """Strict base request schema used by the Web API."""

    model_config = ConfigDict(extra="forbid")


__all__ = ["WebSchema"]
