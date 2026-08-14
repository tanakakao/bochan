"""Search and optimization request schemas for the Web API."""

from typing import Any, Literal

from pydantic import Field

from ._base import WebSchema


class SearchVariableSchema(WebSchema):
    """Search-space settings for one feature column."""

    name: str
    type: Literal["auto", "numeric", "categorical"] = "auto"
    lower: float | None = None
    upper: float | None = None
    step: float | None = None
    fixed: bool = False
    fixed_value: Any | None = None
    categories: list[Any] | None = None


class OutcomeConstraintSchema(WebSchema):
    """Threshold constraint on one selected target in the original value scale."""

    id: str | None = None
    target: str
    operator: Literal["<=", ">="]
    value: float


class AcquisitionSettingsSchema(WebSchema):
    """Acquisition-function settings exposed by the Web workbench."""

    name: str = "EI"
    beta: float = 2.0
    acqf_kwargs: dict[str, Any] = Field(default_factory=dict)


class OptimizerSettingsSchema(WebSchema):
    """Candidate optimizer settings."""

    name: str = "optimize_acqf"
    q: int = Field(default=1, ge=1)
    num_restarts: int = Field(default=10, ge=1)
    raw_samples: int = Field(default=256, ge=1)
    sequential: bool = True
    minimum_candidate_distance_ratio: float = Field(default=1e-3, ge=0.0, le=1.0)


class KSparseSettingsSchema(WebSchema):
    """Limit the number of non-zero variables selected from a feature subset."""

    enabled: bool = False
    columns: list[str] = Field(default_factory=list)
    k: int = Field(default=1, ge=1)
    score: Literal["abs", "value"] = "abs"
    support_selection: str = "topk"
    final_priority: str = "grid"


__all__ = [
    "AcquisitionSettingsSchema",
    "KSparseSettingsSchema",
    "OptimizerSettingsSchema",
    "OutcomeConstraintSchema",
    "SearchVariableSchema",
]
