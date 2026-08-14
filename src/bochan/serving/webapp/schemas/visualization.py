"""Visualization request schemas for the Web API."""

from typing import Any, Literal

from pydantic import Field

from bochan.serving.fastapi.schemas.tabular import (
    FeatureImportanceConfigRequest,
    FeatureImportanceVisualizationRequest,
)

from ._base import WebSchema


class WebFeatureImportanceSettingsSchema(WebSchema):
    """Optional feature-importance execution and presentation settings."""

    enabled: bool = False
    source: Literal["auto", "training", "cross_validation"] = "auto"
    config: FeatureImportanceConfigRequest = Field(default_factory=FeatureImportanceConfigRequest)
    visualization: FeatureImportanceVisualizationRequest = Field(default_factory=FeatureImportanceVisualizationRequest)


class VisualizationRequestSchema(WebSchema):
    """Select one existing Plotly visualization for a fitted Web run."""

    kind: Literal["yyplot", "target_relation", "pareto", "1d", "2d", "ternary"]
    target: str | None = None
    target_x: str | None = None
    target_y: str | None = None
    show_pareto_front: bool = False
    features: list[str] = Field(default_factory=list)
    fixed_values: dict[str, Any] = Field(default_factory=dict)
    show_type: Literal["pred", "acqf"] = "pred"
    n: int = Field(default=50, ge=10, le=150)
    sum_value: float | None = None


__all__ = ["VisualizationRequestSchema", "WebFeatureImportanceSettingsSchema"]
