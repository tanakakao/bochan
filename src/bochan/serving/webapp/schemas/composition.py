"""Composition request schemas for the Web API."""

from typing import Literal

from pydantic import Field

from ._base import WebSchema
from .regression import RegressionRunRequest


class CompositionElementTermSchema(WebSchema):
    """One coefficient multiplied by one element amount."""

    element: str
    coefficient: float = 1.0


class CompositionElementConstraintSchema(WebSchema):
    """Linear equality or inequality between element amounts."""

    terms: list[CompositionElementTermSchema] = Field(min_length=1)
    operator: Literal["=", "<=", ">="] = "="
    rhs: float = 0.0
    basis: Literal["atomic_amount", "weight_amount"] = "atomic_amount"


class CompositionSettingsSchema(WebSchema):
    """Single-formula ratio settings accepted by validation and optimization."""

    enabled: bool = True
    column: str = Field(min_length=1)
    elements: list[str] = Field(default_factory=list)
    normalization: Literal["atomic_fraction", "weight_fraction"] = "atomic_fraction"
    representation: Literal["fractions", "fraction", "clr", "alr", "ilr"] = "ilr"
    reference_element: str | None = None
    pseudocount: float = Field(default=1e-12, gt=0.0)
    precision: int = Field(default=6, ge=1, le=12)
    total: float = Field(default=1.0, gt=0.0)
    coordinate_bounds: tuple[float, float] = (-8.0, 8.0)
    min_components: int = Field(default=1, ge=1)
    max_components: int | None = Field(default=None, ge=1)
    required_components: list[str] = Field(default_factory=list)
    forbidden_components: list[str] = Field(default_factory=list)
    support_selection: Literal["repair", "best_subset"] = "repair"
    best_subset_strategy: Literal["exact", "beam", "auto"] = "auto"
    best_subset_max_combinations: int = Field(default=2000, ge=1)
    best_subset_beam_width: int = Field(default=8, ge=1)
    best_subset_beam_steps: int = Field(default=4, ge=0)
    best_subset_max_evaluations: int = Field(default=200, ge=1)
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    steps: dict[str, float] = Field(default_factory=dict)
    element_constraints: list[CompositionElementConstraintSchema] = Field(default_factory=list)


class CompositionValidationRequest(WebSchema):
    """Formula rows and the same settings used by the optimization request."""

    formulas: list[str] = Field(min_length=1)
    settings: CompositionSettingsSchema


class CompositionRegressionRunRequest(WebSchema):
    """Existing Web regression payload plus one typed composition configuration."""

    run: RegressionRunRequest
    composition: CompositionSettingsSchema


__all__ = [
    "CompositionElementConstraintSchema",
    "CompositionElementTermSchema",
    "CompositionRegressionRunRequest",
    "CompositionSettingsSchema",
    "CompositionValidationRequest",
]
