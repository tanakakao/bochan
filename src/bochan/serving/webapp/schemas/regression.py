"""Regression workflow request schemas for the Web API."""

from typing import Any, Literal

from pydantic import Field

from ._base import WebSchema
from .search import (
    AcquisitionSettingsSchema,
    KSparseSettingsSchema,
    OptimizerSettingsSchema,
    OutcomeConstraintSchema,
    SearchVariableSchema,
)
from .visualization import WebFeatureImportanceSettingsSchema


class RegressionRunRequest(WebSchema):
    """Run single- or multi-objective regression optimization."""

    dataset_id: str
    feature_columns: list[str]
    target_column: str | None = None
    target_columns: list[str] = Field(default_factory=list)
    direction: Literal["maximize", "minimize"] = "maximize"
    directions: dict[str, Literal["maximize", "minimize"]] = Field(default_factory=dict)
    model_type: str = "base"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    fit_maxiter: int = Field(default=128, ge=1)
    normalize: bool = True
    outcome_transform: bool = True
    input_perturbation: bool = False
    n_w: int = Field(default=16, ge=1)
    perturbation_std: float = Field(default=0.1, gt=0.0)
    search_space: list[SearchVariableSchema] = Field(default_factory=list)
    constraints: list[Any] = Field(default_factory=list)
    outcome_constraints: list[OutcomeConstraintSchema] = Field(default_factory=list)
    k_sparse: KSparseSettingsSchema | None = None
    acquisition: AcquisitionSettingsSchema = Field(default_factory=AcquisitionSettingsSchema)
    optimizer: OptimizerSettingsSchema = Field(default_factory=OptimizerSettingsSchema)
    drop_missing: bool = True
    cross_validation: bool = False
    cv_config: dict[str, Any] | None = None
    feature_importance: WebFeatureImportanceSettingsSchema | None = None


__all__ = ["RegressionRunRequest"]
