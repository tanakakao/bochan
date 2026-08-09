"""Observation-aware request and response schemas for tabular FastAPI endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .configs import FitConfigSchema, ModelConfigSchema
from .requests import APIRequest
from .tabular_core import (
    CrossValidationRequest,
    FeatureImportanceConfigRequest,
    FeatureImportanceGroupRequest,
    FeatureImportanceSummaryRecord,
    FeatureImportanceVisualizationRequest,
    TabularCandidateRequest,
    TabularCandidateResponse,
    TabularFeatureImportanceRequest,
    TabularFeatureImportanceResponse,
    TabularModelFitResponse,
    TabularModelLoadResponse,
    TabularPayload,
    TabularPredictRequest,
    TabularPredictResponse,
    TabularTellRequest,
)
from .tabular_core import TabularFitModelRequest as _CoreTabularFitModelRequest


class ExperimentFailureConfigRequest(APIRequest):
    """HTTP configuration for the independent experiment-success classifier."""

    failure_model_config: ModelConfigSchema | None = None
    failure_fit_config: FitConfigSchema | None = None
    min_success_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    eta: float = Field(default=0.05, gt=0.0)
    reduce_q: Literal["prod", "min", "mean"] = "prod"


class TabularFitModelRequest(_CoreTabularFitModelRequest):
    """Fit a tabular optimizer with explicit target and experiment states."""

    target_missing_strategy: Literal["drop", "keep"] = "drop"
    experiment_status_col: str | None = None
    experiment_failure: ExperimentFailureConfigRequest | None = None

    @model_validator(mode="after")
    def validate_observation_fields(self):
        """Keep experiment state independent from feature/target columns."""

        status = self.experiment_status_col
        targets = (
            list(self.target_cols)
            if isinstance(self.target_cols, list)
            else [self.target_cols]
        )
        if status is not None:
            if status in self.input_cols:
                raise ValueError("experiment_status_col must not be included in input_cols.")
            if status in targets:
                raise ValueError("experiment_status_col must not be included in target_cols.")
        if self.experiment_failure is not None and status is None:
            raise ValueError(
                "experiment_failure requires experiment_status_col so success/failure "
                "labels are explicit."
            )
        if self.target_missing_strategy == "keep" and self.impute_targets:
            raise ValueError(
                "target_missing_strategy='keep' cannot be combined with impute_targets=True."
            )
        return self


__all__ = [
    "CrossValidationRequest",
    "ExperimentFailureConfigRequest",
    "FeatureImportanceConfigRequest",
    "FeatureImportanceGroupRequest",
    "FeatureImportanceSummaryRecord",
    "FeatureImportanceVisualizationRequest",
    "TabularCandidateRequest",
    "TabularCandidateResponse",
    "TabularFitModelRequest",
    "TabularFeatureImportanceRequest",
    "TabularFeatureImportanceResponse",
    "TabularModelFitResponse",
    "TabularModelLoadResponse",
    "TabularPayload",
    "TabularPredictRequest",
    "TabularPredictResponse",
    "TabularTellRequest",
]
