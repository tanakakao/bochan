"""Request and response schemas for BochanStudy FastAPI endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .configs import (
    AcquisitionConfigSchema,
    DataContextSchema,
    FitConfigSchema,
    ModelConfigSchema,
    OptimizeConfigSchema,
)
from .requests import APIRequest, TensorOptionsSchema

Direction = Literal["maximize", "minimize"]
TrialStateName = Literal["CANDIDATE", "RUNNING", "COMPLETED", "FAILED"]


class StudyAPIRequest(APIRequest):
    """Strict request base for the canonical Study HTTP contract."""

    model_config = ConfigDict(
        populate_by_name=False,
        protected_namespaces=(),
        extra="forbid",
    )


class StudyGenerationStepSchema(StudyAPIRequest):
    """HTTP representation of one canonical Study generation step."""

    name: str | None = None
    num_trials: int | None = None
    until_completed: int | None = None
    q: int | None = None
    acquisition_config: AcquisitionConfigSchema | str | None = None
    optimize_config: OptimizeConfigSchema | None = None
    data_context: DataContextSchema | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StudyGenerationScheduleSchema(StudyAPIRequest):
    """HTTP representation of a Study generation schedule."""

    steps: list[StudyGenerationStepSchema]


class StudyCreateRequest(StudyAPIRequest):
    """Create an in-memory :class:`bochan.api.BochanStudy`."""

    bo_model_config: ModelConfigSchema | None = Field(default=None, alias="model_config")
    fit_config: FitConfigSchema | None = None
    acquisition_config: AcquisitionConfigSchema | str | None = None
    optimize_config: OptimizeConfigSchema | None = None
    data_context: DataContextSchema | None = None
    bounds: Any | None = None
    n_initial_random: int = 0
    early_stopping_config: dict[str, Any] | None = None
    generation_schedule: (
        StudyGenerationScheduleSchema | list[StudyGenerationStepSchema] | None
    ) = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    initial_X: Any | None = None
    initial_Y: Any | None = None
    initial_metadata: list[dict[str, Any]] | dict[str, Any] | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class StudyRestoreRequest(StudyCreateRequest):
    """Restore trial state from a JSON snapshot and re-inject runtime configs."""

    snapshot: dict[str, Any]


class StudyObservationRequest(StudyAPIRequest):
    X: Any
    Y: Any
    metadata: list[dict[str, Any]] | dict[str, Any] | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class StudyAskRequest(StudyAPIRequest):
    q: int | None = None
    acquisition_config: AcquisitionConfigSchema | str | None = None
    optimize_config: OptimizeConfigSchema | None = None
    data_context: DataContextSchema | None = None
    mark_running: bool = False
    fit: bool = True
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class StudyTellRequest(StudyAPIRequest):
    trial_ids: list[int]
    values: Any
    state: TrialStateName = "COMPLETED"
    metadata: list[dict[str, Any]] | dict[str, Any] | None = None
    check_early_stop: bool = True
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class StudyTrialIdsRequest(StudyAPIRequest):
    trial_ids: list[int]


class StudyFailedRequest(StudyTrialIdsRequest):
    reason: str | None = None


class StudyParetoRequest(StudyAPIRequest):
    output_indices: list[int] | None = None
    directions: list[Direction] | None = None


class StudySummaryResponse(BaseModel):
    study_id: str
    task_type: str
    model_type: str
    n_trials: int
    n_completed: int
    n_pending: int
    n_failed: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    current_generation_step: Any | None = None
    stop_decision: Any | None = None


class StudyListResponse(BaseModel):
    study_ids: list[str]


class StudyAskResponse(BaseModel):
    study_id: str
    trial_ids: list[int]
    candidates: Any
    acq_value: Any | None = None


class StudyTrialsResponse(BaseModel):
    study_id: str
    trials: list[dict[str, Any]]


class StudyBestResponse(BaseModel):
    study_id: str
    result: dict[str, Any]


class StudyParetoResponse(BaseModel):
    study_id: str
    output_indices: list[int]
    directions: list[str]
    pareto_trials: list[dict[str, Any]]
    trials: list[dict[str, Any]]


class StudyHistoryResponse(BaseModel):
    study_id: str
    output_index: int
    direction: str
    records: list[dict[str, Any]]


class StudySnapshotResponse(BaseModel):
    study_id: str
    snapshot: dict[str, Any]


class StudyDeleteResponse(BaseModel):
    study_id: str


__all__ = [
    "StudyAPIRequest",
    "StudyAskRequest",
    "StudyAskResponse",
    "StudyBestResponse",
    "StudyCreateRequest",
    "StudyDeleteResponse",
    "StudyFailedRequest",
    "StudyGenerationScheduleSchema",
    "StudyGenerationStepSchema",
    "StudyHistoryResponse",
    "StudyListResponse",
    "StudyObservationRequest",
    "StudyParetoRequest",
    "StudyParetoResponse",
    "StudyRestoreRequest",
    "StudySnapshotResponse",
    "StudySummaryResponse",
    "StudyTellRequest",
    "StudyTrialIdsRequest",
    "StudyTrialsResponse",
]
