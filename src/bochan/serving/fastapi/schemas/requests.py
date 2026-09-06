"""Request schemas for bochan FastAPI serving."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .configs import (
    AcquisitionConfigSchema,
    DataContextSchema,
    FitConfigSchema,
    ModelConfigSchema,
    OptimizeConfigSchema,
)


class APIRequest(BaseModel):
    """Base request model with API-supported aliases enabled."""

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class TensorOptionsSchema(APIRequest):
    """Tensor conversion options for JSON payloads."""

    dtype: str = "float64"
    device: str | None = None


class LLMConfigSchema(APIRequest):
    """Provider-independent LLM settings for candidate generation."""

    provider: Literal["openai", "gemini"] | str = "openai"
    model: str = "gpt-4.1-mini"
    api_key_env: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 4096
    timeout: float | None = 60.0
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class LLMContextSchema(APIRequest):
    """Optional domain context passed to the LLM prompt builder."""

    variable_names: list[str] | None = None
    target_names: list[str] | None = None
    variable_descriptions: dict[str, str] = Field(default_factory=dict)
    target_descriptions: dict[str, str] = Field(default_factory=dict)
    domain_notes: list[str] = Field(default_factory=list)
    candidate_policy: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FitModelRequest(APIRequest):
    bo_model_config: ModelConfigSchema = Field(alias="model_config")
    train_X: Any
    train_Y: Any
    train_Yvar: Any | None = None
    train_cost: Any | None = None
    bounds: Any | None = None
    fit_config: FitConfigSchema | None = None
    data_context: DataContextSchema | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class LLMPlanRequest(APIRequest):
    """Infer bochan configuration dictionaries from a natural-language goal."""

    goal: str
    mode: Literal["model_config", "full"] = "full"
    train_X: Any | None = None
    train_Y: Any | None = None
    bounds: Any | None = None
    llm_config: LLMConfigSchema | None = None
    llm_context: LLMContextSchema | None = None
    planner_response: Any | None = None
    bo_model_config: dict[str, Any] | None = Field(default=None, alias="model_config")
    fit_config: dict[str, Any] | None = None
    acquisition_config: dict[str, Any] | None = None
    optimize_config: dict[str, Any] | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class AutoCandidateRequest(APIRequest):
    """Plan model settings, fit a model, and generate candidates in one request."""

    goal: str
    train_X: Any
    train_Y: Any
    train_Yvar: Any | None = None
    train_cost: Any | None = None
    bounds: Any | None = None
    llm_config: LLMConfigSchema | None = None
    llm_context: LLMContextSchema | None = None
    planner_response: Any | None = None
    bo_model_config: dict[str, Any] | None = Field(default=None, alias="model_config")
    fit_config: dict[str, Any] | None = None
    acquisition_config: dict[str, Any] | None = None
    optimize_config: dict[str, Any] | None = None
    data_context: DataContextSchema | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class RefitModelRequest(APIRequest):
    """Request body for refitting an existing in-memory optimizer."""

    fit_config: FitConfigSchema | None = None


class TellRequest(APIRequest):
    """Append observations to an existing optimizer and optionally refit."""

    new_X: Any
    new_Y: Any
    new_Yvar: Any | None = None
    new_cost: Any | None = None
    refit: bool = True
    fit_config: FitConfigSchema | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class PredictRequest(APIRequest):
    X: Any
    return_type: Literal["posterior", "mean", "variance", "mean_variance"] = "mean_variance"
    posterior_kwargs: dict[str, Any] = Field(default_factory=dict)
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class _MultiFidelityCandidateMixin(BaseModel):
    target_fidelity: float | None = None
    cost_config: dict[str, Any] | None = None
    fidelity_values: list[float] | dict[int, list[float]] | None = None
    fidelity_assignments: list[dict[int, float]] | None = None
    optimize_fidelity: bool | None = None

    @model_validator(mode="after")
    def validate_query_fidelity_mode(self):
        active_modes = sum(
            (
                self.fidelity_values is not None,
                self.fidelity_assignments is not None,
                bool(self.optimize_fidelity),
            )
        )
        if active_modes > 1:
            raise ValueError(
                "Specify only one of fidelity_values, fidelity_assignments, or "
                "optimize_fidelity=True."
            )
        return self


class CandidateRequest(APIRequest, _MultiFidelityCandidateMixin):
    acq_config: AcquisitionConfigSchema = Field(alias="acquisition_config")
    opt_config: OptimizeConfigSchema = Field(default_factory=OptimizeConfigSchema, alias="optimize_config")
    data_context: DataContextSchema | None = None
    bounds: Any | None = None
    target_task: int | None = None
    goal: str | None = None
    llm_config: LLMConfigSchema | None = None
    llm_context: LLMContextSchema | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class CompareCandidatesRequest(APIRequest, _MultiFidelityCandidateMixin):
    """Generate candidates for multiple acquisition functions on the same fitted model."""

    acq_configs: list[AcquisitionConfigSchema] = Field(alias="acquisition_configs")
    opt_config: OptimizeConfigSchema = Field(default_factory=OptimizeConfigSchema, alias="optimize_config")
    data_context: DataContextSchema | None = None
    bounds: Any | None = None
    target_task: int | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class SuggestRequest(APIRequest, _MultiFidelityCandidateMixin):
    """Stateless fit-and-candidate request."""

    bo_model_config: ModelConfigSchema = Field(default_factory=ModelConfigSchema, alias="model_config")
    train_X: Any
    train_Y: Any
    train_Yvar: Any | None = None
    train_cost: Any | None = None
    bounds: Any
    fit_config: FitConfigSchema | None = None
    acquisition_config: AcquisitionConfigSchema
    optimize_config: OptimizeConfigSchema = Field(default_factory=OptimizeConfigSchema)
    data_context: DataContextSchema | None = None
    tensor_options: TensorOptionsSchema = Field(default_factory=TensorOptionsSchema)


class SaveModelRequest(APIRequest):
    """Request body for saving an in-memory optimizer to disk."""

    filename: str | None = None
    overwrite: bool = False


class LoadModelRequest(APIRequest):
    """Request body for loading a trusted optimizer artifact."""

    filename: str
    map_location: str | None = "cpu"
    trust_pickle: bool = False
