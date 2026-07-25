"""Request and response schemas for tabular FastAPI endpoints."""

# ruff: noqa: I001

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .configs import (
    FitConfigSchema,
    ModelConfigSchema,
    MultiOutputConfigSchema,
    OutcomeConstraintConfigSchema,
)
from .requests import APIRequest


TabularPayload = list[dict[str, Any]] | dict[str, list[Any]]


class TabularFitModelRequest(APIRequest):
    """Fit a :class:`TabularBayesianOptimizer` from JSON tabular data."""

    data: TabularPayload
    bo_model_config: ModelConfigSchema = Field(alias="model_config")
    fit_config: FitConfigSchema | None = None
    multi_output_config: MultiOutputConfigSchema | None = None
    input_cols: list[str]
    target_cols: list[str] | str
    categorical_cols: list[str] = Field(default_factory=list)
    target_categorical_cols: list[str] | None = None
    bounds: Any | None = None
    dtype: str = "float64"
    device: str | None = None
    dropna: bool = True
    missing_strategy: str | None = None
    continuous_impute_strategy: str = "mean"
    categorical_impute_strategy: str = "mode"
    impute_targets: bool = False
    impute_random_state: int | None = None
    impute_max_iter: int = 10
    multiple_impute_sample_posterior: bool = False
    encode_categories: bool = True
    category_maps: dict[str, dict[Any, int]] | None = None
    target_category_maps: dict[str, dict[Any, int]] | None = None
    return_original_categories: bool = True


class TabularTellRequest(APIRequest):
    """Append records containing fitted feature and target columns."""

    data: TabularPayload
    refit: bool = True
    fit_config: FitConfigSchema | None = None


class TabularPredictRequest(APIRequest):
    """Predict from records containing the fitted feature columns."""

    data: TabularPayload
    return_type: Literal["dataframe", "posterior", "mean", "variance", "mean_variance"] = "dataframe"
    include_input: bool = False
    posterior_kwargs: dict[str, Any] = Field(default_factory=dict)


class TabularCandidateRequest(APIRequest):
    """Generate candidates while preserving tabular column names and labels.

    ``outcome_constraint_config`` constrains predicted model outputs. Linear
    constraints on input columns and candidate repair settings belong to
    ``optimize_config``. For convenience, ``constraints`` and ``repair_config``
    may also be supplied at this request's top level; the router moves them into
    the effective optimize config before calling ``TabularBayesianOptimizer``.

    The direct objective and outcome-constraint fields mirror
    :meth:`TabularBayesianOptimizer.candidate`. Only fields explicitly present in
    the JSON request are forwarded, so omitted values retain the optimizer's
    normal ``UNSET`` semantics.
    """

    acq_config: dict[str, Any] = Field(alias="acquisition_config")
    opt_config: dict[str, Any] = Field(default_factory=dict, alias="optimize_config")
    bounds: Any | None = None

    constraints: Any | None = None
    repair_config: dict[str, Any] | None = None

    outcome_constraint_config: OutcomeConstraintConfigSchema | None = None
    objective_mode: Literal["auto", "none", "scalar", "multi_output"] | None = None
    objective_output: Any | None = None
    objective_outputs: list[Any] | None = None
    objective_specs: list[Any] | None = None
    objective_directions: list[Any] | None = None
    objective_weights: list[float] | None = None
    objective_eq_targets: list[float | None] | None = None
    objective_direction: Any | None = None
    objective_weight: float | None = None
    objective_eq_target: float | None = None
    objective_n_w: int | None = None
    objective_risk_type: str | None = None
    objective_alpha: float | None = None
    objective_maximize: bool | None = None
    objective_aggregate_mean_when_no_risk: bool | None = None
    objective_allow_unexpanded: bool | None = None
    objective_utility_values: Any | None = None
    objective_ordinal_likelihood: Any | None = None
    evo_method: str | None = None


class TabularModelFitResponse(BaseModel):
    model_id: str
    task_type: str
    model_type: str
    n_train: int | None = None
    feature_names: list[Any] = Field(default_factory=list)
    target_names: list[Any] = Field(default_factory=list)
    categorical_cols: list[Any] = Field(default_factory=list)
    category_maps: dict[str, Any] = Field(default_factory=dict)
    target_category_maps: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TabularModelLoadResponse(TabularModelFitResponse):
    """Response returned after loading a common tabular ``.bochan.pt`` file."""

    filename: str
    path: str


class TabularPredictResponse(BaseModel):
    model_id: str
    columns: list[str] | None = None
    records: list[dict[str, Any]] | None = None
    value: Any | None = None


class TabularCandidateResponse(BaseModel):
    model_id: str
    columns: list[str]
    candidates: list[dict[str, Any]]
    acq_value: Any


__all__ = [
    "TabularCandidateRequest",
    "TabularCandidateResponse",
    "TabularFitModelRequest",
    "TabularModelFitResponse",
    "TabularModelLoadResponse",
    "TabularPayload",
    "TabularPredictRequest",
    "TabularPredictResponse",
    "TabularTellRequest",
]
