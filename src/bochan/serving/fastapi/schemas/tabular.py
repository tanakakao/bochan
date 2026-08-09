"""Request and response schemas for tabular FastAPI endpoints."""

# ruff: noqa: I001

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .configs import (
    FitConfigSchema,
    ModelConfigSchema,
    MultiOutputConfigSchema,
    OutcomeConstraintConfigSchema,
)
from .requests import APIRequest


TabularPayload = list[dict[str, Any]] | dict[str, list[Any]]


class FeatureImportanceGroupRequest(APIRequest):
    """Column-addressed permutation group."""

    name: str
    columns: list[str] = Field(min_length=1)
    role: str = "group"


class FeatureImportanceConfigRequest(APIRequest):
    """HTTP defaults for core feature-importance inspection."""

    predictive_methods: list[Literal["permutation"]] = Field(
        default_factory=lambda: ["permutation"]
    )
    diagnostic_methods: list[str] = Field(default_factory=lambda: ["auto"])
    n_repeats: int = Field(default=10, ge=1, le=100)
    random_state: int | None = 0
    scoring: str = "auto"
    scoring_direction: Literal["auto", "minimize", "maximize"] = "auto"
    compute_noise_importance: bool = True
    compute_classwise_importance: bool = False
    normalize_importance: bool = False
    clip_negative_importance: bool = False
    feature_groups: list[FeatureImportanceGroupRequest] = Field(default_factory=list)
    return_per_repeat_values: bool = False
    batch_size: int | None = Field(default=None, ge=1)
    unsupported_method_policy: Literal["raise", "warn", "skip"] = "warn"
    error_policy: Literal["raise", "warn", "skip"] = "warn"


class CrossValidationRequest(APIRequest):
    """JSON-safe subset of the core cross-validation configuration."""

    splitter: Literal["auto", "kfold", "stratified", "stratified_kfold", "loo"] = "auto"
    n_splits: int = Field(default=5, ge=2)
    shuffle: bool = True
    random_state: int | None = 0
    classification_average: Literal[
        "auto", "binary", "micro", "macro", "weighted"
    ] = "auto"
    classification_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    positive_class: int | str | float = 1
    zero_division: Literal[0, 1] = 0
    mape_zero_policy: Literal["warn_nan", "ignore", "clip"] = "warn_nan"
    mape_epsilon: float = Field(default=1e-8, gt=0.0)
    feature_importance_config: FeatureImportanceConfigRequest | None = None


class FeatureImportanceVisualizationRequest(APIRequest):
    """Presentation-only feature-importance settings."""

    normalized: bool = False
    top_k: int | None = Field(default=15, ge=1, le=100)
    rank_by: Literal["value", "absolute"] = "value"
    include_negative: bool = True
    show_error_bars: bool = True
    include_predictive: bool = True
    include_noise: bool = True
    include_classwise: bool = False


class TabularFeatureImportanceRequest(APIRequest):
    """Evaluate a fitted tabular model on training or external data."""

    data: TabularPayload | None = None
    config: FeatureImportanceConfigRequest = Field(
        default_factory=FeatureImportanceConfigRequest
    )
    visualization: FeatureImportanceVisualizationRequest | None = Field(
        default_factory=FeatureImportanceVisualizationRequest
    )


class FeatureImportanceSummaryRecord(BaseModel):
    """Compact long-form importance row."""

    output_name: str
    task_type: str
    importance_kind: str
    method: str
    feature: str
    rank: float | int | None = None
    mean: float | None = None
    std: float | None = None
    normalized_mean: float | None = None
    metric_name: str | None = None
    baseline_metric: float | None = None
    feature_type: str | None = None
    role: str | None = None
    indices: list[int] = Field(default_factory=list)


class TabularFeatureImportanceResponse(BaseModel):
    """JSON-safe feature-importance result and optional Plotly views."""

    model_id: str
    source: Literal["training", "external", "cross_validation"]
    result: dict[str, Any]
    summary: list[FeatureImportanceSummaryRecord]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    visualizations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExperimentFailureConfigRequest(APIRequest):
    """HTTP configuration for the independent experiment-success classifier."""

    failure_model_config: ModelConfigSchema | None = None
    failure_fit_config: FitConfigSchema | None = None
    min_success_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    eta: float = Field(default=0.05, gt=0.0)
    reduce_q: Literal["prod", "min", "mean"] = "prod"


class TabularFitModelRequest(APIRequest):
    """Fit a :class:`TabularBayesianOptimizer` from JSON tabular data."""

    data: TabularPayload
    bo_model_config: ModelConfigSchema = Field(alias="model_config")
    fit_config: FitConfigSchema | None = None
    multi_output_config: MultiOutputConfigSchema | None = None
    alpha: float | None = Field(default=None, gt=0.0)
    input_cols: list[str]
    target_cols: list[str] | str
    categorical_cols: list[str] = Field(default_factory=list)
    target_categorical_cols: list[str] | None = None
    bounds: Any | None = None
    dtype: str = "float64"
    device: str | None = None
    dropna: bool = True
    missing_strategy: str | None = None
    target_missing_strategy: Literal["drop", "keep"] = "drop"
    experiment_status_col: str | None = None
    experiment_failure: ExperimentFailureConfigRequest | None = None
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
    cross_validation: bool = False
    cv_config: CrossValidationRequest | None = None

    @model_validator(mode="after")
    def attach_alpha_to_tabular_model(self):
        """Store the JSON-safe alpha until the tabular layer builds a likelihood."""

        if self.alpha is None:
            return self
        model_kwargs = dict(self.bo_model_config.model_kwargs)
        if "likelihood" in model_kwargs:
            raise ValueError(
                "Specify either alpha or model_config.model_kwargs.likelihood, not both."
            )
        model_kwargs["_tabular_noise_alpha"] = float(self.alpha)
        self.bo_model_config.model_kwargs = model_kwargs
        return self

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


class TabularTellRequest(APIRequest):
    """Append records containing fitted feature and target columns."""

    data: TabularPayload
    refit: bool = True
    fit_config: FitConfigSchema | None = None


class TabularPredictRequest(APIRequest):
    """Predict from records containing the fitted feature columns."""

    data: TabularPayload
    return_type: Literal[
        "dataframe", "posterior", "mean", "variance", "mean_variance"
    ] = "dataframe"
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
    cross_validation: dict[str, Any] | None = None


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
