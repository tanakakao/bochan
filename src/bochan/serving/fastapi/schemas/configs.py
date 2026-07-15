"""Serializable Pydantic schemas mirroring bochan.api dataclasses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Schema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class InputTransformConfigSchema(_Schema):
    normalize: bool = True
    perturbation: bool = False
    n_w: int = 16
    std: float = 0.1
    bounds: Any | None = None
    categorical_idx: list[int] | None = None


class FitConfigSchema(_Schema):
    method: str = "auto"
    num_epochs: int | None = None
    lr: float | None = None
    batch_size: int | None = None
    shuffle: bool = True
    verbose: bool = False
    clip_grad_norm: float | None = None
    maxiter: int | None = None
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    fit_kwargs: dict[str, Any] = Field(default_factory=dict)
    mll_kwargs: dict[str, Any] = Field(default_factory=dict)
    skip_fit: bool = False
    fit_func: Any | None = None
    mll_factory: Any | None = None
    mll_cls: Any | None = None
    use_model_make_mll: bool = True
    beta: float | None = None


class OutputConfigSchema(_Schema):
    task_type: str
    model_type: str = "base"
    name: str | None = None
    input_type: Literal["normal", "mixed"] | None = None
    cat_dims: list[int] | None = None
    input_transform_config: InputTransformConfigSchema | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    fit_config: FitConfigSchema | None = None
    output_spec_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Tabular-only target metadata. The FastAPI converter removes these fields
    # before constructing the tensor-oriented core OutputConfig and retains the
    # resolved mapping for string class / ordinal-rank constraints.
    ordered_categories: list[Any] | None = None
    categories: list[Any] | None = None
    category_map: dict[Any, int] | None = None

    @model_validator(mode="after")
    def validate_category_metadata(self):
        declared = [
            name
            for name in ("ordered_categories", "categories", "category_map")
            if getattr(self, name) is not None
        ]
        if len(declared) > 1:
            raise ValueError(
                "Specify only one of ordered_categories, categories, or category_map."
            )
        if self.ordered_categories is not None and self.task_type.lower() != "ordinal":
            raise ValueError("ordered_categories is only valid for ordinal outputs.")
        return self


class MultiOutputConfigSchema(_Schema):
    output_configs: list[OutputConfigSchema | str | dict[str, Any]] | None = None
    output_fit_configs: list[FitConfigSchema | None] | FitConfigSchema | None = None
    output_task_types: list[str] | None = None
    output_names: list[str] | None = None
    wrapper_kwargs: dict[str, Any] = Field(default_factory=dict)
    use_hybrid: bool | None = None
    fit_submodels: bool = True
    fit_wrapper: bool = False
    output_spec_kwargs: list[dict[str, Any]] | None = None
    train_y_slice_dim: int = -1


class ModelConfigSchema(_Schema):
    task_type: str = "regression"
    model_type: str = "base"
    input_type: Literal["normal", "mixed"] | None = None
    cat_dims: list[int] | None = None
    input_transform_config: InputTransformConfigSchema | None = None
    outcome_transform: bool | Any = True
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    multi_output_config: MultiOutputConfigSchema | None = None
    train_x_name: str = "train_X"
    train_y_name: str = "train_Y"
    pass_train_data: bool = True
    pass_cat_dims: bool | None = None
    pass_input_transform: bool = True
    pass_outcome_transform: bool = True


class ObjectiveConfigSchema(_Schema):
    mode: Literal["auto", "none", "scalar", "multi_output"] = "auto"
    output: Any | None = None
    outputs: list[Any] | None = None
    specs: list[Any] | None = None
    directions: list[Any] | None = None
    weights: list[float] | None = None
    eq_targets: list[float | None] | None = None
    direction: Any = "maximize"
    weight: float = 1.0
    eq_target: float | None = None
    n_w: int | None = None
    risk_type: str | None = None
    alpha: float = 0.5
    maximize: bool = True
    aggregate_mean_when_no_risk: bool = True
    allow_unexpanded: bool = True
    utility_values: Any | None = None
    ordinal_likelihood: Any | None = None
    objective_kwargs: dict[str, Any] = Field(default_factory=dict)


class OutcomeConstraintConfigSchema(_Schema):
    constraints: list[Any] | None = None
    output_indices: list[int] = Field(default_factory=list)
    operators: list[Literal["ge", "gt", "le", "lt"]] = Field(default_factory=list)
    thresholds: list[float] = Field(default_factory=list)
    eta: float = 1e-3
    reduce_constraints: str = "prod"
    reduce_q: str = "mean"
    posterior_mode: str = "objective"
    min_feasibility: float = 0.0
    detach_feasibility: bool = False

    @model_validator(mode="after")
    def validate_parallel_lengths(self):
        lengths = {
            len(self.output_indices),
            len(self.operators),
            len(self.thresholds),
        }
        if len(lengths) != 1:
            raise ValueError(
                "output_indices, operators, and thresholds must have the same length."
            )
        return self


class MultiObjectiveConfigSchema(_Schema):
    ref_point: Any | None = None
    Y_baseline: Any | None = None
    partitioning: Any | None = None
    objective_thresholds: Any | None = None
    constraints: Any | None = None
    objective: Any | None = None
    scalarization_weights: Any | None = None
    scalarization_alpha: float = 0.05
    auto_partitioning: bool = True
    auto_scalarization: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataContextSchema(_Schema):
    bounds: Any | None = None
    X_baseline: Any | None = None
    X_pending: Any | None = None
    Y_baseline: Any | None = None
    best_f: Any | None = None
    ref_point: Any | None = None
    partitioning: Any | None = None
    objective_thresholds: Any | None = None
    mc_points: Any | None = None
    constraints: Any | None = None
    multi_objective: MultiObjectiveConfigSchema | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AcquisitionConfigSchema(_Schema):
    name: str
    acqf_cls: Any | None = None
    acqf_factory: Any | None = None
    objective: Any | None = None
    objective_config: ObjectiveConfigSchema | None = None
    objective_factory: Any | None = None
    objective_kwargs: dict[str, Any] = Field(default_factory=dict)
    constraints: Any | None = None
    outcome_constraint_config: OutcomeConstraintConfigSchema | None = None
    sampler: Any | None = None
    acqf_kwargs: dict[str, Any] = Field(default_factory=dict)
    context_fields: tuple[str, ...] = (
        "X_baseline",
        "X_pending",
        "Y_baseline",
        "best_f",
        "ref_point",
        "partitioning",
        "objective_thresholds",
        "mc_points",
        "constraints",
    )
    filter_kwargs_by_signature: bool = True

    @model_validator(mode="after")
    def validate_constraint_source(self):
        if self.constraints is not None and self.outcome_constraint_config is not None:
            raise ValueError(
                "Specify either constraints or outcome_constraint_config, not both."
            )
        return self


class CandidateRepairConfigSchema(_Schema):
    bounds: Any | None = None
    numeric_indices: list[int] | None = None
    steps: Any | None = None
    comp_idx: list[int] | None = None
    k: int = 0
    equality_constraints: Any | None = None
    inequality_constraints: Any | None = None
    inequality_sense: Literal["le", "ge"] = "le"
    fixed_features: dict[int, float] | None = None
    final_sum_constraint: tuple[list[int], float] | None = None
    diversify: bool = False
    diversify_kwargs: dict[str, Any] | None = None
    score: Literal["abs", "value"] = "abs"
    support_selection: Literal["topk", "sample"] = "topk"
    sample_tau: float = 0.2
    sample_eps: float = 0.05
    generator: Any | None = None
    max_iters: int = 12
    num_alternations: int = 2
    final_priority: Literal["grid", "constraints"] = "grid"
    support_eps: float = 0.0


class OptimizeConfigSchema(_Schema):
    q: int = 1
    num_restarts: int = 10
    raw_samples: int = 256
    sequential: bool = False
    optimizer: Any = "optimize_acqf"
    evo_method: Literal["ga", "pso", "sa", "cmaes"] = "ga"
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    post_processing_func: Any | None = None
    repair_config: CandidateRepairConfigSchema | None = None
    fixed_features: dict[int, float] | None = None
    fixed_features_list: list[dict[int, float]] | None = None
    inequality_constraints: Any | None = None
    equality_constraints: Any | None = None
    return_best_only: bool = True
