"""Factory functions used by the high-level BayesianOptimizer API."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .configs import (
    AcquisitionConfig,
    DataContext,
    ModelBundle,
    ObjectiveConfig,
    OptimizeConfig,
)
from .support.callables import _filter_kwargs_for_callable


def _looks_like_ehvi(config: AcquisitionConfig) -> bool:
    name = config.name.lower()
    cls_name = ""
    if config.acqf_cls is not None:
        cls_name = getattr(config.acqf_cls, "__name__", "").lower()
    return "ehvi" in f"{name} {cls_name}" and "nehvi" not in f"{name} {cls_name}"


def _make_fast_nondominated_partitioning(ref_point: Any, Y: Any) -> Any:
    from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning

    return FastNondominatedPartitioning(ref_point=ref_point, Y=Y)


def _make_chebyshev_objective(weights: Any, Y: Any, alpha: float) -> Any:
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

    scalarization = get_chebyshev_scalarization(weights=weights, Y=Y, alpha=alpha)
    return GenericMCObjective(lambda samples, X=None: scalarization(samples))


def _has_configured_objective(acq_config: AcquisitionConfig | None) -> bool:
    return bool(
        acq_config is not None
        and (
            acq_config.objective is not None
            or acq_config.objective_factory is not None
            or acq_config.objective_config is not None
        )
    )


def prepare_multi_objective_context(bundle: ModelBundle, data_context: DataContext, acq_config: AcquisitionConfig | None = None) -> DataContext:
    mo_config = data_context.multi_objective
    if mo_config is None:
        return data_context
    if data_context.Y_baseline is None:
        data_context.Y_baseline = mo_config.Y_baseline
    if data_context.Y_baseline is None:
        data_context.Y_baseline = bundle.train_Y
    if data_context.ref_point is None:
        data_context.ref_point = mo_config.ref_point
    if data_context.partitioning is None:
        data_context.partitioning = mo_config.partitioning
    if data_context.objective_thresholds is None:
        data_context.objective_thresholds = mo_config.objective_thresholds
    if data_context.constraints is None:
        data_context.constraints = mo_config.constraints
    if acq_config is not None and not _has_configured_objective(acq_config) and mo_config.objective is not None:
        acq_config.objective = mo_config.objective
    if acq_config is not None and not _has_configured_objective(acq_config) and mo_config.auto_scalarization and mo_config.scalarization_weights is not None and data_context.Y_baseline is not None:
        acq_config.objective = _make_chebyshev_objective(weights=mo_config.scalarization_weights, Y=data_context.Y_baseline, alpha=mo_config.scalarization_alpha)
    if acq_config is not None and mo_config.auto_partitioning and data_context.partitioning is None and data_context.ref_point is not None and data_context.Y_baseline is not None and _looks_like_ehvi(acq_config):
        data_context.partitioning = _make_fast_nondominated_partitioning(ref_point=data_context.ref_point, Y=data_context.Y_baseline)
    return data_context


def _objective_mode(config: ObjectiveConfig) -> str:
    if config.mode == "auto":
        if config.outputs is not None or config.specs is not None:
            return "multi_output"
        return "scalar"
    return str(config.mode)


def _direction_to_sign(direction: Any) -> float:
    if isinstance(direction, str):
        if direction == "maximize":
            return 1.0
        if direction == "minimize":
            return -1.0
        raise ValueError("direction must be 'maximize' or 'minimize'.")
    if isinstance(direction, bool):
        return 1.0 if direction else -1.0
    sign = float(direction)
    if sign == 0.0:
        raise ValueError("direction / sign must be non-zero.")
    return 1.0 if sign > 0 else -1.0


def _output_to_index(model: Any, output: Any | None, *, default: int = 0) -> int:
    if output is None:
        return int(default)
    if isinstance(output, str):
        from bochan.acquisition.objective import resolve_hybrid_output_indices

        return resolve_hybrid_output_indices(model, [output])[0]
    return int(output)


def _infer_ordinal_likelihood(model: Any, output: Any | None = None) -> Any:
    if output is not None and hasattr(model, "models"):
        idx = _output_to_index(model, output)
        submodel = model.models[idx]
        lik = getattr(submodel, "ordinal_likelihood", None)
        if lik is None:
            lik = getattr(submodel, "likelihood", None)
        if lik is not None:
            return lik
    lik = getattr(model, "ordinal_likelihood", None)
    if lik is None:
        lik = getattr(model, "likelihood", None)
    if lik is not None:
        return lik
    raise ValueError("Could not infer ordinal_likelihood from model. Pass ObjectiveConfig.ordinal_likelihood explicitly.")


def _infer_ordinal_utility_values(model: Any, likelihood: Any | None = None) -> Any:
    import torch

    for obj in (likelihood, model):
        if obj is None:
            continue
        num_classes = getattr(obj, "num_classes", None)
        if num_classes is not None:
            return torch.arange(int(num_classes), dtype=torch.double)
        for name in ("cutpoints", "thresholds", "cuts", "boundaries", "_cutpoints"):
            if hasattr(obj, name):
                value = getattr(obj, name)
                if callable(value):
                    value = value()
                if torch.is_tensor(value):
                    return torch.arange(int(value.numel()) + 1, dtype=value.dtype, device=value.device)
    raise ValueError("Could not infer ordinal utility_values. Pass ObjectiveConfig.utility_values explicitly.")


def _common_objective_kwargs(config: ObjectiveConfig) -> dict[str, Any]:
    kwargs = {
        "n_w": config.n_w,
        "risk_type": config.risk_type,
        "alpha": config.alpha,
        "maximize": config.maximize,
        "aggregate_mean_when_no_risk": config.aggregate_mean_when_no_risk,
        "allow_unexpanded": config.allow_unexpanded,
    }
    kwargs.update(config.objective_kwargs)
    return kwargs


def _build_regression_objective(bundle: ModelBundle, config: ObjectiveConfig) -> Any | None:
    mode = _objective_mode(config)
    if mode == "none":
        return None

    if mode == "multi_output":
        from bochan.acquisition.objective import make_hybrid_multi_output_objective

        outputs = config.outputs
        if outputs is None and config.output is not None:
            outputs = [config.output]
        kwargs = _common_objective_kwargs(config)
        kwargs.update(
            {
                "specs": config.specs,
                "outputs": outputs,
                "directions": config.directions,
                "weights": config.weights,
                "eq_targets": config.eq_targets,
            }
        )
        kwargs = _filter_kwargs_for_callable(make_hybrid_multi_output_objective, kwargs)
        return make_hybrid_multi_output_objective(model=bundle.model, **kwargs)

    if mode != "scalar":
        raise ValueError(f"Unsupported regression objective mode: {config.mode!r}.")

    from bochan.acquisition.objective import RegressionScalarObjective

    output_index = _output_to_index(bundle.model, config.output, default=0)
    kwargs = _common_objective_kwargs(config)
    kwargs.update(
        {
            "output_index": output_index,
            "weight": config.weight,
            "sign": _direction_to_sign(config.direction),
            "eq_target": config.eq_target,
        }
    )
    kwargs = _filter_kwargs_for_callable(RegressionScalarObjective, kwargs)
    return RegressionScalarObjective(**kwargs)


def _build_binary_objective(bundle: ModelBundle, config: ObjectiveConfig) -> Any | None:
    mode = _objective_mode(config)
    if mode == "none":
        return None

    if mode == "multi_output":
        from bochan.acquisition.objective import MultiOutputBinaryClassificationInputPerturbationObjective

        kwargs = _common_objective_kwargs(config)
        kwargs = _filter_kwargs_for_callable(MultiOutputBinaryClassificationInputPerturbationObjective, kwargs)
        return MultiOutputBinaryClassificationInputPerturbationObjective(**kwargs)

    if mode != "scalar":
        raise ValueError(f"Unsupported binary objective mode: {config.mode!r}.")

    from bochan.acquisition.objective import BinaryClassificationScoreObjective

    kwargs = {
        "n_w": config.n_w,
        "risk_type": config.risk_type,
        "alpha": config.alpha,
        "maximize": config.maximize,
    }
    kwargs.update(config.objective_kwargs)
    kwargs = _filter_kwargs_for_callable(BinaryClassificationScoreObjective, kwargs)
    return BinaryClassificationScoreObjective(**kwargs)


def _build_ordinal_objective(bundle: ModelBundle, config: ObjectiveConfig) -> Any | None:
    mode = _objective_mode(config)
    if mode == "none":
        return None

    if mode == "multi_output":
        from bochan.acquisition.objective import MultiOutputOrdinalInputPerturbationObjective

        utility_values = config.utility_values
        if utility_values is None:
            utility_values = _infer_ordinal_utility_values(bundle.model)
        kwargs = _common_objective_kwargs(config)
        kwargs.update(
            {
                "model": bundle.model,
                "utility_values": utility_values,
            }
        )
        kwargs = _filter_kwargs_for_callable(MultiOutputOrdinalInputPerturbationObjective, kwargs)
        return MultiOutputOrdinalInputPerturbationObjective(**kwargs)

    if mode != "scalar":
        raise ValueError(f"Unsupported ordinal objective mode: {config.mode!r}.")

    from bochan.acquisition.objective import OrdinalInputPerturbationExpectedUtilityObjective

    ordinal_likelihood = config.ordinal_likelihood or _infer_ordinal_likelihood(bundle.model, config.output)
    utility_values = config.utility_values
    if utility_values is None:
        utility_values = _infer_ordinal_utility_values(bundle.model, ordinal_likelihood)
    kwargs = _common_objective_kwargs(config)
    kwargs.update(
        {
            "ordinal_likelihood": ordinal_likelihood,
            "utility_values": utility_values,
        }
    )
    kwargs = _filter_kwargs_for_callable(OrdinalInputPerturbationExpectedUtilityObjective, kwargs)
    return OrdinalInputPerturbationExpectedUtilityObjective(**kwargs)


def _build_hybrid_objective(bundle: ModelBundle, config: ObjectiveConfig) -> Any | None:
    mode = _objective_mode(config)
    if mode == "none":
        return None

    if mode == "multi_output":
        from bochan.acquisition.objective import make_hybrid_multi_output_objective

        outputs = config.outputs
        if outputs is None and config.output is not None:
            outputs = [config.output]
        kwargs = _common_objective_kwargs(config)
        kwargs.update(
            {
                "specs": config.specs,
                "outputs": outputs,
                "directions": config.directions,
                "weights": config.weights,
                "eq_targets": config.eq_targets,
            }
        )
        kwargs = _filter_kwargs_for_callable(make_hybrid_multi_output_objective, kwargs)
        return make_hybrid_multi_output_objective(model=bundle.model, **kwargs)

    if mode != "scalar":
        raise ValueError(f"Unsupported hybrid objective mode: {config.mode!r}.")

    from bochan.acquisition.objective import make_hybrid_scalar_objective

    output = config.output if config.output is not None else 0
    kwargs = _common_objective_kwargs(config)
    kwargs.update(
        {
            "output": output,
            "direction": config.direction,
            "weight": config.weight,
            "eq_target": config.eq_target,
        }
    )
    kwargs = _filter_kwargs_for_callable(make_hybrid_scalar_objective, kwargs)
    return make_hybrid_scalar_objective(model=bundle.model, **kwargs)


def build_objective(bundle: ModelBundle, config: AcquisitionConfig, data_context: DataContext | None = None) -> Any | None:
    """AcquisitionConfig から objective を構築する。

    優先順位:
        1. config.objective をそのまま使う。
        2. config.objective_factory で高度に上書きする。
        3. config.objective_config から task_type に応じて自動生成する。
        4. objective なし。
    """
    if config.objective is not None:
        return config.objective

    if config.objective_factory is not None:
        kwargs = {
            "model": bundle.model,
            "bundle": bundle,
            "data_context": data_context,
        }
        kwargs.update(config.objective_kwargs)
        kwargs = _filter_kwargs_for_callable(config.objective_factory, kwargs)
        return config.objective_factory(**kwargs)

    objective_config = config.objective_config
    if objective_config is None:
        return None

    task_type = str(bundle.task_type)
    if task_type in {"regression", "multi_objective"}:
        return _build_regression_objective(bundle, objective_config)
    if task_type == "binary":
        return _build_binary_objective(bundle, objective_config)
    if task_type == "ordinal":
        return _build_ordinal_objective(bundle, objective_config)
    if task_type == "hybrid":
        return _build_hybrid_objective(bundle, objective_config)

    raise NotImplementedError(
        "ObjectiveConfig automatic objective generation is not implemented for "
        f"task_type={task_type!r}. Pass AcquisitionConfig.objective or objective_factory explicitly."
    )


def build_acquisition(bundle: ModelBundle, config: AcquisitionConfig, data_context: DataContext | None = None) -> Any:
    data_context = data_context or DataContext()
    data_context = prepare_multi_objective_context(bundle, data_context, config)
    if config.acqf_factory is not None:
        return config.acqf_factory(bundle=bundle, config=config, data_context=data_context)
    if config.acqf_cls is None:
        raise ValueError("acqf_cls is None. Provide AcquisitionConfig.acqf_cls or acqf_factory.")
    kwargs = {"model": bundle.model}
    kwargs.update(config.acqf_kwargs)
    objective = build_objective(bundle=bundle, config=config, data_context=data_context)
    if objective is not None:
        kwargs["objective"] = objective
    if config.sampler is not None:
        kwargs["sampler"] = config.sampler
    for field_name in config.context_fields:
        value = getattr(data_context, field_name, None)
        if value is not None:
            kwargs[field_name] = value
    for key, value in data_context.extra.items():
        if value is not None:
            kwargs[key] = value
    if config.filter_kwargs_by_signature:
        kwargs = _filter_kwargs_for_callable(config.acqf_cls, kwargs)
    return config.acqf_cls(**kwargs)


def _build_post_processing_func(config: OptimizeConfig, bounds: Any) -> Callable[..., Any] | None:
    """Resolve explicit or config-driven candidate repair post-processing."""
    if config.post_processing_func is not None:
        return config.post_processing_func
    repair = config.repair_config
    if repair is None:
        return None

    from bochan.constraints.postprocess import make_grid_k_sparse_post_processing_func

    repair_bounds = repair.bounds if repair.bounds is not None else bounds
    if repair_bounds is None:
        raise ValueError("bounds is required when OptimizeConfig.repair_config is specified.")

    equality_constraints = repair.equality_constraints
    if equality_constraints is None:
        equality_constraints = config.equality_constraints

    inequality_constraints = repair.inequality_constraints
    if inequality_constraints is None:
        inequality_constraints = config.inequality_constraints

    fixed_features = repair.fixed_features
    if fixed_features is None:
        fixed_features = config.fixed_features

    grid_base = repair_bounds[0]
    if repair.bounds is None:
        grid_base = repair_bounds[0].new_zeros(repair_bounds[0].shape)

    return make_grid_k_sparse_post_processing_func(
        bounds=repair_bounds,
        numeric_indices=repair.numeric_indices,
        steps=repair.steps,
        comp_idx=repair.comp_idx,
        k=repair.k,
        grid_base=grid_base,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        inequality_sense=repair.inequality_sense,
        fixed_features=fixed_features,
        final_sum_constraint=repair.final_sum_constraint,
        diversify=repair.diversify,
        diversify_kwargs=repair.diversify_kwargs,
        score=repair.score,
        support_selection=repair.support_selection,
        sample_tau=repair.sample_tau,
        sample_eps=repair.sample_eps,
        generator=repair.generator,
        max_iters=repair.max_iters,
        num_alternations=repair.num_alternations,
        final_priority=repair.final_priority,
        support_eps=repair.support_eps,
    )


def _optimizer_name(optimizer: str) -> str:
    return optimizer.replace("-", "_").lower()


def _with_sequential(common_kwargs: dict[str, Any], config: OptimizeConfig) -> dict[str, Any]:
    kwargs = dict(common_kwargs)
    kwargs["sequential"] = config.sequential
    return kwargs


def _merge_fixed_features(base: Mapping[int, float] | None, extra: Mapping[int, float] | None) -> dict[int, float]:
    """Merge fixed-feature dictionaries with ``extra`` taking priority."""
    merged = {int(k): float(v) for k, v in (base or {}).items()}
    for key, value in (extra or {}).items():
        merged[int(key)] = float(value)
    return merged


def _merge_fixed_features_list(
    fixed_features: Mapping[int, float] | None,
    fixed_features_list: Sequence[Mapping[int, float]] | None,
) -> list[dict[int, float]] | None:
    """Apply global fixed features to every mixed fixed-feature assignment."""
    base = _merge_fixed_features(fixed_features, None)
    if fixed_features_list is None:
        return [base] if base else None
    if len(fixed_features_list) == 0:
        raise ValueError("fixed_features_list must not be empty when supplied.")
    return [_merge_fixed_features(base, item) for item in fixed_features_list]


def optimize_candidates(acqf: Any, bounds: Any, config: OptimizeConfig) -> tuple[Any, Any]:
    if bounds is None:
        raise ValueError("bounds must be provided.")

    repair = config.repair_config
    if repair is not None and repair.support_selection == "best_subset":
        from .support.best_subset import optimize_best_subset_candidates

        return optimize_best_subset_candidates(
            acqf=acqf,
            bounds=bounds,
            config=config,
            optimize_one=optimize_candidates,
        )

    common_kwargs = {
        "acq_function": acqf,
        "bounds": bounds,
        "q": config.q,
        "num_restarts": config.num_restarts,
        "raw_samples": config.raw_samples,
        "return_best_only": config.return_best_only,
    }

    from .support.one_shot import resolve_one_shot_ic_generator

    one_shot_ic_generator = resolve_one_shot_ic_generator(acqf)
    if one_shot_ic_generator is not None and "ic_generator" not in config.optimizer_kwargs:
        common_kwargs["ic_generator"] = one_shot_ic_generator

    post_processing_func = _build_post_processing_func(config, bounds)
    if post_processing_func is not None:
        common_kwargs["post_processing_func"] = post_processing_func

    if config.fixed_features is not None:
        common_kwargs["fixed_features"] = config.fixed_features
    if config.inequality_constraints is not None:
        common_kwargs["inequality_constraints"] = config.inequality_constraints
    if config.equality_constraints is not None:
        common_kwargs["equality_constraints"] = config.equality_constraints
    common_kwargs.update(config.optimizer_kwargs)

    optimizer = config.optimizer
    if callable(optimizer) and not isinstance(optimizer, str):
        kwargs = _with_sequential(common_kwargs, config)
        kwargs = _filter_kwargs_for_callable(optimizer, kwargs)
        return optimizer(**kwargs)

    optimizer_name = _optimizer_name(str(optimizer))

    if optimizer_name == "optimize_acqf":
        from botorch.optim import optimize_acqf

        kwargs = _with_sequential(common_kwargs, config)
        kwargs = _filter_kwargs_for_callable(optimize_acqf, kwargs)
        return optimize_acqf(**kwargs)

    if optimizer_name == "optimize_acqf_mixed":
        from botorch.optim import optimize_acqf_mixed

        kwargs = _with_sequential(common_kwargs, config)
        merged_fixed_features_list = _merge_fixed_features_list(config.fixed_features, config.fixed_features_list)
        if merged_fixed_features_list is None:
            raise ValueError(
                "OptimizeConfig.fixed_features_list or OptimizeConfig.fixed_features "
                "is required when optimizer='optimize_acqf_mixed'."
            )
        kwargs.pop("fixed_features", None)
        kwargs["fixed_features_list"] = merged_fixed_features_list
        kwargs = _filter_kwargs_for_callable(optimize_acqf_mixed, kwargs)
        return optimize_acqf_mixed(**kwargs)

    if optimizer_name in {"evo", "optimize_acqf_evo"}:
        from bochan.optim import optimize_acqf_evo

        kwargs = _with_sequential(common_kwargs, config)
        kwargs = _filter_kwargs_for_callable(optimize_acqf_evo, kwargs)
        return optimize_acqf_evo(**kwargs)

    if optimizer_name in {"torch", "optimize_acqf_torch"}:
        from bochan.optim import optimize_acqf_torch

        kwargs = _with_sequential(common_kwargs, config)
        kwargs = _filter_kwargs_for_callable(optimize_acqf_torch, kwargs)
        return optimize_acqf_torch(**kwargs)

    if optimizer_name in {"evo_mixed", "optimize_acqf_evo_mixed"}:
        from bochan.optim import optimize_acqf_evo_mixed

        kwargs = _with_sequential(common_kwargs, config)
        if config.fixed_features_list is not None:
            kwargs["fixed_features_list"] = config.fixed_features_list
        kwargs = _filter_kwargs_for_callable(optimize_acqf_evo_mixed, kwargs)
        return optimize_acqf_evo_mixed(**kwargs)

    if optimizer_name in {"torch_mixed", "optimize_acqf_torch_mixed"}:
        from bochan.optim import optimize_acqf_torch_mixed

        kwargs = _with_sequential(common_kwargs, config)
        if config.fixed_features_list is not None:
            kwargs["fixed_features_list"] = config.fixed_features_list
        kwargs = _filter_kwargs_for_callable(optimize_acqf_torch_mixed, kwargs)
        return optimize_acqf_torch_mixed(**kwargs)

    raise ValueError(f"Unknown optimizer: {optimizer}")
