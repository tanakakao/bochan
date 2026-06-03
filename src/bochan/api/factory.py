"""Factory functions used by the high-level BayesianOptimizer API."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Callable

from .configs import (
    AcquisitionConfig,
    CandidateRepairConfig,
    DataContext,
    FitConfig,
    InputType,
    ModelBundle,
    ModelConfig,
    MultiOutputConfig,
    ObjectiveConfig,
    OptimizeConfig,
    OutputConfig,
)


def infer_input_type(cat_dims: list[int] | tuple[int, ...] | None) -> InputType:
    """cat_dims の有無から normal / mixed を推定する。"""
    return "mixed" if cat_dims else "normal"


def _as_cat_dims(cat_dims: Any) -> list[int]:
    if cat_dims is None:
        return []
    return list(cat_dims)


def resolve_model_cls(
    config: ModelConfig,
    model_registry: Mapping[Any, Any] | None = None,
) -> type | Callable[..., Any]:
    """ModelConfig からモデルクラスを解決する。

    `model_registry` を省略した場合は、bochan API 標準の
    `DEFAULT_MODEL_REGISTRY` を使います。
    """
    if config.model_cls is not None:
        return config.model_cls

    if model_registry is None:
        from .model_registry import DEFAULT_MODEL_REGISTRY

        model_registry = DEFAULT_MODEL_REGISTRY

    cat_dims = _as_cat_dims(config.cat_dims)
    input_type = config.input_type or infer_input_type(cat_dims)
    flat_key = (input_type, config.task_type, config.model_type)
    if flat_key in model_registry:
        return model_registry[flat_key]

    try:
        return model_registry[input_type][config.task_type][config.model_type]
    except Exception as exc:
        raise ValueError(
            "Unknown model setting: "
            f"input_type={input_type}, task_type={config.task_type}, "
            f"model_type={config.model_type}"
        ) from exc


def _make_default_bounds(train_X: Any) -> Any:
    if train_X is None:
        raise ValueError("train_X is required to build bounds automatically.")
    try:
        import torch

        if isinstance(train_X, torch.Tensor):
            return torch.cat(
                [
                    train_X.min(dim=0, keepdim=True).values,
                    train_X.max(dim=0, keepdim=True).values,
                ],
                dim=0,
            )
    except Exception:
        pass
    raise TypeError("Automatic bounds generation currently supports torch.Tensor train_X.")


def _build_input_transform_from_config(
    train_X: Any,
    config: ModelConfig,
    cat_dims: list[int],
) -> Any:
    if config.input_transform is not None:
        return config.input_transform
    tf_config = config.input_transform_config
    if tf_config is None:
        return None

    from bochan.models.transforms.input import build_input_transform

    bounds = tf_config.bounds if tf_config.bounds is not None else _make_default_bounds(train_X)
    categorical_idx = tf_config.categorical_idx
    if categorical_idx is None:
        categorical_idx = cat_dims or None

    return build_input_transform(
        train_X=train_X,
        bounds=bounds,
        perturbation=tf_config.perturbation,
        categorical_idx=None if categorical_idx is None else list(categorical_idx),
        n_w=tf_config.n_w,
        std=tf_config.std,
    )


def _build_model_kwargs(train_X: Any, train_Y: Any, config: ModelConfig, cat_dims: list[int]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.pass_train_data:
        kwargs[config.train_x_name] = train_X
        kwargs[config.train_y_name] = train_Y

    pass_cat_dims = config.pass_cat_dims
    if pass_cat_dims is None:
        pass_cat_dims = bool(cat_dims)
    if pass_cat_dims:
        kwargs["cat_dims"] = cat_dims

    input_transform = _build_input_transform_from_config(train_X, config, cat_dims)
    if config.pass_input_transform and input_transform is not None:
        kwargs["input_transform"] = input_transform

    if config.pass_outcome_transform and config.outcome_transform is not None:
        kwargs["outcome_transform"] = config.outcome_transform

    kwargs.update(config.model_kwargs)
    return kwargs


def _infer_num_outputs(train_Y: Any) -> int:
    if train_Y is None:
        raise ValueError("train_Y is required to infer the number of outputs.")
    if not hasattr(train_Y, "shape"):
        raise TypeError("train_Y must have a shape attribute for multi-output construction.")
    if len(train_Y.shape) == 1:
        return 1
    return int(train_Y.shape[-1])


def _slice_output_y(train_Y: Any, output_index: int, dim: int = -1) -> Any:
    if len(train_Y.shape) == 1:
        if output_index != 0:
            raise IndexError("1D train_Y has only one output.")
        return train_Y.reshape(-1, 1) if hasattr(train_Y, "reshape") else train_Y

    ndim = len(train_Y.shape)
    dim = dim if dim >= 0 else ndim + dim
    index = [slice(None)] * ndim
    index[dim] = slice(output_index, output_index + 1)
    return train_Y[tuple(index)]


def _normalize_output_config_like(
    raw: Any,
    parent_config: ModelConfig,
    index: int,
) -> tuple[ModelConfig, str | None, dict[str, Any], FitConfig | None]:
    """文字列・辞書・OutputConfig・ModelConfig を ModelConfig へ正規化する。"""
    if isinstance(raw, ModelConfig):
        return raw, None, {}, None

    if isinstance(raw, str):
        oc = OutputConfig(task_type=raw)
    elif isinstance(raw, OutputConfig):
        oc = raw
    elif isinstance(raw, Mapping):
        oc = OutputConfig(**dict(raw))
    else:
        raise TypeError(
            "output_configs entries must be str, dict, OutputConfig, or ModelConfig. "
            f"Got {type(raw).__name__} at index {index}."
        )

    merged_model_kwargs = dict(parent_config.model_kwargs)
    merged_model_kwargs.update(oc.model_kwargs)

    cfg = replace(
        parent_config,
        task_type=oc.task_type,
        model_type=oc.model_type,
        input_type=oc.input_type if oc.input_type is not None else parent_config.input_type,
        cat_dims=oc.cat_dims if oc.cat_dims is not None else parent_config.cat_dims,
        input_transform_config=oc.input_transform_config
        if oc.input_transform_config is not None
        else parent_config.input_transform_config,
        model_kwargs=merged_model_kwargs,
        multi_output_config=None,
    )
    return cfg, oc.name, dict(oc.output_spec_kwargs), oc.fit_config


def _resolve_output_configs(parent_config: ModelConfig, n_outputs: int) -> tuple[list[ModelConfig], list[str | None], list[dict[str, Any]], list[FitConfig | None]]:
    mo_config = parent_config.multi_output_config
    if mo_config is None:
        raise RuntimeError("multi_output_config is required.")

    names: list[str | None] = [None for _ in range(n_outputs)]
    spec_kwargs: list[dict[str, Any]] = [{} for _ in range(n_outputs)]
    embedded_fit_configs: list[FitConfig | None] = [None for _ in range(n_outputs)]

    if mo_config.output_configs is not None:
        if len(mo_config.output_configs) != n_outputs:
            raise ValueError(f"Expected {n_outputs} output_configs, got {len(mo_config.output_configs)}.")
        configs: list[ModelConfig] = []
        for i, raw in enumerate(mo_config.output_configs):
            cfg, name, kwargs, fit_config = _normalize_output_config_like(raw, parent_config, i)
            configs.append(cfg)
            names[i] = name
            spec_kwargs[i] = kwargs
            embedded_fit_configs[i] = fit_config
        return configs, names, spec_kwargs, embedded_fit_configs

    output_task_types = list(mo_config.output_task_types or [])
    if output_task_types and len(output_task_types) != n_outputs:
        raise ValueError(f"Expected {n_outputs} output_task_types, got {len(output_task_types)}.")

    configs = []
    for i in range(n_outputs):
        task_type = output_task_types[i] if output_task_types else parent_config.task_type
        configs.append(replace(parent_config, task_type=task_type, multi_output_config=None))
    return configs, names, spec_kwargs, embedded_fit_configs


def _resolve_output_fit_configs(
    parent_fit_config: FitConfig | None,
    mo_config: MultiOutputConfig,
    n_outputs: int,
    embedded: Sequence[FitConfig | None] | None = None,
) -> list[FitConfig | None]:
    embedded = list(embedded or [None for _ in range(n_outputs)])
    output_fit_configs = mo_config.output_fit_configs
    if output_fit_configs is None:
        return [embedded[i] or parent_fit_config for i in range(n_outputs)]
    if isinstance(output_fit_configs, FitConfig):
        return [embedded[i] or output_fit_configs for i in range(n_outputs)]
    if len(output_fit_configs) != n_outputs:
        raise ValueError(f"Expected {n_outputs} output_fit_configs, got {len(output_fit_configs)}.")
    return [embedded[i] or output_fit_configs[i] for i in range(n_outputs)]


def _build_single_model(
    train_X: Any,
    train_Y: Any,
    config: ModelConfig,
    *,
    model_registry: Mapping[Any, Any] | None = None,
) -> ModelBundle:
    cat_dims = _as_cat_dims(config.cat_dims)
    input_type = config.input_type or infer_input_type(cat_dims)
    kwargs = _build_model_kwargs(train_X, train_Y, config, cat_dims)

    if config.model_factory is not None:
        model = config.model_factory(**kwargs)
        model_cls_name = getattr(config.model_factory, "__name__", str(config.model_factory))
    else:
        model_cls = resolve_model_cls(config, model_registry=model_registry)
        model = model_cls(**kwargs)
        model_cls_name = getattr(model_cls, "__name__", str(model_cls))

    bundle_train_X = train_X
    bundle_train_Y = train_Y
    if bundle_train_X is None:
        bundle_train_X = getattr(model, "train_X", None)
        if bundle_train_X is None and hasattr(model, "train_inputs"):
            train_inputs = getattr(model, "train_inputs")
            bundle_train_X = train_inputs[0] if isinstance(train_inputs, tuple) else train_inputs
    if bundle_train_Y is None:
        bundle_train_Y = getattr(model, "train_Y", None)
        if bundle_train_Y is None:
            bundle_train_Y = getattr(model, "train_targets", None)

    return ModelBundle(
        model=model,
        train_X=bundle_train_X,
        train_Y=bundle_train_Y,
        model_config=config,
        input_type=input_type,
        task_type=str(config.task_type),
        model_type=str(config.model_type),
        cat_dims=cat_dims,
        metadata={"model_cls": model_cls_name, "model_factory": config.model_factory is not None},
    )


def _build_wrapper_from_submodels(
    submodels: Sequence[Any],
    output_configs: Sequence[ModelConfig],
    mo_config: MultiOutputConfig,
    output_names: Sequence[str | None] | None = None,
    output_spec_kwargs: Sequence[dict[str, Any]] | None = None,
) -> Any:
    if mo_config.wrapper_factory is not None:
        return mo_config.wrapper_factory(submodels=submodels, output_configs=output_configs, config=mo_config)
    if mo_config.wrapper_cls is not None:
        return mo_config.wrapper_cls(*submodels, **mo_config.wrapper_kwargs)

    task_types = [str(cfg.task_type) for cfg in output_configs]
    unique_task_types = set(task_types)
    use_hybrid = mo_config.use_hybrid
    if use_hybrid is None:
        use_hybrid = len(unique_task_types) > 1 or "hybrid" in unique_task_types or "multiclass" in unique_task_types

    if use_hybrid:
        from bochan.models.hybrid import HybridMultiOutputModel, OutputSpec

        fallback_names = [f"y{i}" for i in range(len(submodels))]
        names = list(mo_config.output_names or output_names or fallback_names)
        names = [names[i] if names[i] is not None else fallback_names[i] for i in range(len(submodels))]
        if len(names) != len(submodels):
            raise ValueError(f"Expected {len(submodels)} output_names, got {len(names)}.")

        base_kwargs = list(mo_config.output_spec_kwargs or [{} for _ in range(len(submodels))])
        inline_kwargs = list(output_spec_kwargs or [{} for _ in range(len(submodels))])
        if len(base_kwargs) != len(submodels) or len(inline_kwargs) != len(submodels):
            raise ValueError("output_spec_kwargs length must match the number of outputs.")
        merged_spec_kwargs = [{**base_kwargs[i], **inline_kwargs[i]} for i in range(len(submodels))]

        specs = [
            OutputSpec(name=names[i], task_type=task_types[i], model=submodels[i], **merged_spec_kwargs[i])
            for i in range(len(submodels))
        ]
        return HybridMultiOutputModel(specs=specs, **mo_config.wrapper_kwargs)

    task_type = next(iter(unique_task_types))
    if task_type in {"regression", "multi_objective"}:
        from botorch.models.model_list_gp_regression import ModelListGP

        return ModelListGP(*submodels)
    if task_type == "binary":
        from bochan.models.classification.binary.base import MultiOutputBinaryClassificationModel

        return MultiOutputBinaryClassificationModel(*submodels, **mo_config.wrapper_kwargs)
    if task_type == "ordinal":
        from bochan.models.ordinal.base.multioutput import MultiOutputOrdinalModel

        return MultiOutputOrdinalModel(*submodels, **mo_config.wrapper_kwargs)
    raise ValueError(
        f"No dedicated homogeneous multi-output wrapper is available for task_type={task_type!r}. "
        "Set use_hybrid=True or provide wrapper_cls."
    )


def build_multi_output_model(train_X: Any, train_Y: Any, config: ModelConfig, *, model_registry: Mapping[Any, Any] | None = None) -> ModelBundle:
    mo_config = config.multi_output_config
    if mo_config is None:
        raise RuntimeError("multi_output_config is required.")
    n_outputs = _infer_num_outputs(train_Y)
    output_configs, output_names, inline_spec_kwargs, embedded_fit_configs = _resolve_output_configs(config, n_outputs)
    sub_bundles: list[ModelBundle] = []
    for i, output_config in enumerate(output_configs):
        output_train_Y = _slice_output_y(train_Y, i, dim=mo_config.train_y_slice_dim)
        sub_bundles.append(_build_single_model(train_X=train_X, train_Y=output_train_Y, config=output_config, model_registry=model_registry))
    model = _build_wrapper_from_submodels([b.model for b in sub_bundles], output_configs, mo_config, output_names=output_names, output_spec_kwargs=inline_spec_kwargs)
    bundle_train_X = getattr(model, "train_X", None)
    if bundle_train_X is None and hasattr(model, "train_inputs"):
        train_inputs = getattr(model, "train_inputs")
        bundle_train_X = train_inputs[0] if isinstance(train_inputs, tuple) else train_inputs
    if bundle_train_X is None:
        bundle_train_X = train_X
    bundle_train_Y = getattr(model, "train_Y", None)
    if bundle_train_Y is None:
        bundle_train_Y = getattr(model, "train_targets", None)
    if bundle_train_Y is None:
        bundle_train_Y = train_Y
    return ModelBundle(
        model=model,
        train_X=bundle_train_X,
        train_Y=bundle_train_Y,
        model_config=config,
        input_type=config.input_type or infer_input_type(_as_cat_dims(config.cat_dims)),
        task_type=str(config.task_type),
        model_type=str(config.model_type),
        cat_dims=_as_cat_dims(config.cat_dims),
        metadata={
            "model_cls": model.__class__.__name__,
            "multi_output": True,
            "sub_bundles": sub_bundles,
            "output_configs": output_configs,
            "embedded_fit_configs": embedded_fit_configs,
        },
    )


def build_model(train_X: Any, train_Y: Any, config: ModelConfig, *, model_registry: Mapping[Any, Any] | None = None) -> ModelBundle:
    if config.multi_output_config is not None:
        return build_multi_output_model(train_X, train_Y, config, model_registry=model_registry)
    return _build_single_model(train_X, train_Y, config, model_registry=model_registry)


def _get_num_data(model: Any) -> int:
    if hasattr(model, "model") and hasattr(model.model, "train_inputs"):
        return int(model.model.train_inputs[0].shape[-2])
    if hasattr(model, "train_inputs"):
        train_inputs = model.train_inputs
        return int(train_inputs[0].shape[-2] if isinstance(train_inputs, tuple) else train_inputs.shape[-2])
    if hasattr(model, "train_X"):
        return int(model.train_X.shape[-2])
    raise AttributeError(f"Could not infer num_data from {model.__class__.__name__}.")


def _make_default_mll(bundle: ModelBundle, config: FitConfig) -> Any | None:
    model = bundle.model
    task_type = str(bundle.task_type)

    if config.use_model_make_mll and hasattr(model, "make_mll"):
        return model.make_mll(**config.mll_kwargs)
    if config.mll_factory is not None:
        return config.mll_factory(model, **config.mll_kwargs)
    if config.mll_cls is not None:
        if hasattr(model, "likelihood"):
            return config.mll_cls(model.likelihood, model, **config.mll_kwargs)
        return config.mll_cls(model, **config.mll_kwargs)

    if task_type == "ordinal":
        from bochan.fit import make_ordinal_mll

        return make_ordinal_mll(model, **config.mll_kwargs)

    if task_type in {"binary", "multiclass"}:
        from gpytorch.mlls import VariationalELBO

        likelihood = getattr(model, "likelihood", None)
        mll_model = getattr(model, "model", model)
        return VariationalELBO(likelihood=likelihood, model=mll_model, num_data=_get_num_data(model), **config.mll_kwargs)

    if hasattr(model, "likelihood"):
        from gpytorch.mlls import ExactMarginalLogLikelihood

        return ExactMarginalLogLikelihood(model.likelihood, model, **config.mll_kwargs)

    return None


def _fit_kwargs_from_config(config: FitConfig, *, exact: bool = False) -> dict[str, Any]:
    kwargs = dict(config.fit_kwargs)

    if exact:
        optimizer_kwargs = dict(config.optimizer_kwargs)
        if config.maxiter is not None:
            options = dict(optimizer_kwargs.get("options", {}))
            options.setdefault("maxiter", int(config.maxiter))
            optimizer_kwargs["options"] = options
        if optimizer_kwargs:
            kwargs.setdefault("optimizer_kwargs", optimizer_kwargs)
        return kwargs

    if config.num_epochs is not None:
        kwargs.setdefault("num_epochs", int(config.num_epochs))
    if config.lr is not None:
        kwargs.setdefault("lr", float(config.lr))
    if config.batch_size is not None:
        kwargs.setdefault("batch_size", int(config.batch_size))
    kwargs.setdefault("shuffle", bool(config.shuffle))
    kwargs.setdefault("verbose", bool(config.verbose))
    if config.clip_grad_norm is not None:
        kwargs.setdefault("clip_grad_norm", float(config.clip_grad_norm))
    if config.optimizer_kwargs:
        kwargs.setdefault("optimizer_kwargs", dict(config.optimizer_kwargs))
    return kwargs


def _resolve_fit_func(bundle: ModelBundle, config: FitConfig, mll: Any | None) -> tuple[Callable[..., Any], bool]:
    if config.fit_func is not None:
        return config.fit_func, False

    task_type = str(bundle.task_type)
    model_type = str(bundle.model_type).lower()

    if task_type == "binary":
        if model_type == "rrp":
            from bochan.fit import fit_rrp_binary_classifier_mll

            return fit_rrp_binary_classifier_mll, False
        from bochan.fit import fit_binary_classifier_mll

        return fit_binary_classifier_mll, False

    if task_type == "ordinal":
        if model_type == "rrp":
            from bochan.fit import fit_rrp_ordinal_mll

            return fit_rrp_ordinal_mll, False
        from bochan.fit import fit_ordinal_mll

        return fit_ordinal_mll, False

    if task_type == "multiclass":
        from bochan.fit import fit_multiclass_mll

        return fit_multiclass_mll, False

    if "deepgp" in model_type:
        from bochan.fit import fit_deepgp_mll

        return fit_deepgp_mll, False

    if "deepkernel" in model_type:
        from bochan.fit import fit_deepkernel_mll

        return fit_deepkernel_mll, False

    if mll is not None:
        from botorch.fit import fit_gpytorch_mll

        return fit_gpytorch_mll, True

    model = bundle.model
    if hasattr(model, "fit") and callable(model.fit):
        return model.fit, False

    raise ValueError(
        "No fit function could be inferred. "
        "Set FitConfig.fit_func explicitly or FitConfig.skip_fit=True."
    )


def _fit_single_bundle(bundle: ModelBundle, config: FitConfig | None = None) -> ModelBundle:
    config = config or FitConfig()
    if config.skip_fit:
        bundle.fit_config = config
        return bundle

    mll = _make_default_mll(bundle, config)
    fit_func, exact = _resolve_fit_func(bundle, config, mll)
    fit_target = mll if mll is not None else bundle.model
    fit_kwargs = _fit_kwargs_from_config(config, exact=exact)
    fit_kwargs = _filter_kwargs_for_callable(fit_func, fit_kwargs)
    fit_result = fit_func(fit_target, **fit_kwargs)

    bundle.fit_config = config
    bundle.mll = mll
    bundle.fit_result = fit_result
    bundle.metadata.update({"mll": None if mll is None else mll.__class__.__name__, "fit_func": getattr(fit_func, "__name__", str(fit_func))})
    return bundle


def fit_model(bundle: ModelBundle, config: FitConfig | None = None) -> ModelBundle:
    mo_config = bundle.model_config.multi_output_config
    if mo_config is not None:
        sub_bundles = bundle.metadata.get("sub_bundles", [])
        embedded_fit_configs = bundle.metadata.get("embedded_fit_configs", [])
        output_fit_configs = _resolve_output_fit_configs(config or FitConfig(), mo_config, len(sub_bundles), embedded=embedded_fit_configs)
        if mo_config.fit_submodels:
            for sub_bundle, sub_fit_config in zip(sub_bundles, output_fit_configs):
                _fit_single_bundle(sub_bundle, sub_fit_config)
        bundle.fit_config = config
        bundle.metadata["sub_fit_configs"] = output_fit_configs
        if mo_config.fit_wrapper:
            return _fit_single_bundle(bundle, config)
        return bundle
    return _fit_single_bundle(bundle, config)


def _has_var_keyword(signature: inspect.Signature) -> bool:
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())


def _filter_kwargs_for_callable(func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    if _has_var_keyword(signature):
        return kwargs
    allowed = set(signature.parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


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

    return make_grid_k_sparse_post_processing_func(
        bounds=repair_bounds,
        numeric_indices=repair.numeric_indices,
        steps=repair.steps,
        comp_idx=repair.comp_idx,
        k=repair.k,
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

    common_kwargs = {
        "acq_function": acqf,
        "bounds": bounds,
        "q": config.q,
        "num_restarts": config.num_restarts,
        "raw_samples": config.raw_samples,
        "return_best_only": config.return_best_only,
    }

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
