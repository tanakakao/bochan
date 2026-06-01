"""Factory functions used by the high-level BayesianOptimizer API."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Callable

from .configs import (
    AcquisitionConfig,
    DataContext,
    FitConfig,
    InputType,
    ModelBundle,
    ModelConfig,
    OptimizeConfig,
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

    `config.model_cls` が指定されていればそれを優先します。
    指定されていない場合は `model_registry` を使います。

    対応する registry 形式:
        1. flat: `{("normal", "regression", "base"): SingleTaskGP}`
        2. nested: `{"normal": {"regression": {"base": SingleTaskGP}}}`
    """
    if config.model_cls is not None:
        return config.model_cls

    if model_registry is None:
        raise ValueError(
            "model_cls is None. Provide config.model_cls or pass model_registry."
        )

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


def build_model(
    train_X: Any,
    train_Y: Any,
    config: ModelConfig,
    *,
    model_registry: Mapping[Any, Any] | None = None,
) -> ModelBundle:
    """ModelConfig に基づいてモデルを生成する。"""
    cat_dims = _as_cat_dims(config.cat_dims)
    input_type = config.input_type or infer_input_type(cat_dims)
    model_cls = resolve_model_cls(config, model_registry=model_registry)

    kwargs = {
        config.train_x_name: train_X,
        config.train_y_name: train_Y,
    }

    pass_cat_dims = config.pass_cat_dims
    if pass_cat_dims is None:
        pass_cat_dims = bool(cat_dims)

    if pass_cat_dims:
        kwargs["cat_dims"] = cat_dims

    if config.pass_input_transform and config.input_transform is not None:
        kwargs["input_transform"] = config.input_transform

    if config.pass_outcome_transform and config.outcome_transform is not None:
        kwargs["outcome_transform"] = config.outcome_transform

    kwargs.update(config.model_kwargs)
    model = model_cls(**kwargs)

    return ModelBundle(
        model=model,
        train_X=train_X,
        train_Y=train_Y,
        model_config=config,
        input_type=input_type,
        task_type=str(config.task_type),
        model_type=str(config.model_type),
        cat_dims=cat_dims,
        metadata={
            "model_cls": getattr(model_cls, "__name__", str(model_cls)),
        },
    )


def _make_mll(model: Any, config: FitConfig) -> Any | None:
    if config.use_model_make_mll and hasattr(model, "make_mll"):
        return model.make_mll(**config.mll_kwargs)

    if config.mll_factory is not None:
        return config.mll_factory(model, **config.mll_kwargs)

    if config.mll_cls is not None:
        if hasattr(model, "likelihood"):
            return config.mll_cls(model.likelihood, model, **config.mll_kwargs)
        return config.mll_cls(model, **config.mll_kwargs)

    return None


def fit_model(bundle: ModelBundle, config: FitConfig | None = None) -> ModelBundle:
    """ModelBundle 内のモデルを学習する。

    FitConfig が None または skip_fit=True の場合は何もしません。
    """
    if config is None or config.skip_fit:
        bundle.fit_config = config
        return bundle

    model = bundle.model
    mll = _make_mll(model, config)

    fit_func = config.fit_func
    if fit_func is None:
        if mll is not None:
            from botorch.fit import fit_gpytorch_mll

            fit_func = fit_gpytorch_mll
        elif hasattr(model, "fit") and callable(model.fit):
            fit_func = model.fit
        else:
            raise ValueError(
                "fit_func is None and no default fit function could be inferred. "
                "Set FitConfig.fit_func or FitConfig.skip_fit=True."
            )

    fit_target = mll if mll is not None else model
    fit_result = fit_func(fit_target, **config.fit_kwargs)

    bundle.fit_config = config
    bundle.mll = mll
    bundle.fit_result = fit_result
    bundle.metadata.update(
        {
            "mll": None if mll is None else mll.__class__.__name__,
            "fit_func": getattr(fit_func, "__name__", str(fit_func)),
        }
    )
    return bundle


def _has_var_keyword(signature: inspect.Signature) -> bool:
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )


def _filter_kwargs_for_callable(func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """callable の signature に存在しない kwargs を落とす。

    クラスの場合はコンストラクタ signature を参照します。
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs

    if _has_var_keyword(signature):
        return kwargs

    allowed = set(signature.parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


def build_acquisition(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: DataContext | None = None,
) -> Any:
    """AcquisitionConfig に基づいて獲得関数を生成する。"""
    data_context = data_context or DataContext()

    if config.acqf_factory is not None:
        return config.acqf_factory(
            bundle=bundle,
            config=config,
            data_context=data_context,
        )

    if config.acqf_cls is None:
        raise ValueError(
            "acqf_cls is None. Provide AcquisitionConfig.acqf_cls or acqf_factory."
        )

    kwargs = {"model": bundle.model}
    kwargs.update(config.acqf_kwargs)

    if config.objective is not None:
        kwargs["objective"] = config.objective

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


def optimize_candidates(
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
) -> tuple[Any, Any]:
    """獲得関数を最適化して候補点を返す。"""
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

    if config.post_processing_func is not None:
        common_kwargs["post_processing_func"] = config.post_processing_func

    if config.fixed_features is not None:
        common_kwargs["fixed_features"] = config.fixed_features

    if config.inequality_constraints is not None:
        common_kwargs["inequality_constraints"] = config.inequality_constraints

    if config.equality_constraints is not None:
        common_kwargs["equality_constraints"] = config.equality_constraints

    common_kwargs.update(config.optimizer_kwargs)

    optimizer = config.optimizer

    if callable(optimizer) and not isinstance(optimizer, str):
        kwargs = dict(common_kwargs)
        kwargs["sequential"] = config.sequential
        kwargs = _filter_kwargs_for_callable(optimizer, kwargs)
        return optimizer(**kwargs)

    if optimizer == "optimize_acqf":
        from botorch.optim import optimize_acqf

        kwargs = dict(common_kwargs)
        kwargs["sequential"] = config.sequential
        kwargs = _filter_kwargs_for_callable(optimize_acqf, kwargs)
        return optimize_acqf(**kwargs)

    if optimizer == "optimize_acqf_mixed":
        from botorch.optim import optimize_acqf_mixed

        kwargs = dict(common_kwargs)
        if config.fixed_features_list is None:
            raise ValueError(
                "OptimizeConfig.fixed_features_list is required when "
                "optimizer='optimize_acqf_mixed'."
            )
        kwargs["fixed_features_list"] = config.fixed_features_list
        kwargs["sequential"] = config.sequential
        kwargs = _filter_kwargs_for_callable(optimize_acqf_mixed, kwargs)
        return optimize_acqf_mixed(**kwargs)

    raise ValueError(f"Unknown optimizer: {optimizer}")
