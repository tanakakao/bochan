"""Model construction for the public API."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..configs import (
    FitConfig,
    InputType,
    ModelBundle,
    ModelConfig,
    MultiOutputConfig,
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
        from ..registry.model import DEFAULT_MODEL_REGISTRY

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
            train_inputs = model.train_inputs
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
        metadata={"model_cls": model_cls_name, "model_factory": config.model_factory is not None,
                  "posterior_family": "non_gaussian" if _is_non_gaussian_regression_model(model) else "gaussian",
                  "multi_output": False,
                  "heteroscedastic": "hetero" in type(model).__name__.lower(),
                  "non_gaussian_families": [type(getattr(model, "likelihood", None)).__name__] if _is_non_gaussian_regression_model(model) else []},
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
        if any(_is_non_gaussian_regression_model(model) for model in submodels):
            from bochan.models.regression.multioutput import NonGaussianModelList

            return NonGaussianModelList(*submodels)
        from botorch.models.model_list_gp_regression import ModelListGP

        return ModelListGP(*submodels)
    if task_type == "binary":
        from bochan.models.classification.binary.base import MultiOutputBinaryClassificationModel

        return MultiOutputBinaryClassificationModel(*submodels, **mo_config.wrapper_kwargs)
    if task_type == "ordinal":
        from bochan.models.multioutput.ordinal import MultiOutputOrdinalModel

        return MultiOutputOrdinalModel(*submodels, **mo_config.wrapper_kwargs)
    raise ValueError(
        f"No dedicated homogeneous multi-output wrapper is available for task_type={task_type!r}. "
        "Set use_hybrid=True or provide wrapper_cls."
    )


def _is_non_gaussian_regression_model(model: Any) -> bool:
    """Return whether a model owns a non-Gaussian response posterior.

    Args:
        model: Candidate single-output model.

    Returns:
        Whether the model belongs to the non-Gaussian regression protocol.
    """
    module_name = type(model).__module__
    return module_name.startswith(
        (
            "bochan.models.regression.beta.",
            "bochan.models.regression.gamma.",
            "bochan.models.regression.count.",
        )
    ) or bool(getattr(model, "is_non_gaussian_model", False))


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
        train_inputs = model.train_inputs
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
            "posterior_family": "non_gaussian" if all(_is_non_gaussian_regression_model(b.model) for b in sub_bundles) else ("hybrid" if any(_is_non_gaussian_regression_model(b.model) for b in sub_bundles) else "gaussian"),
            "heteroscedastic": any("hetero" in type(b.model).__name__.lower() for b in sub_bundles),
            "non_gaussian_families": [type(getattr(b.model, "likelihood", None)).__name__ for b in sub_bundles if _is_non_gaussian_regression_model(b.model)],
        },
    )


def build_model(train_X: Any, train_Y: Any, config: ModelConfig, *, model_registry: Mapping[Any, Any] | None = None) -> ModelBundle:
    if config.multi_output_config is not None:
        return build_multi_output_model(train_X, train_Y, config, model_registry=model_registry)
    return _build_single_model(train_X, train_Y, config, model_registry=model_registry)


