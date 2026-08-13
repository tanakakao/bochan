"""Observation-aware model building for the canonical optimizer."""

from __future__ import annotations

from typing import Any

from ..configs import ModelBundle, ModelConfig
from ..factory import (
    _as_cat_dims,
    _build_single_model,
    _build_wrapper_from_submodels,
    _infer_num_outputs,
    _resolve_output_configs,
    build_model,
    infer_input_type,
)


def _normalize_model_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _supports_wide_missing_targets(config: ModelConfig) -> bool:
    """Return whether one model consumes a wide matrix with NaN task cells."""

    name = _normalize_model_name(config.model_type)
    return name == "multitask" or name.endswith("widemultitask")


def _reject_unsupported_correlated_missing(
    config: ModelConfig,
    train_Y: Any,
) -> None:
    import torch

    if not bool(torch.isnan(torch.as_tensor(train_Y)).any()):
        return
    name = _normalize_model_name(config.model_type)
    if name == "kronecker" or name.endswith("kronecker"):
        raise ValueError(
            "Kronecker multi-task models require a complete rectangular target matrix. "
            "Use model_type='multitask' / a '*_wide_multitask' model for partially "
            "observed objectives. Missing targets are not imputed automatically."
        )
    if name == "multifidelity" or name.endswith("multifidelity"):
        raise ValueError(
            "The current multi-fidelity model requires complete target rows. "
            "Partially observed objective cells are not imputed automatically."
        )


def _partial_hybrid_wrapper(
    model: Any,
    *,
    train_X: Any,
    train_Y: Any,
    observed_mask: Any,
) -> Any:
    """Retain the original wide observation table for a Hybrid wrapper."""

    from bochan.models.hybrid import HybridMultiOutputModel

    if not isinstance(model, HybridMultiOutputModel):
        return model
    from bochan.models.hybrid.partial_observation import (
        PartiallyObservedHybridMultiOutputModel,
    )

    return PartiallyObservedHybridMultiOutputModel(
        specs=list(model.specs),
        train_X_wide=train_X,
        train_Y_wide=train_Y,
        observed_mask_wide=observed_mask,
    )


def _build_split_partial_bundle(
    *,
    train_X: Any,
    train_Y: Any,
    config: ModelConfig,
    model_registry: Any = None,
) -> ModelBundle:
    """Build split-output models from only rows observed for each output."""

    import torch

    multi_output_config = config.multi_output_config
    if multi_output_config is None:
        raise RuntimeError(
            "multi_output_config is required for split partial observations."
        )
    n_outputs = _infer_num_outputs(train_Y)
    output_configs, output_names, inline_spec_kwargs, embedded_fit_configs = (
        _resolve_output_configs(config, n_outputs)
    )
    observed_mask = torch.isfinite(torch.as_tensor(train_Y))
    sub_bundles: list[ModelBundle] = []
    observed_counts: list[int] = []

    for index, output_config in enumerate(output_configs):
        mask = observed_mask[:, index]
        count = int(mask.sum().item())
        if count == 0:
            name = output_names[index] or f"output_{index}"
            raise ValueError(
                f"{name}: at least one observed target value is required."
            )
        observed_counts.append(count)
        output_X = train_X[mask]
        output_Y = train_Y[mask, index : index + 1]
        sub_bundles.append(
            _build_single_model(
                train_X=output_X,
                train_Y=output_Y,
                config=output_config,
                model_registry=model_registry,
            )
        )

    model = _build_wrapper_from_submodels(
        [bundle.model for bundle in sub_bundles],
        output_configs,
        multi_output_config,
        output_names=output_names,
        output_spec_kwargs=inline_spec_kwargs,
    )
    model = _partial_hybrid_wrapper(
        model,
        train_X=train_X,
        train_Y=train_Y,
        observed_mask=observed_mask,
    )

    return ModelBundle(
        model=model,
        train_X=train_X,
        train_Y=train_Y,
        model_config=config,
        input_type=config.input_type or infer_input_type(_as_cat_dims(config.cat_dims)),
        task_type=str(config.task_type),
        model_type=str(config.model_type),
        cat_dims=_as_cat_dims(config.cat_dims),
        metadata={
            "model_cls": model.__class__.__name__,
            "multi_output": True,
            "partial_observation": True,
            "observed_per_output": observed_counts,
            "sub_bundles": sub_bundles,
            "output_configs": output_configs,
            "embedded_fit_configs": embedded_fit_configs,
        },
    )


def build_objective_bundle(
    *,
    train_X: Any,
    train_Y: Any,
    config: ModelConfig,
    model_registry: Any = None,
) -> ModelBundle:
    """Build objective models without imputing missing target values."""

    import torch

    Y = torch.as_tensor(train_Y)
    has_missing = bool(torch.isnan(Y).any())
    if not has_missing:
        return build_model(
            train_X=train_X,
            train_Y=train_Y,
            config=config,
            model_registry=model_registry,
        )

    _reject_unsupported_correlated_missing(config, Y)
    if _supports_wide_missing_targets(config):
        return build_model(
            train_X=train_X,
            train_Y=train_Y,
            config=config,
            model_registry=model_registry,
        )

    if config.multi_output_config is not None:
        return _build_split_partial_bundle(
            train_X=train_X,
            train_Y=Y,
            config=config,
            model_registry=model_registry,
        )

    if int(Y.shape[-1]) != 1:
        raise ValueError(
            "Partially observed multi-output data requires model_type='multitask' "
            "or a MultiOutputConfig so each output can be fitted from its observed rows."
        )
    finite = torch.isfinite(Y[:, 0])
    if not bool(finite.any()):
        raise ValueError("The objective has no observed values.")
    return _build_single_model(
        train_X=train_X[finite],
        train_Y=Y[finite],
        config=config,
        model_registry=model_registry,
    )


__all__ = ["build_objective_bundle"]
