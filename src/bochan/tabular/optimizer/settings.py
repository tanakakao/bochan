"""Model-setting normalization used by the tabular optimizer facade."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from bochan.api import InputTransformConfig, ModelConfig, OutputConfig
from bochan.models.regression.gaussian.likelihood import (
    build_multitask_likelihood,
    build_single_task_likelihood,
)

from ..config import UNSET

_TABULAR_NOISE_ALPHA_KEY = "_tabular_noise_alpha"
_ALPHA_SUPPORTED_MODEL_TYPES = frozenset(
    {"base", "deepgp", "deepkernel", "pca", "rembo", "rrp", "robust"}
)
_REGRESSION_TASK_TYPES = frozenset({"regression", "multi_objective"})


def validate_noise_alpha(value: Any | None) -> float | None:
    """Validate a positive Gaussian observation-noise floor."""

    if value is None:
        return None
    alpha = float(value)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be a finite positive value. Got {value!r}.")
    return alpha


def _extract_noise_alpha(
    model_kwargs: Mapping[str, Any] | None,
    explicit_alpha: float | None,
) -> tuple[float | None, dict[str, Any]]:
    kwargs = dict(model_kwargs or {})
    embedded = validate_noise_alpha(kwargs.pop(_TABULAR_NOISE_ALPHA_KEY, None))
    explicit = validate_noise_alpha(explicit_alpha)
    if (
        explicit is not None
        and embedded is not None
        and not math.isclose(explicit, embedded, rel_tol=1e-12, abs_tol=0.0)
    ):
        raise ValueError(
            "Conflicting alpha values were supplied through the tabular API and model settings."
        )
    return explicit if explicit is not None else embedded, kwargs


def _build_noise_floor_likelihood(
    *,
    train_X: Any,
    train_Y: Any,
    model_type: str,
    alpha: float,
) -> Any:
    normalized = "rrp" if model_type == "robust" else model_type
    if normalized not in _ALPHA_SUPPORTED_MODEL_TYPES:
        raise ValueError(f"alpha is not supported for model_type={model_type!r}.")
    if getattr(train_Y, "ndim", 0) == 1:
        train_Y = train_Y.unsqueeze(-1)
    if normalized in {"deepgp", "deepkernel"} and train_Y.shape[-1] > 1:
        likelihood = build_multitask_likelihood(
            train_X=train_X,
            train_Y=train_Y,
            rank=0,
            alpha=alpha,
        )
    else:
        likelihood = build_single_task_likelihood(
            train_X=train_X,
            train_Y=train_Y,
            alpha=alpha,
        )
    return likelihood.to(train_X)


def _apply_alpha_to_output_config(
    raw_config: Any,
    *,
    train_X: Any,
    train_Y: Any,
    explicit_alpha: float | None,
) -> tuple[Any, bool]:
    if isinstance(raw_config, ModelConfig):
        return (
            apply_alpha_to_model_config(
                raw_config,
                train_X=train_X,
                train_Y=train_Y,
                explicit_alpha=explicit_alpha,
            ),
            str(raw_config.task_type) in _REGRESSION_TASK_TYPES,
        )

    if isinstance(raw_config, OutputConfig):
        alpha, kwargs = _extract_noise_alpha(raw_config.model_kwargs, explicit_alpha)
        if str(raw_config.task_type) not in _REGRESSION_TASK_TYPES:
            return replace(raw_config, model_kwargs=kwargs), False
        if alpha is None:
            return replace(raw_config, model_kwargs=kwargs), False
        if "likelihood" in kwargs:
            raise ValueError("Specify either alpha or model_kwargs['likelihood'], not both.")
        kwargs["likelihood"] = _build_noise_floor_likelihood(
            train_X=train_X,
            train_Y=train_Y,
            model_type=str(raw_config.model_type),
            alpha=alpha,
        )
        return replace(raw_config, model_kwargs=kwargs), True

    if isinstance(raw_config, Mapping):
        payload = dict(raw_config)
        alpha, kwargs = _extract_noise_alpha(
            payload.get("model_kwargs"),
            explicit_alpha,
        )
        payload["model_kwargs"] = kwargs
        task_type = str(payload.get("task_type", ""))
        if task_type not in _REGRESSION_TASK_TYPES or alpha is None:
            return payload, False
        if "likelihood" in kwargs:
            raise ValueError("Specify either alpha or model_kwargs['likelihood'], not both.")
        kwargs["likelihood"] = _build_noise_floor_likelihood(
            train_X=train_X,
            train_Y=train_Y,
            model_type=str(payload.get("model_type", "base")),
            alpha=alpha,
        )
        return payload, True

    if isinstance(raw_config, str):
        if raw_config not in _REGRESSION_TASK_TYPES or explicit_alpha is None:
            return raw_config, False
        likelihood = _build_noise_floor_likelihood(
            train_X=train_X,
            train_Y=train_Y,
            model_type="base",
            alpha=explicit_alpha,
        )
        return (
            OutputConfig(
                task_type=raw_config,
                model_type="base",
                model_kwargs={"likelihood": likelihood},
            ),
            True,
        )

    raise TypeError(
        "Unsupported multi-output config entry while applying alpha: "
        f"{type(raw_config).__name__}."
    )


def apply_alpha_to_model_config(
    model_config: ModelConfig,
    *,
    train_X: Any,
    train_Y: Any,
    explicit_alpha: float | None,
) -> ModelConfig:
    """Inject alpha-configured likelihoods into regression model configs."""

    alpha, root_kwargs = _extract_noise_alpha(
        model_config.model_kwargs,
        explicit_alpha,
    )
    if alpha is not None and "likelihood" in root_kwargs:
        raise ValueError("Specify either alpha or model_kwargs['likelihood'], not both.")
    model_config = replace(model_config, model_kwargs=root_kwargs)

    if str(model_config.task_type) == "hybrid":
        multi_output = model_config.multi_output_config
        if multi_output is None or multi_output.output_configs is None:
            if alpha is not None:
                raise ValueError(
                    "alpha requires explicit output_configs for a hybrid tabular model."
                )
            return model_config

        updated_outputs: list[Any] = []
        applied = False
        for index, raw_config in enumerate(multi_output.output_configs):
            updated, output_applied = _apply_alpha_to_output_config(
                raw_config,
                train_X=train_X,
                train_Y=train_Y[..., index : index + 1],
                explicit_alpha=alpha,
            )
            updated_outputs.append(updated)
            applied = applied or output_applied
        if alpha is not None and not applied:
            raise ValueError(
                "alpha was specified, but the hybrid model has no regression output."
            )
        return replace(
            model_config,
            multi_output_config=replace(
                multi_output,
                output_configs=updated_outputs,
            ),
        )

    if str(model_config.task_type) not in _REGRESSION_TASK_TYPES:
        if alpha is not None:
            raise ValueError("alpha is available only for regression outputs.")
        return model_config
    if alpha is None:
        return model_config

    root_kwargs["likelihood"] = _build_noise_floor_likelihood(
        train_X=train_X,
        train_Y=train_Y,
        model_type=str(model_config.model_type),
        alpha=alpha,
    )
    return replace(model_config, model_kwargs=root_kwargs)


def merge_input_transform_config(
    *,
    model_config: Any | None,
    input_transform_config: Any = UNSET,
    normalize: bool | Any = UNSET,
    perturbation: bool | Any = UNSET,
    n_w: int | Any = UNSET,
    std: float | Any = UNSET,
) -> Any:
    """Merge direct input-transform settings into a canonical config."""

    updates = {
        key: value
        for key, value in {
            "normalize": normalize,
            "perturbation": perturbation,
            "n_w": n_w,
            "std": std,
        }.items()
        if value is not UNSET
    }
    if not updates:
        return input_transform_config

    base = input_transform_config
    if base is UNSET and model_config is not None:
        if isinstance(model_config, Mapping):
            base = model_config.get("input_transform_config", UNSET)
        else:
            base = model_config.input_transform_config

    if base is UNSET or base is None:
        return InputTransformConfig(**updates)
    if isinstance(base, Mapping):
        return InputTransformConfig(**{**dict(base), **updates})
    if isinstance(base, InputTransformConfig):
        return replace(base, **updates)
    raise TypeError(
        "Direct input-transform fields require input_transform_config to be "
        f"None, a mapping, or InputTransformConfig. Got {type(base).__name__}."
    )


__all__ = [
    "apply_alpha_to_model_config",
    "merge_input_transform_config",
    "validate_noise_alpha",
]
