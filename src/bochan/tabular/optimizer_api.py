'''Public tabular optimizer API aligned with :mod:`bochan.api` config fields.'''

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from bochan.api import (
    AcquisitionConfig,
    FitConfig,
    InputTransformConfig,
    ModelConfig,
    OptimizeConfig,
    OutputConfig,
)
from bochan.models.regression.gaussian.likelihood import (
    build_multitask_likelihood,
    build_single_task_likelihood,
)

from .builders import UNSET, make_acquisition_config, make_fit_config, make_optimize_config
from .converter import resolve_column_indices
from .optimizer import TabularBayesianOptimizer as _BaseTabularBayesianOptimizer
from .outcome_constraints import apply_tabular_outcome_constraints

apply_tabular_outcome_constraints()

_ACQUISITION_DIRECT_KEYS = {
    "acq_name",
    "name",
    "acqf_cls",
    "acqf_factory",
    "objective",
    "objective_config",
    "objective_factory",
    "objective_kwargs",
    "sampler",
    "acqf_kwargs",
    "context_fields",
    "filter_kwargs_by_signature",
    "objective_mode",
    "objective_output",
    "objective_outputs",
    "objective_specs",
    "objective_directions",
    "objective_weights",
    "objective_eq_targets",
    "objective_direction",
    "objective_weight",
    "objective_eq_target",
    "objective_n_w",
    "objective_risk_type",
    "objective_alpha",
    "objective_maximize",
    "objective_aggregate_mean_when_no_risk",
    "objective_allow_unexpanded",
    "objective_utility_values",
    "objective_ordinal_likelihood",
    "constraints",
    "outcome_constraint_config",
}

_TABULAR_NOISE_ALPHA_KEY = "_tabular_noise_alpha"
_ALPHA_SUPPORTED_MODEL_TYPES = frozenset(
    {
        "base",
        "deepgp",
        "deepkernel",
        "pca",
        "rembo",
        "rrp",
        "robust",
    }
)
_REGRESSION_TASK_TYPES = frozenset({"regression", "multi_objective"})


def _validate_noise_alpha(value: Any | None) -> float | None:
    """Validate and normalize the Gaussian observation-noise floor."""

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
    """Resolve direct and Web-adapter alpha values and remove private metadata."""

    kwargs = dict(model_kwargs or {})
    embedded = kwargs.pop(_TABULAR_NOISE_ALPHA_KEY, None)
    embedded_alpha = _validate_noise_alpha(embedded)
    explicit_alpha = _validate_noise_alpha(explicit_alpha)
    if (
        explicit_alpha is not None
        and embedded_alpha is not None
        and not math.isclose(
            explicit_alpha,
            embedded_alpha,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            "Conflicting alpha values were supplied through the tabular API and "
            "the Web model settings."
        )
    return explicit_alpha if explicit_alpha is not None else embedded_alpha, kwargs


def _build_noise_floor_likelihood(
    *,
    train_X: Any,
    train_Y: Any,
    model_type: str,
    alpha: float,
) -> Any:
    """Build the recommended Gaussian likelihood for one tabular model."""

    normalized_model_type = "rrp" if model_type == "robust" else model_type
    if normalized_model_type not in _ALPHA_SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(_ALPHA_SUPPORTED_MODEL_TYPES - {"robust"}))
        raise ValueError(
            "alpha is supported only for the recommended Gaussian regression "
            f"models: {supported}. Got model_type={model_type!r}."
        )

    if getattr(train_Y, "ndim", 0) == 1:
        train_Y = train_Y.unsqueeze(-1)
    if normalized_model_type in {"deepgp", "deepkernel"} and train_Y.shape[-1] > 1:
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
    """Attach a likelihood to one regression output config when requested."""

    if isinstance(raw_config, ModelConfig):
        return (
            _apply_alpha_to_model_config(
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
        task_type = str(payload.get("task_type", ""))
        alpha, kwargs = _extract_noise_alpha(payload.get("model_kwargs"), explicit_alpha)
        payload["model_kwargs"] = kwargs
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


def _apply_alpha_to_model_config(
    model_config: ModelConfig,
    *,
    train_X: Any,
    train_Y: Any,
    explicit_alpha: float | None,
) -> ModelConfig:
    """Inject alpha-configured likelihoods without changing the tensor Core API."""

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
            raise ValueError("alpha was specified, but the hybrid model has no regression output.")
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


def _merge_input_transform_direct_values(
    *,
    model_config: Any | None,
    input_transform_config: Any = UNSET,
    normalize: bool | Any = UNSET,
    perturbation: bool | Any = UNSET,
    n_w: int | Any = UNSET,
    std: float | Any = UNSET,
) -> Any:
    """Merge direct tabular input-transform fields into ``InputTransformConfig``.

    ``TabularBayesianOptimizer`` is intended to be usable from notebooks without
    importing config objects.  The public API therefore accepts fields such as
    ``perturbation=True`` directly and converts them to the internal
    ``InputTransformConfig`` representation here.
    """

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
        if isinstance(model_config, dict):
            base = model_config.get("input_transform_config", UNSET)
        else:
            base = model_config.input_transform_config

    if base is UNSET or base is None:
        return InputTransformConfig(**updates)
    if isinstance(base, dict):
        return InputTransformConfig(**{**base, **updates})
    if isinstance(base, InputTransformConfig):
        return replace(base, **updates)

    raise TypeError(
        "Direct input transform fields such as perturbation, n_w, std, and "
        "normalize require input_transform_config to be None, dict, or "
        f"InputTransformConfig. Got {type(base).__name__}."
    )


def _apply_input_transform_direct_values(
    kwargs: dict[str, Any],
    *,
    model_config: Any | None,
    normalize: bool | Any = UNSET,
    perturbation: bool | Any = UNSET,
    n_w: int | Any = UNSET,
    std: float | Any = UNSET,
) -> None:
    merged = _merge_input_transform_direct_values(
        model_config=model_config,
        input_transform_config=kwargs.get("input_transform_config", UNSET),
        normalize=normalize,
        perturbation=perturbation,
        n_w=n_w,
        std=std,
    )
    if merged is not UNSET:
        kwargs["input_transform_config"] = merged


def _target_category_map_for_output(
    output: Any,
    target_names: list[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None,
) -> Mapping[Any, int] | None:
    """Return the original-label to class-index map for one target output."""

    if not target_category_maps:
        return None
    resolved = resolve_column_indices([output], target_names)
    if not resolved:
        return None
    target_name = target_names[resolved[0]]
    mapping = target_category_maps.get(target_name)
    if mapping is None:
        mapping = target_category_maps.get(str(target_name))
    return mapping


def _resolve_target_class_value(
    value: Any,
    *,
    output: Any,
    target_names: list[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None,
) -> Any:
    """Resolve a tabular class label while preserving numeric class indices."""

    if not isinstance(value, str):
        return value

    mapping = _target_category_map_for_output(
        output,
        target_names,
        target_category_maps,
    )
    if mapping is not None:
        if value in mapping:
            return int(mapping[value])
        for label, class_index in mapping.items():
            if str(label) == value:
                return int(class_index)

    # Numeric strings were accepted previously because the core spec applies
    # int(value). Preserve that behavior even when a categorical map exists.
    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    if mapping is None:
        raise ValueError(
            f"String target class {value!r} for output {output!r} cannot be "
            "resolved because no target category map is available. Use an "
            "encoded class index or fit with categorical target encoding."
        )

    raise KeyError(
        f"Unknown target class label {value!r} for output {output!r}. "
        f"Available labels: {list(mapping)!r}."
    )


def _resolve_constraint_target_classes(
    value: Any,
    *,
    target_names: list[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None,
) -> Any:
    """Resolve string ``target_class`` fields in one constraint mapping."""

    if not isinstance(value, Mapping):
        return value
    if "target_class" not in value and "target_classes" not in value:
        return value

    resolved = dict(value)
    output = resolved.get("output")
    if output is None:
        return resolved

    target_class = resolved.get("target_class")
    if target_class is not None:
        resolved["target_class"] = _resolve_target_class_value(
            target_class,
            output=output,
            target_names=target_names,
            target_category_maps=target_category_maps,
        )

    target_classes = resolved.get("target_classes")
    if target_classes is not None:
        values = [target_classes] if isinstance(target_classes, str) else list(target_classes)
        resolved["target_classes"] = [
            _resolve_target_class_value(
                item,
                output=output,
                target_names=target_names,
                target_category_maps=target_category_maps,
            )
            for item in values
        ]
    return resolved


def _resolve_outcome_constraint_config_columns(
    value: Any,
    target_names: list[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None = None,
) -> Any:
    """Resolve tabular target names and class labels in a serializable config."""

    if value is UNSET or value is None or not isinstance(value, Mapping):
        return value

    resolved = dict(value)
    if "output_indices" in resolved:
        output_indices = resolve_column_indices(resolved["output_indices"], target_names)
        resolved["output_indices"] = output_indices or []

    constraints = resolved.get("constraints")
    if constraints is not None:
        constraint_values = [constraints] if isinstance(constraints, Mapping) else list(constraints)
        resolved["constraints"] = [
            _resolve_constraint_target_classes(
                item,
                target_names=target_names,
                target_category_maps=target_category_maps,
            )
            for item in constraint_values
        ]
    return resolved


def _resolve_acquisition_config_columns(
    acq_config: Any,
    target_names: list[Any],
    target_category_maps: Mapping[Any, Mapping[Any, int]] | None = None,
) -> Any:
    """Resolve named outcomes and class labels nested in an acquisition config."""

    if not isinstance(acq_config, Mapping):
        return acq_config
    if "outcome_constraint_config" not in acq_config:
        return acq_config

    resolved = dict(acq_config)
    resolved["outcome_constraint_config"] = _resolve_outcome_constraint_config_columns(
        resolved["outcome_constraint_config"],
        target_names,
        target_category_maps,
    )
    return resolved


class TabularBayesianOptimizer(_BaseTabularBayesianOptimizer):
    '''Pandas / numpy friendly optimizer with public API convenience fields.

    This subclass preserves the existing tabular implementation while exposing
    recently added high-level API fields through direct keyword arguments.
    Existing ``model_config``, ``fit_config``, ``acq_config``, and ``opt_config``
    objects remain fully supported.
    '''

    def __init__(
        self,
        model_config: ModelConfig | dict[str, Any] | None = None,
        fit_config: FitConfig | dict[str, Any] | None = None,
        *,
        alpha: float | None = None,
        fit_beta: float | None | Any = UNSET,
        beta: float | None | Any = UNSET,
        normalize: bool | Any = UNSET,
        perturbation: bool | Any = UNSET,
        n_w: int | Any = UNSET,
        std: float | Any = UNSET,
        **kwargs: Any,
    ) -> None:
        self.alpha = _validate_noise_alpha(alpha)
        _apply_input_transform_direct_values(
            kwargs,
            model_config=model_config,
            normalize=normalize,
            perturbation=perturbation,
            n_w=n_w,
            std=std,
        )
        if fit_beta is not UNSET and beta is not UNSET:
            raise ValueError("Specify either fit_beta or beta, not both.")
        if beta is not UNSET:
            fit_beta = beta
        if fit_beta is not UNSET:
            fit_config = make_fit_config(fit_config, fit_beta=fit_beta)
        super().__init__(model_config=model_config, fit_config=fit_config, **kwargs)

    def _model_config_with_tabular_cat_dims(self, dataset: Any) -> ModelConfig:
        """Add inferred categorical dimensions and an optional noise-floor likelihood."""

        model_config = super()._model_config_with_tabular_cat_dims(dataset)
        return _apply_alpha_to_model_config(
            model_config,
            train_X=dataset.X,
            train_Y=dataset.Y,
            explicit_alpha=self.alpha,
        )

    def fit(
        self,
        data: Any | None = None,
        y: Any | None = None,
        *,
        fit_config: FitConfig | dict[str, Any] | None = None,
        alpha: float | None | Any = UNSET,
        fit_beta: float | None | Any = UNSET,
        beta: float | None | Any = UNSET,
        normalize: bool | Any = UNSET,
        perturbation: bool | Any = UNSET,
        n_w: int | Any = UNSET,
        std: float | Any = UNSET,
        **kwargs: Any,
    ) -> TabularBayesianOptimizer:
        if alpha is not UNSET:
            self.alpha = _validate_noise_alpha(alpha)
        _apply_input_transform_direct_values(
            kwargs,
            model_config=kwargs.get("model_config", self.model_config),
            normalize=normalize,
            perturbation=perturbation,
            n_w=n_w,
            std=std,
        )
        if fit_beta is not UNSET and beta is not UNSET:
            raise ValueError("Specify either fit_beta or beta, not both.")
        if beta is not UNSET:
            fit_beta = beta
        if fit_beta is not UNSET:
            fit_config = make_fit_config(fit_config or self.fit_config, fit_beta=fit_beta)
        return super().fit(data=data, y=y, fit_config=fit_config, **kwargs)

    def candidate(
        self,
        acq_config: AcquisitionConfig | dict[str, Any] | None = None,
        opt_config: OptimizeConfig | dict[str, Any] | None = None,
        *,
        constraints: Any = UNSET,
        outcome_constraint_config: Any = UNSET,
        objective_eq_targets: Any = UNSET,
        objective_eq_target: Any = UNSET,
        objective_maximize: Any = UNSET,
        objective_aggregate_mean_when_no_risk: Any = UNSET,
        objective_allow_unexpanded: Any = UNSET,
        objective_ordinal_likelihood: Any = UNSET,
        evo_method: Any = UNSET,
        **kwargs: Any,
    ) -> Any:
        target_names = list(self.dataset.target_names) if self.dataset is not None else []
        target_category_maps = (
            dict(getattr(self.dataset, "target_category_maps", None) or {})
            if self.dataset is not None
            else {}
        )
        if target_names:
            acq_config = _resolve_acquisition_config_columns(
                acq_config,
                target_names,
                target_category_maps,
            )
            outcome_constraint_config = _resolve_outcome_constraint_config_columns(
                outcome_constraint_config,
                target_names,
                target_category_maps,
            )

        acq_values = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if key in _ACQUISITION_DIRECT_KEYS
        }
        acq_values.update(
            {
                "constraints": constraints,
                "outcome_constraint_config": outcome_constraint_config,
                "objective_eq_targets": objective_eq_targets,
                "objective_eq_target": objective_eq_target,
                "objective_maximize": objective_maximize,
                "objective_aggregate_mean_when_no_risk": objective_aggregate_mean_when_no_risk,
                "objective_allow_unexpanded": objective_allow_unexpanded,
                "objective_ordinal_likelihood": objective_ordinal_likelihood,
            }
        )
        if any(value is not UNSET for value in acq_values.values()):
            acq_config = make_acquisition_config(acq_config, **acq_values)
        if evo_method is not UNSET:
            opt_config = make_optimize_config(opt_config, evo_method=evo_method)
        return super().candidate(acq_config=acq_config, opt_config=opt_config, **kwargs)


__all__ = ["TabularBayesianOptimizer"]
