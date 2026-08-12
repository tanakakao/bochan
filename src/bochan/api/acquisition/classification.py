"""Classification objective helpers for high-level acquisition construction."""

from __future__ import annotations

from typing import Any

from .. import factory as _factory
from ..configs import AcquisitionConfig, ModelBundle, ObjectiveConfig


def _num_outputs(bundle: ModelBundle) -> int:
    try:
        return max(1, int(bundle.model.num_outputs))
    except (AttributeError, TypeError, ValueError):
        shape = getattr(bundle.train_Y, "shape", None)
        return 1 if shape is None or len(shape) <= 1 else int(shape[-1])


def _signs(
    bundle: ModelBundle,
    config: ObjectiveConfig,
    kwargs: dict[str, Any],
) -> Any:
    explicit = kwargs.pop("objective_signs", None)
    if explicit is not None:
        return explicit
    if config.directions is not None:
        return [_factory._direction_to_sign(value) for value in config.directions]
    if config.maximize:
        return None
    return [-1.0] * _num_outputs(bundle)


def build_ordinal_objective(
    bundle: ModelBundle,
    config: ObjectiveConfig,
) -> Any:
    """Build the multi-output ordinal objective when requested."""

    if _factory._objective_mode(config) != "multi_output":
        return _factory._build_ordinal_objective(bundle, config)

    from bochan.acquisition.ordinal.bayesian_optimization import (
        qMultiOutputOrdinalUtilityObjective,
    )
    from bochan.acquisition.ordinal.bayesian_optimization._utility_defaults import (
        infer_multioutput_ordinal_utility_values,
    )

    utility_values = config.utility_values
    if utility_values is None:
        utility_values = infer_multioutput_ordinal_utility_values(bundle.model)

    kwargs = dict(config.objective_kwargs)
    ordinal_likelihoods = kwargs.pop("ordinal_likelihoods", None)
    if ordinal_likelihoods is None:
        ordinal_likelihoods = config.ordinal_likelihood
    resolved = {
        "model": bundle.model,
        "utility_values": utility_values,
        "ordinal_likelihoods": ordinal_likelihoods,
        "objective_signs": _signs(bundle, config, kwargs),
        "link": kwargs.pop("link", "auto"),
        "input_perturbation_n_w": kwargs.pop(
            "input_perturbation_n_w",
            kwargs.pop("n_w", config.n_w),
        ),
        "risk_type": kwargs.pop("risk_type", config.risk_type),
        "risk_alpha": kwargs.pop(
            "risk_alpha",
            kwargs.pop("alpha", config.alpha),
        ),
    }
    resolved.update(kwargs)
    resolved = _factory._filter_kwargs_for_callable(
        qMultiOutputOrdinalUtilityObjective,
        resolved,
    )
    return qMultiOutputOrdinalUtilityObjective(**resolved)


def build_multiclass_objective(
    bundle: ModelBundle,
    config: AcquisitionConfig,
) -> Any | None:
    """Build the probability objective for multi-output multiclass BO."""

    from bochan.acquisition.multiclass.bayesian_optimization.input_perturbation import (
        InputPerturbationMultiOutputObjectiveAdapter,
    )
    from bochan.acquisition.multiclass.bayesian_optimization.multi_output import (
        MulticlassTargetProbabilityObjective,
    )

    objective_config = config.objective_config
    if objective_config is None:
        return None
    mode = _factory._objective_mode(objective_config)
    if mode == "none":
        return None
    if mode != "multi_output":
        raise ValueError(
            "Automatic multiclass objectives support mode='multi_output' for "
            "EHVI, NEHVI, NParEGO, and NSGA-II."
        )

    kwargs = dict(objective_config.objective_kwargs)
    acq_kwargs = dict(config.acqf_kwargs)
    target_class = kwargs.pop("target_class", acq_kwargs.get("target_class"))
    output_target_classes = kwargs.pop(
        "output_target_classes",
        acq_kwargs.get("output_target_classes"),
    )
    utility_values = kwargs.pop("utility_values", objective_config.utility_values)
    if utility_values is None:
        utility_values = acq_kwargs.get("utility_values")
    objective_signs = _signs(bundle, objective_config, kwargs)
    if objective_signs is None:
        objective_signs = acq_kwargs.get("objective_signs")

    base = MulticlassTargetProbabilityObjective(
        target_class=target_class,
        output_target_classes=output_target_classes,
        num_outputs=_num_outputs(bundle),
        class_reduction=kwargs.pop(
            "class_reduction",
            acq_kwargs.get("class_reduction", "mean"),
        ),
        utility_values=utility_values,
        objective_signs=objective_signs,
        eps=float(kwargs.pop("eps", acq_kwargs.get("eps", 1e-8))),
    )

    n_w = kwargs.pop("n_w", objective_config.n_w)
    if n_w is None or int(n_w) <= 1:
        return base
    return InputPerturbationMultiOutputObjectiveAdapter(
        base,
        n_w=int(n_w),
        risk_type=kwargs.pop("risk_type", objective_config.risk_type),
        alpha=float(kwargs.pop("alpha", objective_config.alpha)),
    )


def objective_keeps_perturbation_expanded(config: AcquisitionConfig) -> bool:
    objective_config = config.objective_config
    return bool(
        objective_config is not None
        and objective_config.n_w is not None
        and int(objective_config.n_w) > 1
        and objective_config.risk_type is None
        and objective_config.aggregate_mean_when_no_risk is False
    )


def prepare_objective_instance(
    objective: Any,
    config: AcquisitionConfig,
) -> Any:
    """Configure only the constructed objective instance for perturbation shapes."""

    if objective is None:
        return None

    inner_objective = getattr(objective, "inner_objective", None)
    if inner_objective is not None and hasattr(inner_objective, "_verify_output_shape"):
        objective_config = config.objective_config
        if (
            objective_config is not None
            and objective_config.n_w is not None
            and int(objective_config.n_w) > 1
        ):
            inner_objective._verify_output_shape = False

    if objective_keeps_perturbation_expanded(config):
        objective._verify_output_shape = False
    return objective


__all__ = [
    "build_multiclass_objective",
    "build_ordinal_objective",
    "prepare_objective_instance",
]
