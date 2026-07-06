"""Input-perturbation defaults for ordinal and multiclass vector objectives."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from . import engine as _engine
from . import engine_defaults as _engine_defaults
from . import factory as _factory
from .automatic_default_utils import _num_outputs as _bundle_num_outputs
from .configs import AcquisitionConfig, ModelBundle, ObjectiveConfig


_APPLIED = False
_BASE_BUILD_OBJECTIVE = _factory.build_objective
_BASE_BUILD_ORDINAL = _factory._build_ordinal_objective
_BASE_RESOLVE_OBJECTIVE = _engine._resolve_objective_config_n_w_from_input_transform
_BASE_RESOLVE_NPAREGO_CLASS = _engine_defaults._resolve_default_regression_nparego_class


def _normalize(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _is_vector_strategy(config: AcquisitionConfig) -> bool:
    name = _normalize(config.name)
    cls_name = _normalize(getattr(config.acqf_cls, "__name__", ""))
    combined = f"{name}{cls_name}"
    return (
        name in {
            "nsgaii",
            "nsga2",
            "ehi",
            "qehi",
            "ehvi",
            "qehvi",
            "nehi",
            "qnehi",
            "nehvi",
            "qnehvi",
        }
        or "nparego" in combined
        or "expectedhypervolumeimprovement" in combined
    )


def _is_multi_output(bundle: ModelBundle | None) -> bool:
    if bundle is None:
        return False
    if bool(bundle.metadata.get("multi_output", False)):
        return True
    try:
        return int(getattr(bundle.model, "num_outputs", 1)) > 1
    except (TypeError, ValueError):
        return False


def _num_outputs(bundle: ModelBundle) -> int:
    try:
        return max(1, int(getattr(bundle.model, "num_outputs")))
    except (AttributeError, TypeError, ValueError):
        shape = getattr(bundle.train_Y, "shape", None)
        return 1 if shape is None or len(shape) <= 1 else int(shape[-1])


def _resolve_hybrid_nparego_class(
    bundle: ModelBundle,
    config: AcquisitionConfig,
) -> AcquisitionConfig:
    """Route hybrid multi-output NParEGO to bochan's vector implementation.

    ``engine_defaults`` historically only applies this route when
    ``bundle.task_type == 'regression'``. A heterogeneous Hybrid bundle still
    exposes a vector objective space and must use the same self-scalarizing
    NParEGO implementation; otherwise the generic ``qExpectedImprovement``
    path returns one value per candidate, shaped ``t_batch x q``.
    """

    resolved = _BASE_RESOLVE_NPAREGO_CLASS(bundle, config)
    if resolved.acqf_cls is not config.acqf_cls:
        return resolved
    if _normalize(config.name) not in {"nparego", "qnparego"}:
        return config
    if str(bundle.task_type) != "hybrid" or _bundle_num_outputs(bundle.train_Y) < 2:
        return config

    from bochan.acquisition.regression.bayesian_optimization import (
        qMultiOutputRegressionNParEGO,
    )

    return replace(config, acqf_cls=qMultiOutputRegressionNParEGO)


def _signs(bundle: ModelBundle, config: ObjectiveConfig, kwargs: dict[str, Any]):
    explicit = kwargs.pop("objective_signs", None)
    if explicit is not None:
        return explicit
    if config.directions is not None:
        return [_factory._direction_to_sign(value) for value in config.directions]
    if config.maximize:
        return None
    return [-1.0] * _num_outputs(bundle)


def _build_ordinal(bundle: ModelBundle, config: ObjectiveConfig):
    if _factory._objective_mode(config) != "multi_output":
        return _BASE_BUILD_ORDINAL(bundle, config)

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


def _build_multiclass(bundle: ModelBundle, config: AcquisitionConfig):
    from bochan.acquisition.multiclass.bayesian_optimization.input_perturbation_compat import (
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
            "Automatic multiclass objectives currently support mode='multi_output' "
            "for EHVI, NEHVI, NParEGO, and NSGA-II."
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


def _build_objective(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: Any | None = None,
):
    if (
        str(bundle.task_type) != "multiclass"
        or config.objective is not None
        or config.objective_factory is not None
        or config.objective_config is None
    ):
        return _BASE_BUILD_OBJECTIVE(
            bundle=bundle,
            config=config,
            data_context=data_context,
        )
    return _build_multiclass(bundle, config)


def _keep_constrained_perturbation_q_expanded(
    *,
    bundle: ModelBundle | None,
    config: AcquisitionConfig,
) -> AcquisitionConfig:
    """Keep constrained MC acquisition objective and constraints shape-aligned.

    BoTorch's constrained MC acquisitions compute feasibility indicators before
    their q-reduction. With one-to-many InputPerturbation, posterior samples have
    q_like = q * n_w. Therefore both the objective values and the constraint
    values must remain on q_like at the constraint-weighting step. Aggregating the
    objective or the constraints to q at this stage causes q versus q*n_w shape
    errors inside BoTorch.
    """

    if bundle is None or config.constraints is None or config.objective_config is None:
        return config
    n_w = _engine._input_transform_n_w_from_bundle(bundle, output=config.objective_config.output)
    if n_w is None or int(n_w) <= 1:
        return config
    if config.objective_config.risk_type is not None:
        return config
    if config.objective_config.aggregate_mean_when_no_risk is False:
        return config
    if "aggregate_mean_when_no_risk" in config.objective_config.objective_kwargs:
        return config

    return replace(
        config,
        objective_config=replace(
            config.objective_config,
            aggregate_mean_when_no_risk=False,
        ),
    )


def _resolve_objective(
    *,
    acq_config: AcquisitionConfig,
    bundle: ModelBundle | None,
) -> AcquisitionConfig:
    resolved = _BASE_RESOLVE_OBJECTIVE(
        acq_config=acq_config,
        bundle=bundle,
    )
    resolved = _keep_constrained_perturbation_q_expanded(bundle=bundle, config=resolved)
    if (
        bundle is None
        or str(bundle.task_type) != "multiclass"
        or resolved.objective is not None
        or resolved.objective_factory is not None
        or resolved.objective_config is not None
    ):
        return resolved

    if not _is_multi_output(bundle) or not _is_vector_strategy(resolved):
        return resolved

    # A multiclass vector strategy always needs a class-reducing objective,
    # regardless of whether InputPerturbation is enabled. Without it, NSGA-II
    # receives raw probabilities shaped (..., q, m, C) instead of objective
    # values shaped (..., q, m). n_w is only needed for perturbation aggregation.
    n_w = _engine._input_transform_n_w_from_bundle(bundle)
    return replace(
        resolved,
        objective_config=ObjectiveConfig(
            mode="multi_output",
            n_w=n_w,
            risk_type=None,
        ),
    )


def apply_classification_perturbation_defaults() -> None:
    """Register the compatibility routes once."""

    global _APPLIED
    if _APPLIED:
        return

    from .hetero_ordinal_perturbation_compat import (
        apply_hetero_ordinal_perturbation_compat,
    )

    _factory._build_ordinal_objective = _build_ordinal
    _factory.build_objective = _build_objective
    _engine._resolve_objective_config_n_w_from_input_transform = _resolve_objective
    _engine_defaults._resolve_objective_config_n_w_from_input_transform = _resolve_objective
    _engine_defaults._resolve_default_regression_nparego_class = (
        _resolve_hybrid_nparego_class
    )
    apply_hetero_ordinal_perturbation_compat()
    _APPLIED = True


__all__ = ["apply_classification_perturbation_defaults"]
