"""Automatic defaults for the high-level Bayesian optimizer."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any, Callable

from .automatic_best_f import compute_best_f
from .automatic_default_utils import _num_outputs
from .automatic_multiobjective import (
    make_default_ref_point,
    make_partitioning,
    observed_multiobjective_values,
)
from .configs import (
    AcquisitionConfig,
    DataContext,
    FitConfig,
    ModelBundle,
    ModelConfig,
    MultiOutputConfig,
    OptimizeConfig,
)
from .engine import BayesianOptimizer as _BaseBayesianOptimizer
from .engine import _resolve_objective_config_n_w_from_input_transform


def _normalize_name(value: Any) -> str:
    """Return a compact lower-case identifier for an acquisition name."""

    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _acquisition_kind(config: AcquisitionConfig) -> str | None:
    """Classify acquisitions that need automatically inferred context values."""

    name = _normalize_name(config.name)
    cls_name = _normalize_name(getattr(config.acqf_cls, "__name__", ""))
    combined = f"{name} {cls_name}"

    if "nparego" in combined:
        return "nparego"
    if (
        name
        in {"nehi", "qnehi", "nehvi", "qnehvi", "noisyexpectedhypervolumeimprovement"}
        or "noisyexpectedhypervolumeimprovement" in cls_name
    ):
        return "nehvi"
    if name in {
        "ehi",
        "qehi",
        "ehvi",
        "qehvi",
        "expectedhypervolumeimprovement",
        "qexpectedhypervolumeimprovement",
    } or (
        "expectedhypervolumeimprovement" in cls_name
        and "noisyexpectedhypervolumeimprovement" not in cls_name
    ):
        return "ehvi"
    if name in {
        "ei",
        "qei",
        "expectedimprovement",
        "qexpectedimprovement",
        "pi",
        "qpi",
        "probabilityofimprovement",
        "qprobabilityofimprovement",
    }:
        return "ei_pi"
    if (
        ("expectedimprovement" in cls_name or "probabilityofimprovement" in cls_name)
        and "hypervolume" not in cls_name
        and "noisyexpectedimprovement" not in cls_name
    ):
        return "ei_pi"
    return None


def resolve_multi_output_model_config(
    model_config: ModelConfig,
    train_Y: Any,
) -> ModelConfig:
    """Resolve automatic wrapping for targets with two or more columns.

    Correlated multi-task models consume wide targets directly and must remain a
    single model rather than being split into a ModelList-style wrapper.
    """

    if _normalize_name(model_config.model_type) in {"kronecker", "multitask"}:
        return model_config
    if model_config.multi_output_config is not None or _num_outputs(train_Y) < 2:
        return model_config
    return replace(model_config, multi_output_config=MultiOutputConfig())


def _resolve_default_regression_nparego_class(
    bundle: ModelBundle,
    config: AcquisitionConfig,
) -> AcquisitionConfig:
    """Use bochan's regression NParEGO implementation for multi-output regression.

    The short aliases ``nparego`` and ``qnparego`` historically resolve to
    BoTorch ``qExpectedImprovement`` and rely on a separately scalarized
    objective. For multi-output regression, default to
    ``qMultiOutputRegressionNParEGO`` instead so a normal
    ``RegressionLinearMCObjective`` can be supplied as the multi-output
    preprocessing objective.

    Explicit canonical acquisition names and non-regression tasks are left
    unchanged.
    """

    if _normalize_name(config.name) not in {"nparego", "qnparego"}:
        return config
    if str(bundle.task_type) != "regression" or _num_outputs(bundle.train_Y) < 2:
        return config

    from bochan.acquisition.regression.bayesian_optimization import (
        qMultiOutputRegressionNParEGO,
    )

    return replace(config, acqf_cls=qMultiOutputRegressionNParEGO)


def _explicit_acqf_value(config: AcquisitionConfig, name: str) -> Any:
    """Return a non-None value explicitly supplied in ``acqf_kwargs``."""

    value = config.acqf_kwargs.get(name)
    return value if value is not None else None


def _keyword_mode(func: Callable[..., Any] | None, name: str) -> str:
    """Return ``explicit``, ``variadic``, or ``unsupported`` for a keyword."""

    if func is None:
        return "explicit"
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return "explicit"
    if name in signature.parameters:
        return "explicit"
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return "variadic"
    return "unsupported"


def _callable_accepts_keyword(func: Callable[..., Any] | None, name: str) -> bool:
    return _keyword_mode(func, name) != "unsupported"


def _place_context_value(
    config: AcquisitionConfig,
    context: DataContext,
    name: str,
    value: Any,
) -> tuple[AcquisitionConfig, DataContext]:
    """Place an inferred value where signature filtering will preserve it."""

    mode = _keyword_mode(config.acqf_cls, name)
    if mode == "unsupported":
        return config, context
    if mode == "variadic" and config.filter_kwargs_by_signature:
        kwargs = dict(config.acqf_kwargs)
        kwargs[name] = value
        setattr(context, name, None)
        return replace(config, acqf_kwargs=kwargs), context
    setattr(context, name, value)
    return config, context


def _is_ordinal_utility_acquisition(config: AcquisitionConfig) -> bool:
    """Return whether the acquisition derives from the ordinal utility BO base."""

    acqf_cls = config.acqf_cls
    if acqf_cls is None:
        return False
    try:
        return any(
            base.__name__ == "_OrdinalPointwiseUtilityBOBase"
            for base in inspect.getmro(acqf_cls)
        )
    except (AttributeError, TypeError):
        return False


def _resolve_default_ordinal_objective(
    bundle: ModelBundle,
    config: AcquisitionConfig,
) -> AcquisitionConfig:
    """Create expected-utility objective for ordinal utility acquisitions.

    Explicit ``objective``, ``objective_factory`` and ``objective_config`` always
    take precedence. Utility values are inferred as ``[0, ..., K - 1]`` from
    ``num_classes`` or the ordinal cutpoints.
    """

    if str(bundle.task_type) != "ordinal":
        return config
    if (
        config.objective is not None
        or config.objective_factory is not None
        or config.objective_config is not None
        or not _is_ordinal_utility_acquisition(config)
    ):
        return config

    from bochan.acquisition.objective import OrdinalExpectedUtilityMCObjective
    from .factory import _infer_ordinal_likelihood, _infer_ordinal_utility_values

    likelihood = _infer_ordinal_likelihood(bundle.model)
    utility_values = _infer_ordinal_utility_values(bundle.model, likelihood)
    objective = OrdinalExpectedUtilityMCObjective(
        ordinal_likelihood=likelihood,
        utility_values=utility_values,
    )
    return replace(config, objective=objective)


def _resolve_default_nparego_objective(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> AcquisitionConfig:
    """Create a random Chebyshev scalarization for default NParEGO use.

    Explicit objective settings always take precedence. The scalarization is
    built from observed objective values and a random simplex weight vector, as
    in the standard NParEGO construction. The same objective is subsequently
    used when inferring ``best_f``.
    """

    if (
        config.objective is not None
        or config.objective_factory is not None
        or config.objective_config is not None
    ):
        return config

    import torch
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

    values = observed_multiobjective_values(bundle, config, context)
    values = torch.as_tensor(values)
    if values.ndim != 2 or values.shape[-1] < 2:
        raise ValueError(
            "NParEGO requires observed values with shape [n, m] and m >= 2. "
            f"Got {tuple(values.shape)}."
        )

    concentration = torch.ones(
        values.shape[-1],
        dtype=values.dtype,
        device=values.device,
    )
    weights = torch.distributions.Dirichlet(concentration).sample()
    scalarization = get_chebyshev_scalarization(weights=weights, Y=values)
    objective = GenericMCObjective(lambda samples, X=None: scalarization(samples))
    return replace(config, objective=objective)


def _resolve_best_f_default(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    """Fill ``best_f`` without overwriting an explicit configuration value."""

    explicit = _explicit_acqf_value(config, "best_f")
    if explicit is not None:
        context.best_f = None
        return config, context

    value = context.best_f
    if value is None and _callable_accepts_keyword(config.acqf_cls, "best_f"):
        value = compute_best_f(bundle, config, context)
    if value is not None:
        config, context = _place_context_value(config, context, "best_f", value)
    return config, context


def resolve_acquisition_defaults(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    """Fill acquisition-specific defaults without overwriting explicit values."""

    from .factory import prepare_multi_objective_context

    config = _resolve_default_regression_nparego_class(bundle, config)
    context = prepare_multi_objective_context(bundle, context, config)
    config = _resolve_default_ordinal_objective(bundle, config)
    kind = _acquisition_kind(config)
    if kind is None:
        return config, context

    if kind == "nparego":
        config = _resolve_default_nparego_objective(bundle, config, context)

    if kind in {"ei_pi", "nparego"}:
        config, context = _resolve_best_f_default(bundle, config, context)
        if kind == "ei_pi":
            return config, context

    explicit_ref = _explicit_acqf_value(config, "ref_point")
    explicit_partitioning = _explicit_acqf_value(config, "partitioning")

    ref_point = explicit_ref if explicit_ref is not None else context.ref_point
    partitioning = (
        explicit_partitioning
        if explicit_partitioning is not None
        else context.partitioning
    )

    needs_ref = ref_point is None and _callable_accepts_keyword(
        config.acqf_cls,
        "ref_point",
    )
    needs_partitioning = (
        kind == "ehvi"
        and partitioning is None
        and _callable_accepts_keyword(config.acqf_cls, "partitioning")
    )

    values = None
    if needs_ref or needs_partitioning:
        values = observed_multiobjective_values(bundle, config, context)
    if needs_ref:
        margin = float(context.extra.get("ref_point_margin", 0.1))
        ref_point = make_default_ref_point(values, margin=margin)
    if needs_partitioning and ref_point is not None:
        partitioning = make_partitioning(ref_point, values)

    if explicit_ref is not None:
        context.ref_point = None
    elif ref_point is not None:
        config, context = _place_context_value(
            config,
            context,
            "ref_point",
            ref_point,
        )

    if explicit_partitioning is not None:
        context.partitioning = None
    elif partitioning is not None:
        config, context = _place_context_value(
            config,
            context,
            "partitioning",
            partitioning,
        )
    return config, context


def resolve_acquisition_data_context(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> DataContext:
    """Compatibility helper returning only the resolved ``DataContext``."""

    _, context = resolve_acquisition_defaults(bundle, config, context)
    return context


class BayesianOptimizer(_BaseBayesianOptimizer):
    """High-level optimizer with model and acquisition defaults inferred from data."""

    def fit(
        self,
        train_X: Any,
        train_Y: Any,
        *,
        model_config: ModelConfig | None = None,
        fit_config: FitConfig | None = None,
    ) -> "BayesianOptimizer":
        resolved_model_config = resolve_multi_output_model_config(
            model_config or self.model_config,
            train_Y,
        )
        super().fit(
            train_X,
            train_Y,
            model_config=resolved_model_config,
            fit_config=fit_config,
        )
        return self

    def _prepare_default_acquisition_context(
        self,
        acq_config: AcquisitionConfig,
        data_context: DataContext | None,
    ) -> tuple[AcquisitionConfig, DataContext]:
        self._check_fitted()
        base_context = self._resolve_data_context(data_context)
        context = replace(base_context, extra=dict(base_context.extra))
        resolved_config = self._resolve_acquisition_config(acq_config)
        resolved_config = _resolve_objective_config_n_w_from_input_transform(
            acq_config=resolved_config,
            bundle=self.bundle,
        )
        resolved_config, context = resolve_acquisition_defaults(
            self.bundle,
            resolved_config,
            context,
        )
        return resolved_config, context

    def acquisition(
        self,
        acq_config: AcquisitionConfig,
        *,
        data_context: DataContext | None = None,
    ) -> Any:
        resolved_config, context = self._prepare_default_acquisition_context(
            acq_config,
            data_context,
        )
        return super().acquisition(resolved_config, data_context=context)

    def candidate(
        self,
        acq_config: AcquisitionConfig,
        opt_config: OptimizeConfig,
        *,
        data_context: DataContext | None = None,
        bounds: Any | None = None,
        return_result: bool = False,
    ) -> Any:
        resolved_config, context = self._prepare_default_acquisition_context(
            acq_config,
            data_context,
        )
        return super().candidate(
            resolved_config,
            opt_config,
            data_context=context,
            bounds=bounds,
            return_result=return_result,
        )
