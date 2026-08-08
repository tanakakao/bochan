"""Automatic defaults for information-theoretic and look-ahead BO.

This module connects BoTorch KG, MES, JES, MO-MES, MO-JES, and HVKG to
bochan's high-level API without duplicating the acquisition algorithms
 themselves. Where BoTorch exposes a registered acquisition input constructor,
bochan delegates auxiliary-input construction to it. Multi-objective entropy
search uses BoTorch's public ``sample_optimal_points`` and
``compute_sample_box_decomposition`` utilities because these acquisitions do
not currently have registered input constructors.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .automatic_default_utils import _num_outputs
from .automatic_multiobjective import (
    make_default_ref_point,
    observed_multiobjective_values,
)
from .configs import AcquisitionConfig, DataContext, ModelBundle, OptimizeConfig


def _normalize_name(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def information_acquisition_kind(config: AcquisitionConfig) -> str | None:
    """Return the supported information / look-ahead acquisition family."""

    name = _normalize_name(config.name)
    cls_name = _normalize_name(getattr(config.acqf_cls, "__name__", ""))
    if cls_name == "qlowerboundmultiobjectivemaxvalueentropysearch" or name in {
        "momes",
        "qmomes",
        "mesmo",
        "qmesmo",
        "multiobjectivemes",
        "qmultiobjectivemes",
    }:
        return "mo_mes"
    if cls_name == "qlowerboundmultiobjectivejointentropysearch" or name in {
        "mojes",
        "qmojes",
        "multiobjectivejes",
        "qmultiobjectivejes",
    }:
        return "mo_jes"
    if cls_name == "qhypervolumeknowledgegradient" or name in {"hvkg", "qhvkg"}:
        return "hvkg"
    if cls_name == "qknowledgegradient" or name in {
        "kg",
        "qkg",
        "knowledgegradient",
        "qknowledgegradient",
    }:
        return "kg"
    if cls_name == "qjointentropysearch" or name in {"jes", "qjes"}:
        return "jes"
    if cls_name == "qmaxvalueentropy" or name in {"mes", "qmes"}:
        return "mes"
    return None


def is_information_acquisition(config: AcquisitionConfig) -> bool:
    """Return whether ``config`` selects a supported information acquisition."""

    return information_acquisition_kind(config) is not None


def _require_supported_task(bundle: ModelBundle, kind: str) -> None:
    task = str(bundle.task_type)
    if task not in {"regression", "multi_objective", "hybrid"}:
        raise ValueError(
            f"{kind.upper()} is exposed by the high-level API only for regression / "
            f"multi-objective / hybrid models. Current task_type={task!r}."
        )


def _require_native_multiobjective_task(bundle: ModelBundle, kind: str) -> None:
    task = str(bundle.task_type)
    if task not in {"regression", "multi_objective"}:
        raise ValueError(
            f"{kind.upper()} requires homogeneous Gaussian regression objectives. "
            f"Current task_type={task!r}."
        )
    if _num_outputs(bundle.train_Y) < 2:
        raise ValueError(
            f"{kind.upper()} requires a multi-output model with at least two objectives."
        )


def _has_configured_objective(config: AcquisitionConfig) -> bool:
    return bool(
        config.objective is not None
        or config.objective_factory is not None
        or config.objective_config is not None
    )


def _require_posterior_transform_for_multi_output(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    kind: str,
) -> None:
    if _num_outputs(bundle.train_Y) < 2:
        return
    if config.acqf_kwargs.get("posterior_transform") is None:
        raise ValueError(
            f"{kind.upper()} requires a scalar posterior. For a multi-output model, "
            "pass an explicit posterior_transform in AcquisitionConfig.acqf_kwargs."
        )


def _require_scalar_kg_objective(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    objective: Any,
) -> None:
    """Require an explicit scalarization route for multi-output KG."""

    if _num_outputs(bundle.train_Y) < 2:
        return
    posterior_transform = config.acqf_kwargs.get("posterior_transform")
    if objective is None and posterior_transform is None:
        raise ValueError(
            "KG requires a scalar terminal objective. For a multi-output model, "
            "configure AcquisitionConfig.objective/objective_config or pass an "
            "explicit posterior_transform in AcquisitionConfig.acqf_kwargs."
        )


def _bounds_as_pairs(bounds: Any) -> list[tuple[float, float]]:
    """Convert bochan ``2 x d`` / ``d x 2`` bounds to BoTorch pair format."""

    import torch

    value = torch.as_tensor(bounds).detach().cpu()
    if value.ndim != 2:
        raise ValueError(
            f"bounds must be a 2D array/tensor. Got shape={tuple(value.shape)}."
        )
    if value.shape[0] == 2:
        value = value.transpose(0, 1)
    elif value.shape[1] != 2:
        raise ValueError(
            "bounds must have shape [2, d] or [d, 2]. "
            f"Got shape={tuple(value.shape)}."
        )
    return [(float(row[0]), float(row[1])) for row in value]


def _bounds_as_tensor(bounds: Any, reference: Any) -> Any:
    """Convert bounds to BoTorch's ``2 x d`` tensor on the model data device."""

    import torch

    dtype = getattr(reference, "dtype", None)
    device = getattr(reference, "device", None)
    value = torch.as_tensor(bounds, dtype=dtype, device=device)
    if value.ndim != 2:
        raise ValueError(
            f"bounds must be a 2D array/tensor. Got shape={tuple(value.shape)}."
        )
    if value.shape[0] == 2:
        return value
    if value.shape[1] == 2:
        return value.transpose(0, 1)
    raise ValueError(
        "bounds must have shape [2, d] or [d, 2]. "
        f"Got shape={tuple(value.shape)}."
    )


def _training_dataset(bundle: ModelBundle) -> Any:
    """Adapt bochan training tensors to BoTorch's metadata-aware dataset API."""

    from botorch.utils.datasets import SupervisedDataset

    n_features = int(bundle.train_X.shape[-1])
    train_y = bundle.train_Y
    n_outputs = int(train_y.shape[-1]) if train_y.ndim > 1 else 1
    return SupervisedDataset(
        X=bundle.train_X,
        Y=train_y,
        feature_names=[f"x_{idx}" for idx in range(n_features)],
        outcome_names=[f"y_{idx}" for idx in range(n_outputs)],
    )


def _get_botorch_input_constructor(acqf_cls: type) -> Any:
    """Small test seam around BoTorch's public input-constructor registry."""

    from botorch.acquisition.input_constructors import get_acqf_input_constructor

    return get_acqf_input_constructor(acqf_cls)


def _sample_multiobjective_optimal_points(
    *,
    model: Any,
    bounds: Any,
    num_samples: int,
    num_points: int,
    maximize: bool,
    optimizer: Any = None,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Sample Pareto sets/fronts through BoTorch's public RFF utility."""

    from botorch.acquisition.multi_objective.utils import (
        random_search_optimizer,
        sample_optimal_points,
    )

    return sample_optimal_points(
        model=model,
        bounds=bounds,
        num_samples=num_samples,
        num_points=num_points,
        optimizer=random_search_optimizer if optimizer is None else optimizer,
        maximize=maximize,
        optimizer_kwargs=dict(optimizer_kwargs or {}),
    )


def _compute_multiobjective_hypercell_bounds(
    pareto_fronts: Any,
    *,
    maximize: bool,
) -> Any:
    """Compute entropy-search integration boxes with BoTorch's public utility."""

    from botorch.acquisition.multi_objective.utils import (
        compute_sample_box_decomposition,
    )

    return compute_sample_box_decomposition(
        pareto_fronts=pareto_fronts,
        maximize=maximize,
    )


def _merge_generated_inputs(
    config: AcquisitionConfig,
    generated: dict[str, Any],
) -> AcquisitionConfig:
    """Merge generated constructor inputs while preserving explicit user values."""

    kwargs = {key: value for key, value in generated.items() if key != "model"}
    kwargs.update(config.acqf_kwargs)
    return replace(config, acqf_kwargs=kwargs)


def _resolve_kg(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    """Resolve qKnowledgeGradient terminal value and scalar objective semantics."""

    _require_supported_task(bundle, "kg")

    from .factory import build_objective

    objective = build_objective(bundle=bundle, config=config, data_context=context)
    _require_scalar_kg_objective(bundle, config, objective)
    if objective is not None and config.objective is None:
        config = replace(config, objective=objective)

    kwargs = dict(config.acqf_kwargs)
    kwargs.setdefault(
        "num_fantasies",
        int(context.extra.get("kg_num_fantasies", 64)),
    )
    config = replace(config, acqf_kwargs=kwargs)

    if config.acqf_kwargs.get("current_value") is not None:
        return config, context
    if context.bounds is None:
        raise ValueError(
            "KG requires bounds when current_value is not supplied explicitly."
        )
    if context.X_pending is not None:
        raise ValueError(
            "Automatic KG current_value does not condition on X_pending. When pending "
            "points are present, supply AcquisitionConfig.acqf_kwargs['current_value'] "
            "explicitly using the pending-conditioned terminal value."
        )

    constructor = _get_botorch_input_constructor(config.acqf_cls)
    generated = constructor(
        model=bundle.model,
        training_data=_training_dataset(bundle),
        bounds=_bounds_as_pairs(context.bounds),
        objective=objective,
        posterior_transform=config.acqf_kwargs.get("posterior_transform"),
        num_fantasies=int(config.acqf_kwargs["num_fantasies"]),
        with_current_value=True,
    )
    return _merge_generated_inputs(config, generated), context


def _resolve_mes(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    _require_supported_task(bundle, "mes")
    if _has_configured_objective(config):
        raise ValueError(
            "MES does not consume AcquisitionConfig.objective/objective_config. "
            "Use maximize for direction and posterior_transform for output scalarization."
        )
    _require_posterior_transform_for_multi_output(bundle, config, "mes")

    if config.acqf_kwargs.get("candidate_set") is not None:
        return config, context
    if context.bounds is None:
        raise ValueError(
            "MES requires bounds when candidate_set is not supplied explicitly."
        )

    constructor = _get_botorch_input_constructor(config.acqf_cls)
    generated = constructor(
        model=bundle.model,
        training_data=_training_dataset(bundle),
        bounds=_bounds_as_pairs(context.bounds),
        posterior_transform=config.acqf_kwargs.get("posterior_transform"),
        candidate_size=int(context.extra.get("mes_candidate_size", 1000)),
        maximize=bool(config.acqf_kwargs.get("maximize", True)),
    )
    return _merge_generated_inputs(config, generated), context


def _resolve_jes(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    _require_supported_task(bundle, "jes")
    if _has_configured_objective(config):
        raise ValueError(
            "JES does not consume AcquisitionConfig.objective/objective_config. "
            "Use posterior_transform when a scalarized posterior is required."
        )
    _require_posterior_transform_for_multi_output(bundle, config, "jes")

    optimal_inputs = config.acqf_kwargs.get("optimal_inputs")
    optimal_outputs = config.acqf_kwargs.get("optimal_outputs")
    if (optimal_inputs is None) != (optimal_outputs is None):
        raise ValueError(
            "JES requires optimal_inputs and optimal_outputs to be supplied together."
        )
    if optimal_inputs is not None:
        return config, context
    if context.bounds is None:
        raise ValueError(
            "JES requires bounds when optimal samples are not supplied explicitly."
        )

    constructor = _get_botorch_input_constructor(config.acqf_cls)
    generated = constructor(
        model=bundle.model,
        bounds=_bounds_as_pairs(context.bounds),
        num_optima=int(context.extra.get("jes_num_optima", 64)),
        condition_noiseless=bool(
            config.acqf_kwargs.get("condition_noiseless", True)
        ),
        posterior_transform=config.acqf_kwargs.get("posterior_transform"),
        X_pending=context.X_pending,
        estimation_type=str(config.acqf_kwargs.get("estimation_type", "LB")),
        num_samples=int(config.acqf_kwargs.get("num_samples", 64)),
    )
    return _merge_generated_inputs(config, generated), context


def _prepare_native_multiobjective_entropy_context(
    config: AcquisitionConfig,
    context: DataContext,
    kind: str,
) -> DataContext:
    """Keep MO entropy search in native Pareto space without scalarization."""

    if _has_configured_objective(config):
        raise ValueError(
            f"{kind.upper()} operates directly on all model outputs as Pareto "
            "objectives and does not accept AcquisitionConfig.objective/objective_config."
        )
    if config.acqf_kwargs.get("posterior_transform") is not None:
        raise ValueError(
            f"{kind.upper()} does not accept posterior_transform. Use native model "
            "outputs as objectives, or use scalar MES/JES after explicit scalarization."
        )
    if config.sampler is not None:
        raise ValueError(
            f"{kind.upper()} manages its entropy Monte Carlo samples internally via "
            "num_samples and does not accept AcquisitionConfig.sampler."
        )

    mo_config = context.multi_objective
    if mo_config is not None:
        if mo_config.objective is not None:
            raise ValueError(
                f"{kind.upper()} does not consume MultiObjectiveConfig.objective. "
                "The model outputs themselves define the Pareto objectives."
            )
        if mo_config.auto_scalarization and mo_config.scalarization_weights is not None:
            raise ValueError(
                f"{kind.upper()} must not use MultiObjectiveConfig scalarization. "
                "Disable scalarization or use scalar MES/JES instead."
            )
        if mo_config.constraints is not None:
            raise ValueError(
                f"Automatic constrained {kind.upper()} is not implemented in this "
                "high-level path. Supply an unconstrained objective model."
            )
        if mo_config.auto_scalarization:
            context = replace(
                context,
                multi_objective=replace(mo_config, auto_scalarization=False),
            )

    if context.constraints is not None:
        raise ValueError(
            f"Automatic constrained {kind.upper()} is not implemented in this "
            "high-level path."
        )
    return context


def _mo_entropy_settings(
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, bool, int, int, Any, dict[str, Any]]:
    """Resolve shared BoTorch Pareto-sampling and entropy-estimator settings."""

    kwargs = dict(config.acqf_kwargs)
    estimation_type = str(
        kwargs.get(
            "estimation_type",
            context.extra.get("mo_entropy_estimation_type", "LB"),
        )
    )
    if estimation_type not in {"0", "LB", "LB2", "MC"}:
        raise ValueError(
            "MO entropy estimation_type must be one of '0', 'LB', 'LB2', or 'MC'."
        )
    num_samples = int(
        kwargs.get("num_samples", context.extra.get("mo_entropy_num_samples", 64))
    )
    if num_samples <= 0:
        raise ValueError("MO entropy num_samples must be positive.")
    kwargs.setdefault("estimation_type", estimation_type)
    kwargs.setdefault("num_samples", num_samples)

    num_pareto_samples = int(context.extra.get("mo_entropy_num_pareto_samples", 8))
    num_pareto_points = int(context.extra.get("mo_entropy_num_pareto_points", 8))
    if num_pareto_samples <= 0 or num_pareto_points <= 0:
        raise ValueError(
            "mo_entropy_num_pareto_samples and mo_entropy_num_pareto_points must "
            "both be positive."
        )
    maximize = bool(context.extra.get("mo_entropy_maximize", True))
    optimizer = context.extra.get("mo_entropy_optimizer")
    optimizer_kwargs = dict(context.extra.get("mo_entropy_optimizer_kwargs", {}) or {})
    return (
        replace(config, acqf_kwargs=kwargs),
        maximize,
        num_pareto_samples,
        num_pareto_points,
        optimizer,
        optimizer_kwargs,
    )


def _require_continuous_auto_pareto_sampling(
    bundle: ModelBundle,
    kind: str,
) -> None:
    if bundle.cat_dims or str(bundle.input_type) == "mixed":
        raise ValueError(
            f"Automatic {kind.upper()} Pareto sampling uses BoTorch "
            "sample_optimal_points over continuous bounds. For mixed/categorical "
            "inputs, supply the required Pareto samples / hypercell bounds explicitly."
        )


def _generate_multiobjective_pareto_samples(
    bundle: ModelBundle,
    context: DataContext,
    *,
    kind: str,
    maximize: bool,
    num_pareto_samples: int,
    num_pareto_points: int,
    optimizer: Any,
    optimizer_kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    if context.bounds is None:
        raise ValueError(
            f"{kind.upper()} requires bounds when Pareto samples are not supplied "
            "explicitly."
        )
    _require_continuous_auto_pareto_sampling(bundle, kind)
    bounds = _bounds_as_tensor(context.bounds, bundle.train_X)
    return _sample_multiobjective_optimal_points(
        model=bundle.model,
        bounds=bounds,
        num_samples=num_pareto_samples,
        num_points=num_pareto_points,
        maximize=maximize,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
    )


def _resolve_mo_mes(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    """Resolve native Pareto multi-objective Max-value Entropy Search."""

    _require_native_multiobjective_task(bundle, "mo-mes")
    context = _prepare_native_multiobjective_entropy_context(config, context, "mo-mes")
    (
        config,
        maximize,
        num_pareto_samples,
        num_pareto_points,
        optimizer,
        optimizer_kwargs,
    ) = _mo_entropy_settings(config, context)

    if config.acqf_kwargs.get("hypercell_bounds") is not None:
        return config, context

    _, pareto_fronts = _generate_multiobjective_pareto_samples(
        bundle,
        context,
        kind="mo-mes",
        maximize=maximize,
        num_pareto_samples=num_pareto_samples,
        num_pareto_points=num_pareto_points,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
    )
    hypercell_bounds = _compute_multiobjective_hypercell_bounds(
        pareto_fronts,
        maximize=maximize,
    )
    kwargs = dict(config.acqf_kwargs)
    kwargs["hypercell_bounds"] = hypercell_bounds
    return replace(config, acqf_kwargs=kwargs), context


def _resolve_mo_jes(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    """Resolve native Pareto multi-objective Joint Entropy Search."""

    _require_native_multiobjective_task(bundle, "mo-jes")
    context = _prepare_native_multiobjective_entropy_context(config, context, "mo-jes")
    (
        config,
        maximize,
        num_pareto_samples,
        num_pareto_points,
        optimizer,
        optimizer_kwargs,
    ) = _mo_entropy_settings(config, context)

    kwargs = dict(config.acqf_kwargs)
    pareto_sets = kwargs.get("pareto_sets")
    pareto_fronts = kwargs.get("pareto_fronts")
    hypercell_bounds = kwargs.get("hypercell_bounds")

    if (pareto_sets is None) != (pareto_fronts is None):
        raise ValueError(
            "MO-JES requires pareto_sets and pareto_fronts to be supplied together."
        )
    if pareto_sets is None and hypercell_bounds is not None:
        raise ValueError(
            "MO-JES hypercell_bounds cannot be supplied without the matching "
            "pareto_sets and pareto_fronts."
        )

    if pareto_sets is None:
        pareto_sets, pareto_fronts = _generate_multiobjective_pareto_samples(
            bundle,
            context,
            kind="mo-jes",
            maximize=maximize,
            num_pareto_samples=num_pareto_samples,
            num_pareto_points=num_pareto_points,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        kwargs["pareto_sets"] = pareto_sets
        kwargs["pareto_fronts"] = pareto_fronts

    if hypercell_bounds is None:
        hypercell_bounds = _compute_multiobjective_hypercell_bounds(
            pareto_fronts,
            maximize=maximize,
        )
        kwargs["hypercell_bounds"] = hypercell_bounds

    return replace(config, acqf_kwargs=kwargs), context


def _prepare_hvkg_multiobjective_context(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    """Prepare MO context without applying generic scalarization to HVKG."""

    from .factory import prepare_multi_objective_context

    mo_config = context.multi_objective
    if mo_config is not None and mo_config.auto_scalarization:
        context = replace(
            context,
            multi_objective=replace(mo_config, auto_scalarization=False),
        )
    context = prepare_multi_objective_context(bundle, context, config)
    return config, context


def _resolve_hvkg_ref_point(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext, Any]:
    explicit = config.acqf_kwargs.get("ref_point")
    if explicit is not None:
        context.ref_point = None
        return config, context, explicit

    ref_point = context.ref_point
    if ref_point is None:
        values = observed_multiobjective_values(bundle, config, context)
        margin = float(context.extra.get("ref_point_margin", 0.1))
        ref_point = make_default_ref_point(values, margin=margin)
        context.ref_point = ref_point
    return config, context, ref_point


def _resolve_hvkg(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    _require_supported_task(bundle, "hvkg")
    if _num_outputs(bundle.train_Y) < 2:
        raise ValueError(
            "HVKG requires a multi-output model with at least two objectives."
        )

    config, context = _prepare_hvkg_multiobjective_context(bundle, config, context)
    config, context, ref_point = _resolve_hvkg_ref_point(bundle, config, context)

    if config.acqf_kwargs.get("current_value") is not None:
        return config, context
    if context.bounds is None:
        raise ValueError(
            "HVKG requires bounds when current_value is not supplied explicitly."
        )

    from .factory import build_objective

    objective = build_objective(bundle=bundle, config=config, data_context=context)
    if objective is not None and config.objective is None:
        config = replace(config, objective=objective)

    constructor = _get_botorch_input_constructor(config.acqf_cls)
    generated = constructor(
        model=bundle.model,
        training_data=_training_dataset(bundle),
        bounds=_bounds_as_pairs(context.bounds),
        objective_thresholds=None,
        objective=objective,
        posterior_transform=config.acqf_kwargs.get("posterior_transform"),
        num_fantasies=int(config.acqf_kwargs.get("num_fantasies", 8)),
        num_pareto=int(config.acqf_kwargs.get("num_pareto", 10)),
        ref_point=ref_point,
    )
    return _merge_generated_inputs(config, generated), context


def resolve_information_acquisition_defaults(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    """Resolve auxiliary inputs for KG/MES/JES/MO-MES/MO-JES/HVKG."""

    kind = information_acquisition_kind(config)
    if kind == "kg":
        return _resolve_kg(bundle, config, context)
    if kind == "mes":
        return _resolve_mes(bundle, config, context)
    if kind == "jes":
        return _resolve_jes(bundle, config, context)
    if kind == "mo_mes":
        return _resolve_mo_mes(bundle, config, context)
    if kind == "mo_jes":
        return _resolve_mo_jes(bundle, config, context)
    if kind == "hvkg":
        return _resolve_hvkg(bundle, config, context)
    return config, context


def resolve_information_optimizer_defaults(
    config: AcquisitionConfig,
    opt_config: OptimizeConfig,
) -> OptimizeConfig:
    """Apply optimizer settings required by entropy search and one-shot KG/HVKG.

    BoTorch qMES uses sequential or cyclic optimization for q > 1. BoTorch's
    multi-objective entropy-search tutorial likewise uses sequential greedy
    optimization for q > 1 because the lower-bound batch criterion need not be
    monotone. The bochan high-level API therefore enables sequential
    optimization for these cases. KG and HVKG are one-shot acquisitions and
    must remain joint; an explicit sequential setting is normalized to False.
    """

    kind = information_acquisition_kind(config)
    if (
        kind in {"mes", "mo_mes", "mo_jes"}
        and opt_config.q > 1
        and not opt_config.sequential
    ):
        return replace(opt_config, sequential=True)
    if kind in {"kg", "hvkg"} and opt_config.sequential:
        return replace(opt_config, sequential=False)
    return opt_config


__all__ = [
    "information_acquisition_kind",
    "is_information_acquisition",
    "resolve_information_acquisition_defaults",
    "resolve_information_optimizer_defaults",
]
