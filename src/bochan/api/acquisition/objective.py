"""Objective construction for the public acquisition API."""

from __future__ import annotations

from typing import Any

from ..configs import AcquisitionConfig, DataContext, ModelBundle, ObjectiveConfig
from ..support.callables import _filter_kwargs_for_callable


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


__all__ = ["build_objective"]
