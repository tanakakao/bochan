"""Task-aware automatic ``best_f`` calculations."""

from __future__ import annotations

from typing import Any

from .automatic_default_utils import (
    _call_objective,
    _infer_ordinal_utility_values,
    _objective_config_value,
    _sub_bundles,
)
from .automatic_multiobjective import _regression_observed_values
from .configs import AcquisitionConfig, DataContext, ModelBundle


def _compute_binary_best_f(bundle: ModelBundle, config: AcquisitionConfig) -> Any:
    import torch

    from bochan.acquisition.binary.bayesian_optimization import compute_binary_best_f

    kwargs = {
        "apply_sigmoid_if_needed": config.acqf_kwargs.get(
            "apply_sigmoid_if_needed", False
        ),
        "risk_type": config.acqf_kwargs.get(
            "risk_type",
            _objective_config_value(config, "risk_type"),
        ),
        "alpha": config.acqf_kwargs.get(
            "alpha",
            _objective_config_value(config, "alpha", 0.5),
        ),
        "eps": config.acqf_kwargs.get("eps", 1e-6),
        "best_f_margin": config.acqf_kwargs.get("best_f_margin", 1e-4),
        "best_f_quantile": config.acqf_kwargs.get("best_f_quantile"),
    }
    sub_bundles = _sub_bundles(bundle)
    if sub_bundles:
        return torch.stack(
            [
                compute_binary_best_f(sub.model, sub.train_X, **kwargs)
                for sub in sub_bundles
            ]
        )
    return compute_binary_best_f(bundle.model, bundle.train_X, **kwargs)


def _compute_ordinal_best_f(bundle: ModelBundle, config: AcquisitionConfig) -> Any:
    import torch

    from bochan.acquisition.ordinal.bayesian_optimization import (
        compute_ordinal_expected_utility_best_f,
    )

    utility_values = _objective_config_value(config, "utility_values")
    if utility_values is None:
        utility_values = config.acqf_kwargs.get("utility_values")
    if utility_values is None:
        utility_values = _infer_ordinal_utility_values(bundle.model)
    maximize = bool(_objective_config_value(config, "maximize", True))

    sub_bundles = _sub_bundles(bundle)
    if not sub_bundles:
        return compute_ordinal_expected_utility_best_f(
            bundle.model,
            bundle.train_X,
            utility_values=utility_values,
            maximize=maximize,
        )

    utilities = utility_values
    if not (
        isinstance(utilities, (list, tuple))
        and len(utilities) == len(sub_bundles)
        and (isinstance(utilities[0], (list, tuple)) or hasattr(utilities[0], "shape"))
    ):
        utilities = [utilities for _ in sub_bundles]
    return torch.stack(
        [
            compute_ordinal_expected_utility_best_f(
                sub.model,
                sub.train_X,
                utility_values=utilities[i],
                maximize=maximize,
            )
            for i, sub in enumerate(sub_bundles)
        ]
    )


def _compute_multiclass_best_f(bundle: ModelBundle, config: AcquisitionConfig) -> Any:
    import torch

    from bochan.acquisition.multiclass.bayesian_optimization import (
        compute_multiclass_target_probability_best_f,
    )

    target_class = config.acqf_kwargs.get("target_class")
    output_target_classes = config.acqf_kwargs.get("output_target_classes")
    class_reduction = config.acqf_kwargs.get("class_reduction", "mean")
    apply_softmax = config.acqf_kwargs.get("apply_softmax_if_needed", True)
    eps = config.acqf_kwargs.get("eps", 1e-8)

    sub_bundles = _sub_bundles(bundle)
    if sub_bundles:
        if output_target_classes is None:
            targets = [target_class for _ in sub_bundles]
        else:
            targets = list(output_target_classes)
            if len(targets) != len(sub_bundles):
                raise ValueError(
                    "output_target_classes length must match the number of outputs."
                )
        return torch.stack(
            [
                compute_multiclass_target_probability_best_f(
                    sub.model,
                    sub.train_X,
                    target_class=targets[i],
                    class_reduction=class_reduction,
                    apply_softmax_if_needed=apply_softmax,
                    eps=eps,
                )
                for i, sub in enumerate(sub_bundles)
            ]
        )

    return compute_multiclass_target_probability_best_f(
        bundle.model,
        bundle.train_X,
        target_class=target_class,
        class_reduction=class_reduction,
        apply_softmax_if_needed=apply_softmax,
        eps=eps,
    )


def _multifidelity_target_values(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> Any | None:
    """Return observed values at the configured target fidelity when available."""

    if str(bundle.model_type).replace("_", "").replace("-", "").lower() != "multifidelity":
        return None
    target_index = getattr(bundle.model, "target_fidelity_index", None)
    if target_index is None:
        return None

    import torch

    from .factory import build_objective

    values = torch.as_tensor(bundle.train_Y)
    if values.ndim != 2 or int(target_index) >= values.shape[-1]:
        return None
    target_values = values[:, int(target_index) : int(target_index) + 1]
    finite_rows = torch.isfinite(target_values).all(dim=-1)
    values = target_values[finite_rows]
    if values.numel() == 0:
        raise ValueError("No finite observations are available at target_fidelity.")
    objective = build_objective(bundle=bundle, config=config, data_context=context)
    if objective is not None:
        objective_X = bundle.train_X
        if torch.is_tensor(objective_X) and objective_X.shape[-2] == finite_rows.shape[0]:
            objective_X = objective_X[finite_rows]
        values = _call_objective(objective, values, objective_X)
    return values


def _compute_regression_best_f(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> Any:
    values = _multifidelity_target_values(bundle, config, context)
    if values is None:
        values = _regression_observed_values(bundle, config, context)
    import torch

    values = torch.as_tensor(values)
    return values.reshape(-1).max().detach()


def _compute_best_f(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> Any:
    task_type = str(bundle.task_type)
    if task_type == "binary":
        return _compute_binary_best_f(bundle, config)
    if task_type == "ordinal":
        return _compute_ordinal_best_f(bundle, config)
    if task_type == "multiclass":
        return _compute_multiclass_best_f(bundle, config)
    return _compute_regression_best_f(bundle, config, context)


def compute_best_f(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> Any:
    """Compute the EI / PI baseline in the acquisition objective space."""
    return _compute_best_f(bundle, config, context)
