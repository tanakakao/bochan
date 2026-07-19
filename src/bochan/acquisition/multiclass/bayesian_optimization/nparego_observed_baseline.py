"""Observed-baseline support for multiclass qNParEGO."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import torch
from torch import Tensor


def _shares_training_labels(value: Any, train_Y: Any) -> bool:
    """Return whether a baseline is the model's original wide label tensor."""

    if value is None or train_Y is None:
        return False
    if value is train_Y:
        return True
    if not torch.is_tensor(value) or not torch.is_tensor(train_Y):
        return False
    return (
        value.shape == train_Y.shape
        and value.device == train_Y.device
        and value.data_ptr() == train_Y.data_ptr()
    )


def _wide_training_labels(model: Any) -> Tensor | None:
    """Return wide multiclass labels when the model retains them."""

    for name in ("train_Y_wide", "train_Y", "train_targets"):
        value = getattr(model, name, None)
        if torch.is_tensor(value) and value.ndim == 2:
            return value
    return None


def _complete_label_rows(train_Y: Tensor) -> Tensor:
    """Keep rows with every multiclass output observed."""

    train_Y = torch.as_tensor(train_Y)
    if train_Y.ndim != 2:
        raise ValueError(
            "Multiclass NParEGO baseline labels must have shape [n, m]. "
            f"Got shape={tuple(train_Y.shape)}."
        )
    finite = torch.isfinite(train_Y)
    complete = finite.all(dim=-1)
    if bool(complete.any()):
        return train_Y[complete]

    observed_counts = finite.sum(dim=0).detach().cpu().tolist()
    raise ValueError(
        "Multiclass NParEGO requires at least one training row with every "
        "output observed to construct its objective-space baseline. Partially "
        f"observed rows remain usable for model fitting. Observed counts per "
        f"output={observed_counts}."
    )


def _observed_objective_baseline(
    module: Any,
    *,
    model: Any,
    baseline: Tensor | None,
    target_class: int | Sequence[int] | None,
    output_target_classes: Sequence[int] | None,
    class_reduction: str,
    utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor | None,
    objective_signs: Sequence[float] | Tensor | None,
    class_offset: int,
) -> Tensor | None:
    """Convert an automatically injected raw-label baseline to objective values."""

    raw_train_Y = _wide_training_labels(model)
    if not _shares_training_labels(baseline, raw_train_Y):
        return baseline

    complete_train_Y = _complete_label_rows(raw_train_Y)
    return module.compute_observed_multiclass_utility(
        train_Y=complete_train_Y,
        target_class=target_class,
        output_target_classes=output_target_classes,
        class_reduction=class_reduction,
        utility_values=utility_values,
        objective_signs=objective_signs,
        class_offset=class_offset,
    )


def configure_multiclass_nparego_observed_baseline(module: Any) -> None:
    """Convert an automatically injected raw label baseline to objective values."""

    acquisition_cls = module.qMultiOutputMulticlassNParEGO
    if getattr(acquisition_cls, "_bochan_observed_baseline_patched", False):
        return

    original_init = acquisition_cls.__init__

    @wraps(original_init)
    def supported_init(
        self,
        model,
        X_baseline: Tensor,
        ref_point,
        *args,
        **kwargs,
    ) -> None:
        kwargs["Y_baseline"] = _observed_objective_baseline(
            module,
            model=model,
            baseline=kwargs.get("Y_baseline"),
            target_class=kwargs.get("target_class"),
            output_target_classes=kwargs.get("output_target_classes"),
            class_reduction=kwargs.get("class_reduction", "mean"),
            utility_values=kwargs.get("utility_values"),
            objective_signs=kwargs.get("objective_signs"),
            class_offset=kwargs.get("class_offset", 0),
        )

        original_init(
            self,
            model,
            X_baseline,
            ref_point,
            *args,
            **kwargs,
        )

    acquisition_cls.__init__ = supported_init
    acquisition_cls._bochan_observed_baseline_patched = True
    acquisition_cls._bochan_original_init_before_observed_baseline = original_init


def configure_hetero_multiclass_nparego_observed_baseline(
    module: Any,
    acquisition_cls: type,
) -> None:
    """Add observed-baseline support to hetero multiclass qNParEGO.

    The heteroscedastic subclass historically omitted ``Y_baseline`` and
    ``train_Y`` from its public constructor even though the high-level API
    injects ``Y_baseline`` for all internal NParEGO implementations. This
    wrapper exposes the complete baseline-related signature, preserves utility
    configuration, and replaces the posterior-derived baseline score with the
    observed objective-space score when one is available.
    """

    if getattr(acquisition_cls, "_bochan_hetero_observed_baseline_patched", False):
        return

    original_init = acquisition_cls.__init__

    def supported_init(
        self,
        model,
        X_baseline: Tensor,
        ref_point,
        *,
        target_class: int | Sequence[int] | None = None,
        output_target_classes: Sequence[int] | None = None,
        class_reduction: str = "mean",
        utility_values: Sequence[Sequence[float]] | Sequence[float] | Tensor | None = None,
        objective_signs: Sequence[float] | Tensor | None = None,
        train_Y: Tensor | None = None,
        Y_baseline: Tensor | None = None,
        class_offset: int = 0,
        weights: Tensor | None = None,
        sampler: Any = None,
        objective: Any = None,
        rho: float = 0.05,
        noise_mode: str = "inverse_linear",
        noise_combine: str = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        eps: float = 1e-8,
    ) -> None:
        if objective is None:
            objective = module.MulticlassTargetProbabilityObjective(
                target_class=target_class,
                output_target_classes=output_target_classes,
                num_outputs=int(torch.as_tensor(ref_point).numel()),
                class_reduction=class_reduction,
                utility_values=utility_values,
                objective_signs=objective_signs,
                eps=eps,
            )

        baseline = Y_baseline
        if baseline is None and train_Y is not None:
            baseline = module.compute_observed_multiclass_utility(
                train_Y=_complete_label_rows(train_Y),
                target_class=target_class,
                output_target_classes=output_target_classes,
                class_reduction=class_reduction,
                utility_values=utility_values,
                objective_signs=objective_signs,
                class_offset=class_offset,
            )
        else:
            baseline = _observed_objective_baseline(
                module,
                model=model,
                baseline=baseline,
                target_class=target_class,
                output_target_classes=output_target_classes,
                class_reduction=class_reduction,
                utility_values=utility_values,
                objective_signs=objective_signs,
                class_offset=class_offset,
            )

        original_init(
            self,
            model=model,
            X_baseline=X_baseline,
            ref_point=ref_point,
            target_class=target_class,
            output_target_classes=output_target_classes,
            class_reduction=class_reduction,
            weights=weights,
            sampler=sampler,
            objective=objective,
            rho=rho,
            noise_mode=noise_mode,
            noise_combine=noise_combine,
            noise_penalty_lambda=noise_penalty_lambda,
            noise_min_weight=noise_min_weight,
            noise_weight_scale=noise_weight_scale,
            noise_model_outputs_log_var=noise_model_outputs_log_var,
            eps=eps,
        )

        if baseline is not None:
            values = torch.as_tensor(
                baseline,
                device=X_baseline.device,
                dtype=X_baseline.dtype,
            ).unsqueeze(0).unsqueeze(0)
            baseline_score = self._scalarize(values)
            if baseline_score.ndim >= 2 and baseline_score.shape[-1] == 1:
                baseline_score = baseline_score.squeeze(-1)
            best_value = baseline_score.max()
            if torch.is_tensor(getattr(self, "best_value", None)):
                self.best_value.copy_(best_value.to(self.best_value))
            else:
                self.register_buffer("best_value", best_value)

    supported_init.__name__ = original_init.__name__
    supported_init.__qualname__ = original_init.__qualname__
    supported_init.__doc__ = original_init.__doc__

    acquisition_cls.__init__ = supported_init
    acquisition_cls._bochan_hetero_observed_baseline_patched = True
    acquisition_cls._bochan_original_init_before_hetero_observed_baseline = original_init


__all__ = [
    "configure_hetero_multiclass_nparego_observed_baseline",
    "configure_multiclass_nparego_observed_baseline",
]
