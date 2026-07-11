"""Observed-baseline support for multiclass qNParEGO."""

from __future__ import annotations

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
        baseline = kwargs.get("Y_baseline")
        raw_train_Y = _wide_training_labels(model)
        if _shares_training_labels(baseline, raw_train_Y):
            complete_train_Y = _complete_label_rows(raw_train_Y)
            kwargs["Y_baseline"] = module.compute_observed_multiclass_utility(
                train_Y=complete_train_Y,
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


__all__ = ["configure_multiclass_nparego_observed_baseline"]
