from __future__ import annotations

"""Compatibility adapter for robust ordinal ``num_classes`` inference."""

import inspect
from functools import wraps
from typing import Optional, TypeVar

from torch import Tensor

from bochan.models.ordinal.base.models import (
    _BaseOrdinalGPModel,
    _infer_num_classes_from_train_Y,
)


T = TypeVar("T", bound=type)


def _resolve_num_classes(
    *,
    train_X: Tensor,
    train_Y: Tensor,
    num_classes: Optional[int],
) -> int:
    """Resolve an explicit class count or infer it from canonical ordinal labels."""
    raw_train_X = _BaseOrdinalGPModel._canonicalize_train_X(train_X)
    canonical_train_Y = _BaseOrdinalGPModel._canonicalize_train_Y(
        train_Y,
        n=raw_train_X.shape[-2],
        device=raw_train_X.device,
    )
    if num_classes is None:
        return _infer_num_classes_from_train_Y(canonical_train_Y)
    return int(num_classes)


def enable_num_classes_inference(model_cls: T) -> T:
    """Allow a robust ordinal model class to infer ``num_classes`` from ``train_Y``.

    The adapter updates the existing class object rather than introducing a public
    subclass. This preserves module paths, ``isinstance`` behavior, rebuilding via
    ``self.__class__``, and saved-model compatibility.
    """
    if getattr(model_cls, "_num_classes_inference_enabled", False):
        return model_cls

    original_init = model_cls.__init__
    original_signature = inspect.signature(original_init)

    @wraps(original_init)
    def wrapped_init(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *args,
        num_classes: Optional[int] = None,
        **kwargs,
    ) -> None:
        resolved_num_classes = _resolve_num_classes(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
        )
        original_init(
            self,
            train_X,
            train_Y,
            *args,
            num_classes=resolved_num_classes,
            **kwargs,
        )

    parameters = [
        parameter.replace(default=None, annotation=Optional[int])
        if parameter.name == "num_classes"
        else parameter
        for parameter in original_signature.parameters.values()
    ]
    annotations = dict(getattr(wrapped_init, "__annotations__", {}))
    annotations["num_classes"] = Optional[int]

    setattr(wrapped_init, "__signature__", original_signature.replace(parameters=parameters))
    setattr(wrapped_init, "__annotations__", annotations)
    setattr(model_cls, "__init__", wrapped_init)
    setattr(model_cls, "_num_classes_inference_enabled", True)
    return model_cls


__all__ = ["enable_num_classes_inference"]
