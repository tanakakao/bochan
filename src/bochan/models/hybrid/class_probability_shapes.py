"""Shape compatibility for hybrid ordinal and multiclass probabilities.

DeepGP probability accessors may retain one or more leading likelihood-sample
axes.  Those axes are not model outputs and must not be confused with the
``q`` or output axes when a hybrid model combines heterogeneous submodels.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

_PATCHED = False


def _shape_endswith(shape: torch.Size | tuple[int, ...], suffix: tuple[int, ...]) -> bool:
    if len(suffix) == 0:
        return True
    if len(shape) < len(suffix):
        return False
    return tuple(shape[-len(suffix) :]) == suffix


def _select_class_probability_output(
    probs: Tensor,
    X: Tensor,
    *,
    output_index: int,
    name: str,
) -> Tensor:
    """Return canonical ``batch x q x classes`` probabilities.

    Supported layouts are:

    - ``sample_dims x batch_shape x q x classes``
    - ``sample_dims x batch_shape x q x outputs x classes``

    A missing singleton ``q`` axis is also restored for ``q == 1``.  Leading
    DeepGP / likelihood sample dimensions are averaged after the public input
    shape has been identified from the trailing dimensions.
    """

    if not torch.is_tensor(probs):
        raise TypeError(f"{name} must be a Tensor. Got {type(probs)}.")
    if probs.ndim < 1:
        raise RuntimeError(f"{name} must include a class dimension.")

    input_batch_q = tuple(int(size) for size in X.shape[:-1])
    batch_shape = tuple(int(size) for size in X.shape[:-2])
    raw_q = int(X.shape[-2])
    missing_q = False

    # Standard single-output classifier layout.  Match the complete public
    # batch+q suffix rather than relying on ndim: DeepGP adds leading sample
    # axes, which previously made q look like a model-output axis.
    if _shape_endswith(probs.shape[:-1], input_batch_q):
        selected = probs
    # Internal multi-output classifier layout: ... x batch x q x m x C.
    elif probs.ndim >= 2 and _shape_endswith(probs.shape[:-2], input_batch_q):
        num_outputs = int(probs.shape[-2])
        if output_index >= num_outputs:
            raise IndexError(
                f"output_index={output_index} is out of bounds for "
                f"{name}.shape={tuple(probs.shape)}."
            )
        selected = probs[..., output_index, :]
    elif raw_q == 1 and _shape_endswith(probs.shape[:-1], batch_shape):
        selected = probs
        missing_q = True
    elif raw_q == 1 and probs.ndim >= 2 and _shape_endswith(
        probs.shape[:-2],
        batch_shape,
    ):
        num_outputs = int(probs.shape[-2])
        if output_index >= num_outputs:
            raise IndexError(
                f"output_index={output_index} is out of bounds for "
                f"{name}.shape={tuple(probs.shape)}."
            )
        selected = probs[..., output_index, :]
        missing_q = True
    else:
        raise RuntimeError(
            f"Could not align {name} with X. "
            f"X.shape={tuple(X.shape)}, probabilities.shape={tuple(probs.shape)}. "
            "Expected trailing batch/q/class dimensions, optionally with one "
            "model-output axis before the class axis."
        )

    target_ndim = X.ndim - (1 if missing_q else 0)
    while selected.ndim > target_ndim:
        selected = selected.mean(dim=0)

    if missing_q:
        selected = selected.unsqueeze(-2)

    expected_prefix = tuple(int(size) for size in X.shape[:-1])
    if selected.ndim != X.ndim or tuple(selected.shape[:-1]) != expected_prefix:
        raise RuntimeError(
            f"Could not reduce {name} to the public hybrid shape. "
            f"Expected prefix={expected_prefix}, got shape={tuple(selected.shape)}."
        )
    return selected


def apply_hybrid_class_probability_shapes(hybrid_cls: type) -> None:
    """Install DeepGP-safe probability shape handling on a hybrid model class."""

    global _PATCHED
    if _PATCHED:
        return

    original_ordinal = hybrid_cls._ordinal_class_probs

    def _ordinal_class_probs(self, spec: Any, X: Tensor, **kwargs: Any) -> Tensor:
        fn = getattr(spec.model, "class_probs", None)
        if callable(fn):
            probs = self._call_class_probs(fn, X, **kwargs)
            if torch.is_tensor(probs):
                return _select_class_probability_output(
                    probs,
                    X,
                    output_index=spec.output_index,
                    name=f"{spec.name}.class_probs",
                ).clamp_min(0.0)
        return original_ordinal(self, spec, X, **kwargs)

    def _multiclass_probs(self, spec: Any, X: Tensor, **kwargs: Any) -> Tensor:
        fn = getattr(spec.model, "class_probs", None)
        if callable(fn):
            probs = self._call_class_probs(fn, X, **kwargs)
            if torch.is_tensor(probs):
                return _select_class_probability_output(
                    probs,
                    X,
                    output_index=spec.output_index,
                    name=f"{spec.name}.class_probs",
                ).clamp_min(0.0)

        post = self._call_accessor(
            spec.model,
            ("probability_posterior", "posterior"),
            X,
            **kwargs,
        )
        probs, _ = self._posterior_mean_variance(post, spec.name)
        return _select_class_probability_output(
            probs,
            X,
            output_index=spec.output_index,
            name=f"{spec.name}.posterior.mean",
        ).clamp_min(0.0)

    hybrid_cls._ordinal_class_probs = _ordinal_class_probs
    hybrid_cls._multiclass_probs = _multiclass_probs
    hybrid_cls._bochan_class_probability_shapes_patched = True
    _PATCHED = True


__all__ = ["apply_hybrid_class_probability_shapes"]
