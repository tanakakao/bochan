"""Default utility values for ordinal multi-output acquisitions."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from . import multi_output as _multi_output
from ._fixed_likelihood_graph import apply_fixed_ordinal_likelihood_graph_compat

apply_fixed_ordinal_likelihood_graph_compat(_multi_output)


def _num_classes_from_object(obj: Any) -> int | None:
    if obj is None:
        return None
    num_classes = getattr(obj, "num_classes", None)
    if num_classes is not None:
        return int(num_classes)
    for name in ("cutpoints", "thresholds", "cuts", "boundaries", "_cutpoints"):
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if callable(value):
            value = value()
        if torch.is_tensor(value):
            return int(value.numel()) + 1
    return None


def _infer_num_outputs(model: Any) -> int:
    num_outputs = getattr(model, "num_outputs", None)
    if num_outputs is not None:
        try:
            return max(1, int(num_outputs))
        except (TypeError, ValueError):
            pass
    submodels = getattr(model, "models", None)
    if submodels is not None:
        try:
            return max(1, len(submodels))
        except TypeError:
            pass
    return 1


def infer_multioutput_ordinal_utility_values(model: Any) -> Tensor | list[Tensor]:
    """Infer one ordinal utility vector per model output.

    A one-dimensional tensor represents one output in the downstream utility
    normalizer. Therefore a shared class scale must still be repeated for every
    output of a multi-output model.
    """
    counts: list[int] = []
    num_outputs = _infer_num_outputs(model)

    for obj in (
        model,
        getattr(model, "ordinal_likelihood", None),
        getattr(model, "likelihood", None),
    ):
        count = _num_classes_from_object(obj)
        if count is not None:
            counts = [count]
            break

    if not counts and hasattr(model, "models"):
        for submodel in model.models:
            count = None
            for obj in (
                submodel,
                getattr(submodel, "ordinal_likelihood", None),
                getattr(submodel, "likelihood", None),
            ):
                count = _num_classes_from_object(obj)
                if count is not None:
                    break
            if count is None:
                targets = getattr(submodel, "train_targets", None)
                if targets is None:
                    targets = getattr(submodel, "train_Y", None)
                if targets is not None and torch.as_tensor(targets).numel() > 0:
                    count = int(torch.as_tensor(targets).max().item()) + 1
            if count is None:
                raise ValueError(
                    "Could not infer ordinal class count for one multi-output submodel. "
                    "Pass utility_values explicitly."
                )
            counts.append(count)

    if not counts:
        targets = getattr(model, "train_targets", None)
        if targets is None:
            targets = getattr(model, "train_Y", None)
        if targets is not None and torch.as_tensor(targets).numel() > 0:
            counts = [int(torch.as_tensor(targets).max().item()) + 1]

    if not counts:
        raise ValueError(
            "Could not infer ordinal utility_values from model metadata or training labels. "
            "Pass utility_values explicitly."
        )

    if len(counts) == 1 and num_outputs > 1:
        counts = counts * num_outputs
    elif len(counts) not in {1, num_outputs}:
        raise ValueError(
            "Inferred ordinal class counts do not match model outputs: "
            f"counts={counts}, num_outputs={num_outputs}."
        )

    device = None
    dtype = torch.double
    train_input = getattr(model, "train_X", None)
    if train_input is None and hasattr(model, "train_inputs"):
        train_inputs = model.train_inputs
        if isinstance(train_inputs, tuple) and train_inputs:
            train_input = train_inputs[0]
    if torch.is_tensor(train_input):
        device = train_input.device
        dtype = train_input.dtype if train_input.is_floating_point() else torch.double

    utilities = [torch.arange(count, device=device, dtype=dtype) for count in counts]
    if num_outputs == 1:
        return utilities[0]
    return utilities
