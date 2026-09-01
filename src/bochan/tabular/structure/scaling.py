"""Scalable mixed acquisition optimization for structure-aware models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


def _merge_fixed_features(
    base: Mapping[int, float] | None,
    extra: Mapping[int, float],
) -> dict[int, float]:
    """Merge fixed features with the internally derived assignment taking priority."""

    merged = {int(index): float(value) for index, value in (base or {}).items()}
    for index, value in extra.items():
        merged[int(index)] = float(value)
    return merged


def optimize_structure_alternating(
    acq_function: Any,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    *,
    structure_dim: int,
    structure_values: Sequence[float],
    process_fixed_features_list: Sequence[Mapping[int, float]],
    fixed_features: Mapping[int, float] | None = None,
    post_processing_func: Any | None = None,
    inequality_constraints: Any | None = None,
    equality_constraints: Any | None = None,
    return_best_only: bool = True,
    sequential: bool = True,
    options: Mapping[str, Any] | None = None,
    alternating_options: Mapping[str, Any] | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize a discrete structure selector while preserving process tuples.

    The process categorical dimensions remain fixed to one currently observed
    joint assignment per inner optimization. Only the structure selector is
    exposed as a categorical dimension to BoTorch's alternating mixed optimizer.
    This avoids a full ``n_structures * n_process_assignments`` enumeration of
    continuous solves while preserving the feasible process-category contract.

    The scalable backend is intentionally restricted to ``q=1``. Batch BO falls
    back to exact enumeration so different q slots may choose different process
    category assignments without changing the established candidate semantics.
    """

    del sequential
    if q != 1:
        raise ValueError("Structure alternating optimization currently requires q=1.")
    if not return_best_only:
        raise ValueError(
            "Structure alternating optimization requires return_best_only=True."
        )
    if not structure_values:
        raise ValueError("structure_values must contain at least one structure index.")
    if not process_fixed_features_list:
        process_fixed_features_list = ({},)
    if fixed_features is not None and int(structure_dim) in fixed_features:
        raise ValueError(
            "Fix the structure through structure_ids, not fixed_features, "
            "when alternating structure optimization is active."
        )

    from botorch.optim import optimize_acqf_mixed_alternating

    best_candidate: Tensor | None = None
    best_value: Tensor | None = None
    best_score = float("-inf")
    categorical = {
        int(structure_dim): [float(value) for value in structure_values]
    }
    resolved_options = {"initialization_strategy": "random"}
    resolved_options.update(dict(options or {}))
    resolved_options.update(dict(alternating_options or {}))

    for process_fixed in process_fixed_features_list:
        inner_fixed = _merge_fixed_features(fixed_features, process_fixed)
        candidate, acq_value = optimize_acqf_mixed_alternating(
            acq_function=acq_function,
            bounds=bounds,
            cat_dims=categorical,
            options=resolved_options,
            q=1,
            raw_samples=int(raw_samples),
            num_restarts=int(num_restarts),
            post_processing_func=post_processing_func,
            sequential=True,
            fixed_features=inner_fixed or None,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            return_acq_values=True,
        )
        if acq_value is None:
            raise RuntimeError("BoTorch alternating optimizer did not return an acquisition value.")
        score = float(torch.as_tensor(acq_value).detach().max().cpu().item())
        if best_candidate is None or score > best_score:
            best_score = score
            best_candidate = candidate
            best_value = torch.as_tensor(acq_value)

    if best_candidate is None or best_value is None:
        raise RuntimeError("Structure alternating optimization produced no candidate.")
    return best_candidate, best_value


__all__ = ["optimize_structure_alternating"]
