"""Shared final-candidate uniqueness handling for optimizer backends.

The initial optimizer call keeps its native joint or sequential semantics. This
module acts on the final experiment-space representation when configured, then
refills duplicate slots from additional q=1 restart optima of the same
acquisition and backend. No acquisition wrapper or runtime method replacement
is used.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

OptimizeOnce = Callable[..., tuple[Any, Any]]

_NATIVE_BATCH_OPTIMIZERS = {
    "nsgaii",
    "nsga2",
    "optimize_acqf_nsgaii",
    "thompson_sampling",
    "optimize_thompson_sampling",
    "thompson_sampling_mixed",
    "optimize_thompson_sampling_mixed",
    "llm_candidate_set",
    "optimize_acqf_llm",
    "optimize_acqf_llm_candidate_set",
}


def _optimizer_name(config: Any) -> str:
    optimizer = getattr(config, "optimizer", "")
    if callable(optimizer) and not isinstance(optimizer, str):
        return "callable"
    return str(optimizer).replace("-", "_").lower()


def _candidate_matrix(candidates: Any) -> Tensor | None:
    """Return a final q-by-d matrix when the backend result is unambiguous."""

    if not torch.is_tensor(candidates):
        return None
    if candidates.ndim == 1:
        return candidates.unsqueeze(0)
    if candidates.ndim == 2:
        return candidates
    if candidates.ndim == 3 and candidates.shape[0] == 1:
        return candidates.squeeze(0)
    return None


def _coerce_tolerances(
    values: Sequence[float] | Tensor | None,
    *,
    like: Tensor,
) -> Tensor | None:
    if values is None:
        return None
    tolerances = torch.as_tensor(values, device=like.device, dtype=like.dtype)
    if tolerances.ndim != 1 or tolerances.numel() != like.shape[-1]:
        raise ValueError(
            "duplicate_tolerances must contain exactly one value per feature; "
            f"got {tuple(tolerances.shape)} for d={like.shape[-1]}."
        )
    if not torch.isfinite(tolerances).all() or (tolerances < 0).any():
        raise ValueError("duplicate_tolerances must contain finite non-negative values.")
    return tolerances


def _rows_are_duplicate(
    row: Tensor,
    reference: Tensor,
    *,
    tolerance: float,
    tolerances: Tensor | None,
) -> bool:
    if tolerances is None:
        return torch.allclose(row, reference, rtol=0.0, atol=tolerance)

    delta = (row - reference).abs()
    exact_dimensions = tolerances <= 0
    if exact_dimensions.any() and not torch.all(delta[exact_dimensions] <= tolerance):
        return False

    scaled_dimensions = ~exact_dimensions
    if not scaled_dimensions.any():
        return True
    normalized_distance = torch.linalg.vector_norm(delta[scaled_dimensions] / tolerances[scaled_dimensions])
    return bool(normalized_distance < 1.0)


def _is_duplicate(
    row: Tensor,
    selected: Sequence[Tensor],
    tolerance: float,
    tolerances: Tensor | None,
) -> bool:
    return any(
        _rows_are_duplicate(
            row,
            reference,
            tolerance=tolerance,
            tolerances=tolerances,
        )
        for reference in selected
    )


def _split_unique_rows(
    candidates: Tensor,
    *,
    tolerance: float,
    tolerances: Tensor | None,
) -> tuple[list[Tensor], list[Tensor]]:
    selected: list[Tensor] = []
    duplicates: list[Tensor] = []
    for row in candidates:
        detached = row.detach().clone()
        if _is_duplicate(detached, selected, tolerance, tolerances):
            duplicates.append(detached)
        else:
            selected.append(detached)
    return selected, duplicates


def count_unique_candidate_rows(
    candidates: Tensor,
    *,
    tolerance: float = 1e-10,
    tolerances: Sequence[float] | Tensor | None = None,
) -> int:
    """Count unique rows using the same final-space rule as candidate refill."""

    matrix = _candidate_matrix(candidates)
    if matrix is None:
        raise ValueError("candidates must be a one-, two-, or singleton-batch three-dimensional tensor.")
    resolved = _coerce_tolerances(tolerances, like=matrix)
    selected, _ = _split_unique_rows(
        matrix,
        tolerance=float(tolerance),
        tolerances=resolved,
    )
    return len(selected)


def _pool_rows(candidates: Any, acq_values: Any) -> list[Tensor]:
    """Return restart candidates ordered from highest to lowest acquisition value."""

    if not torch.is_tensor(candidates):
        return []
    if candidates.ndim == 1:
        rows = candidates.unsqueeze(0)
    elif candidates.ndim == 2:
        rows = candidates
    elif candidates.ndim >= 3 and candidates.shape[-2] == 1:
        rows = candidates.reshape(-1, candidates.shape[-1])
    else:
        return []

    order = torch.arange(rows.shape[0], device=rows.device)
    if torch.is_tensor(acq_values):
        scores = acq_values.detach().reshape(-1)
        if scores.numel() == rows.shape[0]:
            finite_scores = torch.nan_to_num(
                scores,
                nan=-torch.inf,
                neginf=-torch.inf,
                posinf=torch.inf,
            )
            order = torch.argsort(finite_scores, descending=True)

    return [rows[index].detach().clone() for index in order.tolist()]


def _coerce_pending_tensor(value: Any, *, like: Tensor) -> Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        pending = value
    elif isinstance(value, (list, tuple)):
        tensors = [item for item in value if torch.is_tensor(item) and item.numel() > 0]
        if not tensors:
            return None
        pending = torch.cat(
            [item.reshape(-1, item.shape[-1]) for item in tensors],
            dim=-2,
        )
    else:
        return None

    if pending.ndim == 1:
        pending = pending.unsqueeze(0)
    return pending.reshape(-1, pending.shape[-1]).to(
        device=like.device,
        dtype=like.dtype,
    )


def _pending_with_selected(original_pending: Any, selected: Sequence[Tensor]) -> Tensor:
    selected_tensor = torch.stack(list(selected), dim=0)
    pending = _coerce_pending_tensor(original_pending, like=selected_tensor)
    if pending is None:
        return selected_tensor
    return torch.cat([pending, selected_tensor], dim=-2)


def _set_pending_if_supported(acqf: Any, X_pending: Any) -> bool:
    setter = getattr(acqf, "set_X_pending", None)
    if not callable(setter):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setter(X_pending)
    except Exception:
        return False
    return True


def _evaluate_final_batch(acqf: Any, candidates: Tensor, fallback: Any) -> Any:
    try:
        with torch.no_grad():
            return acqf(candidates)
    except Exception:
        return fallback


def _apply_final_postprocess(candidates: Any, function: Any) -> Any:
    if function is None:
        return candidates
    processed = function(candidates)
    if not torch.is_tensor(processed):
        raise TypeError("final_candidate_postprocess must return a Tensor.")
    if torch.is_tensor(candidates) and processed.shape != candidates.shape:
        raise RuntimeError(
            "final_candidate_postprocess must preserve candidate shape; "
            f"got {tuple(candidates.shape)} -> {tuple(processed.shape)}."
        )
    return processed


def ensure_unique_candidates(
    *,
    acqf: Any,
    bounds: Any,
    config: Any,
    candidates: Any,
    acq_value: Any,
    optimize_once: OptimizeOnce,
) -> tuple[Any, Any]:
    """Refill duplicate final candidates without altering initial batch semantics.

    The first occurrences from the final experiment-space result are retained.
    Duplicate slots are filled from additional q=1 optimization restarts. Each
    refill candidate is transformed by ``final_candidate_postprocess`` before
    duplicate comparison, so rounding, categorical decoding, fixed values, and
    constraint repair are reflected in the decision.
    """

    if not bool(getattr(config, "ensure_unique_candidates", True)):
        return candidates, acq_value
    if not bool(getattr(config, "return_best_only", True)):
        return candidates, acq_value
    if _optimizer_name(config) in _NATIVE_BATCH_OPTIMIZERS:
        return candidates, acq_value

    final_postprocess = getattr(config, "final_candidate_postprocess", None)
    processed_candidates = _apply_final_postprocess(candidates, final_postprocess)
    requested_q = int(getattr(config, "q", 1))
    if requested_q <= 1:
        if processed_candidates is candidates:
            return candidates, acq_value
        return processed_candidates, _evaluate_final_batch(
            acqf,
            processed_candidates,
            acq_value,
        )

    matrix = _candidate_matrix(processed_candidates)
    if matrix is None or matrix.shape[0] != requested_q:
        return processed_candidates, acq_value

    tolerance = float(getattr(config, "duplicate_tolerance", 1e-10))
    tolerances = _coerce_tolerances(
        getattr(config, "duplicate_tolerances", None),
        like=matrix,
    )
    selected, original_duplicates = _split_unique_rows(
        matrix,
        tolerance=tolerance,
        tolerances=tolerances,
    )
    if len(selected) == requested_q:
        if processed_candidates is candidates:
            return candidates, acq_value
        return processed_candidates, _evaluate_final_batch(
            acqf,
            matrix,
            acq_value,
        )

    max_attempts = int(getattr(config, "duplicate_refill_attempts", 4))
    minimum_restarts = int(getattr(config, "duplicate_pool_restarts", 16))
    base_restarts = max(int(getattr(config, "num_restarts", 1)), minimum_restarts)
    base_raw_samples = max(
        int(getattr(config, "raw_samples", 1)),
        base_restarts * 16,
    )

    original_pending = getattr(acqf, "X_pending", None)
    last_error: Exception | None = None
    try:
        for attempt in range(max_attempts):
            if len(selected) >= requested_q:
                break

            pending = _pending_with_selected(original_pending, selected)
            _set_pending_if_supported(acqf, pending)

            scale = 2**attempt
            refill_optimizer_kwargs = dict(getattr(config, "optimizer_kwargs", {}) or {})
            for key in (
                "batch_initial_conditions",
                "acq_function_sequence",
                "return_full_tree",
            ):
                refill_optimizer_kwargs.pop(key, None)

            refill_config = replace(
                config,
                q=1,
                sequential=False,
                return_best_only=False,
                num_restarts=base_restarts * scale,
                raw_samples=base_raw_samples * scale,
                optimizer_kwargs=refill_optimizer_kwargs,
            )
            try:
                pool_candidates, pool_values = optimize_once(
                    acqf=acqf,
                    bounds=bounds,
                    config=refill_config,
                )
                pool_candidates = _apply_final_postprocess(
                    pool_candidates,
                    final_postprocess,
                )
            except Exception as exc:
                last_error = exc
                continue

            for row in _pool_rows(pool_candidates, pool_values):
                if _is_duplicate(row, selected, tolerance, tolerances):
                    continue
                selected.append(row)
                if len(selected) >= requested_q:
                    break
    finally:
        _set_pending_if_supported(acqf, original_pending)

    unresolved = requested_q - len(selected)
    if unresolved > 0:
        fallback_rows = list(original_duplicates)
        if not fallback_rows:
            fallback_rows = [row.detach().clone() for row in matrix]
        index = 0
        while len(selected) < requested_q:
            selected.append(fallback_rows[index % len(fallback_rows)])
            index += 1

        detail = f" Last refill error: {last_error}" if last_error is not None else ""
        warnings.warn(
            "Could not produce the requested number of unique final candidates; "
            f"{unresolved} duplicate slot(s) remain.{detail}",
            RuntimeWarning,
            stacklevel=2,
        )

    final_candidates = torch.stack(selected[:requested_q], dim=0)
    final_acq_value = _evaluate_final_batch(acqf, final_candidates, acq_value)
    return final_candidates, final_acq_value


__all__ = ["count_unique_candidate_rows", "ensure_unique_candidates"]
