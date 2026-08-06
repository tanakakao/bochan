"""Candidate-batch duplicate handling for the React Web workflow.

The Web defaults deliberately use joint q-batch optimization for ordinary
continuous searches and sequential optimization for mixed categorical searches.
Keep that policy unchanged. If repair, grid rounding, or category decoding makes
final tensor candidates identical, refill only the duplicate slots with q=1
searches.

Native ``X_pending`` semantics are used when the acquisition supports them, but
``X_pending`` is not a universal duplicate constraint. A normalized-distance
exclusion wrapper is therefore applied around every scalar refill acquisition so
custom active-learning and level-set acquisitions are handled consistently too.
"""

from __future__ import annotations

import warnings
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

_NATIVE_BATCH_METHODS = {
    "nsgaii",
    "nsga2",
    "thompson_sampling",
    "optimize_thompson_sampling",
}
_WEB_CANDIDATE_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "bochan_web_candidate_context",
    default=None,
)


def _normalized_method(name: Any) -> str:
    return str(name or "normal").replace("-", "_").lower()


def resolve_web_candidate_sequential(
    *,
    q: int,
    search_method: str,
    requested: bool,
) -> tuple[bool, str]:
    """Report the requested Web batch policy without overriding it.

    Ordinary continuous searches intentionally keep ``sequential=False`` so the
    acquisition optimizes the requested q-batch jointly. Mixed searches and
    CMA-ES are already sent with ``True`` by the Web client. Native population or
    posterior-sampling methods own their batch semantics and remain non-sequential.
    """

    q = int(q)
    if q <= 1:
        return bool(requested), "single"

    method = _normalized_method(search_method)
    if method in _NATIVE_BATCH_METHODS:
        return False, "native_batch"
    return bool(requested), (
        "sequential_pending" if requested else "joint_batch"
    )


def prepare_web_candidate_request(request: Any) -> tuple[Any, dict[str, Any]]:
    """Return the original request and diagnostics for its batch policy."""

    optimizer = getattr(request, "optimizer", None)
    if optimizer is None:
        return request, {
            "candidate_batch_strategy": "unknown",
            "candidate_sequential_requested": False,
            "candidate_sequential_effective": False,
        }

    requested = bool(getattr(optimizer, "sequential", False))
    effective, strategy = resolve_web_candidate_sequential(
        q=int(getattr(optimizer, "q", 1)),
        search_method=str(getattr(optimizer, "name", "normal")),
        requested=requested,
    )
    return request, {
        "candidate_batch_strategy": strategy,
        "candidate_sequential_requested": requested,
        "candidate_sequential_effective": effective,
    }


def _canonical_candidate_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 12)
    return str(value)


def candidate_uniqueness_metadata(result: dict[str, Any]) -> dict[str, int]:
    """Count unique final candidates after all Web post-processing."""

    candidates = list(result.get("candidates") or [])
    keys: list[tuple[tuple[str, Any], ...]] = []
    for candidate in candidates:
        values = candidate.get("encoded_values") or candidate.get("values") or {}
        key = tuple(
            (str(name), _canonical_candidate_value(value))
            for name, value in sorted(values.items(), key=lambda item: str(item[0]))
        )
        keys.append(key)

    unique_count = len(set(keys))
    return {
        "candidate_count": len(keys),
        "candidate_unique_count": unique_count,
        "candidate_duplicate_count": len(keys) - unique_count,
    }


def _tensor_candidate_key(row: Any) -> tuple[float, ...]:
    values = row.detach().cpu().reshape(-1).tolist()
    return tuple(round(float(value), 12) for value in values)


def _as_candidate_matrix(candidates: Any) -> Any:
    if candidates.ndim == 1:
        return candidates.unsqueeze(0)
    return candidates.reshape(-1, candidates.shape[-1])


def _pending_with_selected(base_pending: Any, selected: list[Any]) -> Any:
    selected_tensor = torch.stack(selected, dim=0)
    if base_pending is None:
        return selected_tensor
    pending = base_pending
    if pending.ndim == 1:
        pending = pending.unsqueeze(0)
    pending = pending.reshape(-1, pending.shape[-1]).to(
        device=selected_tensor.device,
        dtype=selected_tensor.dtype,
    )
    return torch.cat([pending, selected_tensor], dim=-2)


def _set_pending_if_supported(acqf: Any, X_pending: Any) -> bool:
    """Set native pending points when supported, without relying on them."""

    setter = getattr(acqf, "set_X_pending", None)
    if not callable(setter):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setter(X_pending)
    except (TypeError, NotImplementedError, RuntimeError):
        return False
    return True


def _coerce_bounds(bounds: Any, *, like: Tensor) -> Tensor:
    bounds_tensor = torch.as_tensor(bounds, device=like.device, dtype=like.dtype)
    if bounds_tensor.ndim != 2 or bounds_tensor.shape[0] != 2:
        raise ValueError(
            "Candidate refill requires bounds with shape (2, d), got "
            f"{tuple(bounds_tensor.shape)}."
        )
    return bounds_tensor


class _ExcludedCandidateAcquisition(AcquisitionFunction):
    """Subtract a smooth exclusion penalty from any scalar acquisition."""

    def __init__(
        self,
        base_acqf: Any,
        *,
        excluded: Tensor,
        bounds: Tensor,
        radius: float,
        penalty_weight: float,
    ) -> None:
        super().__init__(model=getattr(base_acqf, "model", None))
        self.base_acqf = base_acqf
        self.register_buffer("excluded", excluded.detach().clone())
        self.register_buffer("exclusion_bounds", bounds.detach().clone())
        self.radius = max(float(radius), 1e-12)
        self.penalty_weight = max(float(penalty_weight), 1.0)

    def forward(self, X: Tensor) -> Tensor:
        base_value = self.base_acqf(X)
        lower = self.exclusion_bounds[0]
        span = (self.exclusion_bounds[1] - lower).abs().clamp_min(1e-12)
        normalized_X = (X - lower) / span
        normalized_excluded = (self.excluded - lower) / span

        distances = torch.cdist(
            normalized_X.reshape(-1, normalized_X.shape[-1]),
            normalized_excluded.reshape(-1, normalized_excluded.shape[-1]),
        )
        min_distance = distances.min(dim=-1).values.reshape(*X.shape[:-1])
        penalty_per_point = self.penalty_weight * torch.exp(
            -0.5 * (min_distance / self.radius).pow(2)
        )
        penalty = penalty_per_point.max(dim=-1).values

        while penalty.ndim > base_value.ndim:
            penalty = penalty.max(dim=0).values
        return base_value - penalty


def _acquisition_scale(acqf: Any, selected: list[Any]) -> float:
    """Estimate a stable penalty scale from q=1 acquisition values."""

    if not selected:
        return 1.0
    X = torch.stack(selected, dim=0).unsqueeze(-2)
    try:
        with torch.no_grad():
            values = torch.as_tensor(acqf(X))
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return 1.0
        return max(float(finite.abs().max().item()), 1.0)
    except Exception:
        return 1.0


def _refill_radius(attempt: int) -> float:
    """Increase normalized exclusion radius only after repeated collisions."""

    return min(0.25, 0.002 * (1.8 ** max(int(attempt) - 1, 0)))


def _optimize_refill_candidate(
    *,
    acqf: Any,
    bounds: Any,
    opt_config: Any,
) -> tuple[Any, Any]:
    from bochan.api.optimizer_api import optimize_candidates

    return optimize_candidates(acqf=acqf, bounds=bounds, config=opt_config)


def _refill_duplicate_candidate_result(initial_result: Any) -> Any:
    """Replace duplicate repaired candidates without changing the initial q policy."""

    state = _WEB_CANDIDATE_CONTEXT.get()
    if state is None:
        return initial_result
    if _normalized_method(state.get("search_method")) in _NATIVE_BATCH_METHODS:
        return initial_result

    candidates = _as_candidate_matrix(initial_result.candidates)
    requested_q = int(getattr(initial_result.opt_config, "q", candidates.shape[0]))
    if requested_q <= 1 or candidates.shape[0] <= 1:
        return initial_result

    selected: list[Any] = []
    seen: set[tuple[float, ...]] = set()
    duplicates: list[Any] = []
    for row in candidates:
        key = _tensor_candidate_key(row)
        if key in seen:
            duplicates.append(row.detach().clone())
            continue
        seen.add(key)
        selected.append(row.detach().clone())

    missing = requested_q - len(selected)
    state["candidate_initial_duplicate_count"] = max(missing, 0)
    if missing <= 0:
        return initial_result

    bounds = getattr(initial_result.data_context, "bounds", None)
    if bounds is None:
        state["candidate_refill_error"] = "Candidate refill requires optimization bounds."
        return initial_result
    bounds_tensor = _coerce_bounds(bounds, like=candidates)

    max_attempts = max(10, missing * 8)
    attempts = 0
    refilled = 0
    base_acqf = initial_result.acqf
    base_pending = getattr(initial_result.data_context, "X_pending", None)
    original_pending = getattr(base_acqf, "X_pending", None)
    penalty_scale = _acquisition_scale(base_acqf, selected)
    refill_opt_config = replace(
        initial_result.opt_config,
        q=1,
        sequential=False,
    )

    try:
        while len(selected) < requested_q and attempts < max_attempts:
            attempts += 1
            pending = _pending_with_selected(base_pending, selected)
            native_pending = _set_pending_if_supported(base_acqf, pending)
            excluded = torch.stack(selected, dim=0)
            refill_acqf = _ExcludedCandidateAcquisition(
                base_acqf,
                excluded=excluded,
                bounds=bounds_tensor,
                radius=_refill_radius(attempts),
                penalty_weight=penalty_scale * 1_000_000.0,
            )
            try:
                refill_candidates, _ = _optimize_refill_candidate(
                    acqf=refill_acqf,
                    bounds=bounds_tensor,
                    opt_config=refill_opt_config,
                )
            except Exception as exc:
                state["candidate_refill_error"] = str(exc)
                break

            refill_matrix = _as_candidate_matrix(refill_candidates)
            if refill_matrix.shape[0] == 0:
                continue
            row = refill_matrix[0].detach().clone()
            key = _tensor_candidate_key(row)
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            refilled += 1
            state["candidate_native_pending_used"] = bool(native_pending)
    finally:
        _set_pending_if_supported(base_acqf, original_pending)

    state["candidate_refill_attempts"] = attempts
    state["candidate_refill_count"] = refilled
    state["candidate_exclusion_penalty_used"] = True

    # Preserve the requested candidate count when the feasible repaired space has
    # fewer than q distinct points. Residual duplicates remain explicit in the
    # final metadata instead of silently dropping rows.
    fallback_rows = iter(duplicates)
    while len(selected) < requested_q:
        try:
            selected.append(next(fallback_rows))
        except StopIteration:
            selected.append(
                candidates[len(selected) % candidates.shape[0]].detach().clone()
            )

    return replace(
        initial_result,
        candidates=torch.stack(selected[:requested_q], dim=0),
    )


def _install_tabular_candidate_refill() -> None:
    from bochan.tabular.optimizer import TabularBayesianOptimizer

    if getattr(TabularBayesianOptimizer, "_web_candidate_refill_installed", False):
        return

    original_candidate = TabularBayesianOptimizer.candidate

    @wraps(original_candidate)
    def candidate_with_duplicate_refill(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_candidate(self, *args, **kwargs)
        if not bool(kwargs.get("return_result", False)):
            return result
        return _refill_duplicate_candidate_result(result)

    TabularBayesianOptimizer.candidate = candidate_with_duplicate_refill
    TabularBayesianOptimizer._web_candidate_refill_installed = True


def install_web_candidate_batch_diversity(
    workflows_module: Any,
    workflows_tabular_module: Any,
) -> None:
    """Install duplicate refill and Web result diagnostics once."""

    if getattr(workflows_tabular_module, "_candidate_batch_diversity_installed", False):
        return

    _install_tabular_candidate_refill()
    original = workflows_tabular_module.run_regression_web_workflow

    @wraps(original)
    def wrapped(request: Any, store: Any) -> dict[str, Any]:
        prepared_request, strategy_metadata = prepare_web_candidate_request(request)
        optimizer = getattr(prepared_request, "optimizer", None)
        state: dict[str, Any] = {
            "search_method": getattr(optimizer, "name", "normal"),
            "candidate_refill_attempts": 0,
            "candidate_refill_count": 0,
            "candidate_initial_duplicate_count": 0,
            "candidate_native_pending_used": False,
            "candidate_exclusion_penalty_used": False,
        }
        token = _WEB_CANDIDATE_CONTEXT.set(state)
        try:
            result = original(prepared_request, store)
        finally:
            _WEB_CANDIDATE_CONTEXT.reset(token)

        metadata = dict(result.get("metadata") or {})
        metadata.update(strategy_metadata)
        metadata.update(candidate_uniqueness_metadata(result))
        metadata.update(
            {
                "candidate_initial_duplicate_count": int(
                    state.get("candidate_initial_duplicate_count", 0)
                ),
                "candidate_refill_attempts": int(
                    state.get("candidate_refill_attempts", 0)
                ),
                "candidate_refill_count": int(
                    state.get("candidate_refill_count", 0)
                ),
                "candidate_native_pending_used": bool(
                    state.get("candidate_native_pending_used", False)
                ),
                "candidate_exclusion_penalty_used": bool(
                    state.get("candidate_exclusion_penalty_used", False)
                ),
            }
        )
        if state.get("candidate_refill_error"):
            metadata["candidate_refill_error"] = state["candidate_refill_error"]
        result["metadata"] = metadata
        return result

    workflows_tabular_module.run_regression_web_workflow = wrapped
    workflows_module._run_regression_web_workflow = wrapped
    workflows_tabular_module._candidate_batch_diversity_installed = True


__all__ = [
    "candidate_uniqueness_metadata",
    "install_web_candidate_batch_diversity",
    "prepare_web_candidate_request",
    "resolve_web_candidate_sequential",
]
