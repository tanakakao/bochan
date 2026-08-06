"""Candidate-batch duplicate handling for the React Web workflow.

The Web defaults deliberately use joint q-batch optimization for ordinary
continuous searches and sequential optimization for mixed categorical searches.
Keep that policy unchanged.  If repair, grid rounding, or category decoding makes
final tensor candidates identical, refill only the duplicate slots with q=1
searches conditioned on the unique candidates through ``X_pending``.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from typing import Any

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
    acquisition optimizes the requested q-batch jointly.  Mixed searches and
    CMA-ES are already sent with ``True`` by the Web client.  Native population or
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
    import torch

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


def _refill_duplicate_candidate_result(
    tabular_optimizer: Any,
    initial_result: Any,
    original_candidate: Any,
) -> Any:
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

    max_attempts = max(8, missing * 6)
    attempts = 0
    refilled = 0
    base_context = initial_result.data_context
    refill_opt_config = replace(
        initial_result.opt_config,
        q=1,
        sequential=False,
    )

    while len(selected) < requested_q and attempts < max_attempts:
        attempts += 1
        pending = _pending_with_selected(
            getattr(base_context, "X_pending", None),
            selected,
        )
        refill_context = replace(base_context, X_pending=pending)
        try:
            refill_result = original_candidate(
                tabular_optimizer,
                initial_result.acq_config,
                refill_opt_config,
                data_context=refill_context,
                return_result=True,
            )
        except Exception as exc:  # keep the original q-batch if refill is unsupported
            state["candidate_refill_error"] = str(exc)
            break

        refill_candidates = _as_candidate_matrix(refill_result.candidates)
        if refill_candidates.shape[0] == 0:
            continue
        row = refill_candidates[0].detach().clone()
        key = _tensor_candidate_key(row)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        refilled += 1

    state["candidate_refill_attempts"] = attempts
    state["candidate_refill_count"] = refilled

    # Preserve the requested candidate count even when an acquisition does not
    # implement pending-point semantics.  Unresolved duplicates remain visible in
    # the final metadata instead of silently dropping rows.
    fallback_rows = iter(duplicates)
    while len(selected) < requested_q:
        try:
            selected.append(next(fallback_rows))
        except StopIteration:
            selected.append(candidates[len(selected) % candidates.shape[0]].detach().clone())

    import torch

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
        return _refill_duplicate_candidate_result(
            self,
            result,
            original_candidate,
        )

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
