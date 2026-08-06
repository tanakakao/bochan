"""Candidate-batch diversity policy for the React Web workflow.

The Web workbench presents a small set of experiments that should normally be
meaningfully distinct.  Joint q-batch optimization can converge multiple rows
to the same optimum, and grid / categorical post-processing can make nearly
equal continuous solutions exactly equal.  For optimizers that support pending
points, use greedy sequential selection so each new candidate is conditioned on
those already selected.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

_NATIVE_BATCH_METHODS = {
    "nsgaii",
    "nsga2",
    "thompson_sampling",
    "optimize_thompson_sampling",
}


def _normalized_method(name: Any) -> str:
    return str(name or "normal").replace("-", "_").lower()


def resolve_web_candidate_sequential(
    *,
    q: int,
    search_method: str,
    requested: bool,
) -> tuple[bool, str]:
    """Resolve the effective Web batch strategy.

    The Web UI does not expose joint-versus-sequential optimization as an
    advanced control.  For ``q > 1``, gradient and evolutionary acquisition
    optimizers therefore use greedy pending-point selection by default.  Native
    batch selectors already enforce their own diversity and remain unchanged.

    Args:
        q: Number of requested candidates.
        search_method: Web search-method name.
        requested: Sequential flag received from the client.  This is retained
            in diagnostics, but Web defaults are made robust to older clients
            that sent ``False`` for continuous search spaces.

    Returns:
        Effective sequential flag and a diagnostic strategy name.
    """

    del requested
    q = int(q)
    if q <= 1:
        return False, "single"

    method = _normalized_method(search_method)
    if method in _NATIVE_BATCH_METHODS:
        return False, "native_batch"
    return True, "sequential_pending"


def prepare_web_candidate_request(request: Any) -> tuple[Any, dict[str, Any]]:
    """Return a request whose optimizer uses the effective diversity policy."""

    optimizer = getattr(request, "optimizer", None)
    if optimizer is None:
        return request, {
            "candidate_batch_strategy": "unknown",
            "candidate_sequential_requested": False,
            "candidate_sequential_effective": False,
        }

    requested = bool(getattr(optimizer, "sequential", False))
    q = int(getattr(optimizer, "q", 1))
    search_method = str(getattr(optimizer, "name", "normal"))
    effective, strategy = resolve_web_candidate_sequential(
        q=q,
        search_method=search_method,
        requested=requested,
    )

    if effective == requested:
        prepared = request
    elif hasattr(optimizer, "model_copy") and hasattr(request, "model_copy"):
        prepared_optimizer = optimizer.model_copy(update={"sequential": effective})
        prepared = request.model_copy(update={"optimizer": prepared_optimizer})
    else:
        # Lightweight namespace / test compatibility.  Avoid mutating the
        # caller's request when a normal object can be copied safely.
        from copy import copy

        prepared = copy(request)
        prepared_optimizer = copy(optimizer)
        prepared_optimizer.sequential = effective
        prepared.optimizer = prepared_optimizer

    return prepared, {
        "candidate_batch_strategy": strategy,
        "candidate_sequential_requested": requested,
        "candidate_sequential_effective": effective,
    }


def _canonical_candidate_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        # Candidate serialization can introduce insignificant floating-point
        # noise.  Twelve decimals is strict enough to identify only effectively
        # identical experimental settings.
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


def install_web_candidate_batch_diversity(
    workflows_module: Any,
    workflows_tabular_module: Any,
) -> None:
    """Install the Web candidate diversity adapter once."""

    if getattr(workflows_tabular_module, "_candidate_batch_diversity_installed", False):
        return

    original = workflows_tabular_module.run_regression_web_workflow

    @wraps(original)
    def wrapped(request: Any, store: Any) -> dict[str, Any]:
        prepared_request, strategy_metadata = prepare_web_candidate_request(request)
        result = original(prepared_request, store)
        metadata = dict(result.get("metadata") or {})
        metadata.update(strategy_metadata)
        metadata.update(candidate_uniqueness_metadata(result))
        result["metadata"] = metadata
        return result

    workflows_tabular_module.run_regression_web_workflow = wrapped
    # workflows.py binds the internal callable at import time.  Update that
    # binding as well so both package and direct app imports use the adapter.
    workflows_module._run_regression_web_workflow = wrapped
    workflows_tabular_module._candidate_batch_diversity_installed = True


__all__ = [
    "candidate_uniqueness_metadata",
    "install_web_candidate_batch_diversity",
    "prepare_web_candidate_request",
    "resolve_web_candidate_sequential",
]
