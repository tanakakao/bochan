from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_exact(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"{path}: expected {count} occurrence(s), found {actual} for:\n{old}"
        )
    target.write_text(text.replace(old, new), encoding="utf-8")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


# Public optimizer configuration: support final-space postprocessing and
# feature-wise minimum candidate distances.
replace_exact(
    "src/bochan/api/optimizer_config.py",
    "from __future__ import annotations\n\nfrom collections.abc import Callable, Sequence\n",
    "from __future__ import annotations\n\nimport math\nfrom collections.abc import Callable, Sequence\n",
)
replace_exact(
    "src/bochan/api/optimizer_config.py",
    "    ensure_unique_candidates: bool = True\n"
    "    duplicate_tolerance: float = 1e-10\n"
    "    duplicate_refill_attempts: int = 4\n",
    "    ensure_unique_candidates: bool = True\n"
    "    duplicate_tolerance: float = 1e-10\n"
    "    duplicate_tolerances: Sequence[float] | None = None\n"
    "    final_candidate_postprocess: Callable[[Any], Any] | None = None\n"
    "    duplicate_refill_attempts: int = 4\n",
)
replace_exact(
    "src/bochan/api/optimizer_config.py",
    "        if self.duplicate_tolerance < 0:\n"
    "            raise ValueError(\"duplicate_tolerance must be non-negative.\")\n"
    "        if self.duplicate_refill_attempts < 1:\n",
    "        if self.duplicate_tolerance < 0:\n"
    "            raise ValueError(\"duplicate_tolerance must be non-negative.\")\n"
    "        if self.duplicate_tolerances is not None:\n"
    "            tolerances = tuple(float(value) for value in self.duplicate_tolerances)\n"
    "            if any(not math.isfinite(value) or value < 0 for value in tolerances):\n"
    "                raise ValueError(\n"
    "                    \"duplicate_tolerances must contain finite non-negative values.\"\n"
    "                )\n"
    "            self.duplicate_tolerances = tolerances\n"
    "        if (\n"
    "            self.final_candidate_postprocess is not None\n"
    "            and not callable(self.final_candidate_postprocess)\n"
    "        ):\n"
    "            raise ValueError(\"final_candidate_postprocess must be callable.\")\n"
    "        if self.duplicate_refill_attempts < 1:\n",
)

write(
    "src/bochan/api/candidate_uniqueness.py",
    '''"""Shared final-candidate uniqueness handling for optimizer backends.

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
        raise ValueError(
            "duplicate_tolerances must contain finite non-negative values."
        )
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
    if exact_dimensions.any() and not torch.all(
        delta[exact_dimensions] <= tolerance
    ):
        return False

    scaled_dimensions = ~exact_dimensions
    if not scaled_dimensions.any():
        return True
    normalized_distance = torch.linalg.vector_norm(
        delta[scaled_dimensions] / tolerances[scaled_dimensions]
    )
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
        raise ValueError(
            "candidates must be a one-, two-, or singleton-batch three-dimensional tensor."
        )
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
        tensors = [
            item
            for item in value
            if torch.is_tensor(item) and item.numel() > 0
        ]
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
            refill_optimizer_kwargs = dict(
                getattr(config, "optimizer_kwargs", {}) or {}
            )
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
''',
)

# Web response acquisition-value semantics.
replace_exact(
    "src/bochan/serving/webapp/target_results.py",
    '''def _broadcast_acq_values(acq_value: Any, n: int) -> list[float | None]:
    """Broadcast scalar acquisition values to candidate rows."""

    try:
        values = acq_value.detach().cpu().reshape(-1).tolist()
    except Exception:
        values = [acq_value]
    values = [float(value) for value in values if value is not None]
    if not values:
        return [None for _ in range(n)]
    if len(values) == 1:
        return values * n
    if len(values) < n:
        return values + [values[-1]] * (n - len(values))
    return values[:n]
''',
    '''def _flatten_acq_values(acq_value: Any) -> list[float]:
    """Return acquisition values as a flat Python list."""

    try:
        values = acq_value.detach().cpu().reshape(-1).tolist()
    except Exception:
        values = [acq_value]
    return [float(value) for value in values if value is not None]


def _batch_acq_value(acq_value: Any, n: int) -> float | None:
    """Return a scalar value only when it represents the complete q-batch."""

    values = _flatten_acq_values(acq_value)
    return values[0] if n > 1 and len(values) == 1 else None


def _broadcast_acq_values(acq_value: Any, n: int) -> list[float | None]:
    """Return only genuine per-candidate acquisition values."""

    values = _flatten_acq_values(acq_value)
    if len(values) == n:
        return values
    if n == 1 and values:
        return [values[0]]
    return [None for _ in range(n)]
''',
)

# Web request schema.
replace_exact(
    "src/bochan/serving/webapp/app.py",
    "    sequential: bool = True\n\n\nclass KSparseSettingsSchema",
    "    sequential: bool = True\n"
    "    minimum_candidate_distance_ratio: float = Field(\n"
    "        default=1e-3, ge=0.0, le=1.0\n"
    "    )\n\n\nclass KSparseSettingsSchema",
)

# Web workflow helpers and final-space optimizer configuration.
replace_exact(
    "src/bochan/serving/webapp/workflows_tabular.py",
    "from bochan.desktop.services import (\n",
    "from bochan.api.candidate_uniqueness import count_unique_candidate_rows\n"
    "from bochan.desktop.services import (\n",
)
replace_exact(
    "src/bochan/serving/webapp/workflows_tabular.py",
    "    _build_visualizations,\n"
    "    _candidate_rows,\n"
    "    _figure_payload,\n",
    "    _batch_acq_value,\n"
    "    _build_visualizations,\n"
    "    _candidate_rows,\n"
    "    _figure_payload,\n",
)
replace_exact(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''def _response_task_type(
    target_settings: list[dict[str, Any]],
    target_columns: list[str],
) -> str:
    task_types = [setting["task_type"] for setting in target_settings]
    if len(set(task_types)) > 1:
        return "hybrid"
    if task_types[0] == "regression":
        return "multi_objective" if len(target_columns) > 1 else "regression"
    return str(task_types[0])


def run_regression_web_workflow''',
    '''def _response_task_type(
    target_settings: list[dict[str, Any]],
    target_columns: list[str],
) -> str:
    task_types = [setting["task_type"] for setting in target_settings]
    if len(set(task_types)) > 1:
        return "hybrid"
    if task_types[0] == "regression":
        return "multi_objective" if len(target_columns) > 1 else "regression"
    return str(task_types[0])


def _candidate_distance_tolerances(
    encoded: dict[str, Any],
    *,
    relative_distance: float,
) -> list[float]:
    """Resolve final-space minimum distances for every encoded feature."""

    ratio = float(relative_distance)
    if ratio < 0 or ratio > 1:
        raise ValueError("minimum_candidate_distance_ratio must be between 0 and 1.")
    lower = [float(value) for value in encoded["bounds"][0]]
    upper = [float(value) for value in encoded["bounds"][1]]
    categorical = {int(index) for index in encoded["cat_dims"]}
    fixed = {int(index) for index in encoded["fixed_features"]}
    steps = {int(index): abs(float(value)) for index, value in encoded["steps"].items()}

    tolerances: list[float] = []
    for index, (low, high) in enumerate(zip(lower, upper, strict=True)):
        if index in categorical or index in fixed:
            tolerances.append(0.0)
            continue
        range_distance = abs(high - low) * ratio
        resolution_distance = 0.5 * steps.get(index, 0.0)
        tolerances.append(max(range_distance, resolution_distance, 1e-12))
    return tolerances


def run_regression_web_workflow''',
)
replace_exact(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''    opt_config = OptimizeConfig(
        q=request.optimizer.q,
        num_restarts=request.optimizer.num_restarts,
        raw_samples=request.optimizer.raw_samples,
        sequential=request.optimizer.sequential,
        optimizer=_resolve_optimizer_value(resolved_optimizer),
        optimizer_kwargs=optimizer_kwargs,
        repair_config=repair_config,
        fixed_features=encoded_features["fixed_features"] or None,
        inequality_constraints=inequality_constraints or None,
        equality_constraints=equality_constraints or None,
    )
''',
    '''    minimum_distance_ratio = float(
        getattr(request.optimizer, "minimum_candidate_distance_ratio", 1e-3)
    )
    duplicate_tolerances = _candidate_distance_tolerances(
        encoded_features,
        relative_distance=minimum_distance_ratio,
    )

    def final_candidate_postprocess(value: Any) -> Any:
        return _postprocess_candidates(
            value,
            request=processing_request,
            encoded=encoded_features,
        )

    opt_config = OptimizeConfig(
        q=request.optimizer.q,
        num_restarts=request.optimizer.num_restarts,
        raw_samples=request.optimizer.raw_samples,
        sequential=request.optimizer.sequential,
        optimizer=_resolve_optimizer_value(resolved_optimizer),
        optimizer_kwargs=optimizer_kwargs,
        repair_config=repair_config,
        fixed_features=encoded_features["fixed_features"] or None,
        inequality_constraints=inequality_constraints or None,
        equality_constraints=equality_constraints or None,
        duplicate_tolerances=duplicate_tolerances,
        final_candidate_postprocess=final_candidate_postprocess,
    )
''',
)
replace_exact(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''    raw_candidates = candidate_result.candidates
    raw_acq_value = candidate_result.acq_value
    candidates = _postprocess_candidates(
        raw_candidates,
        request=processing_request,
        encoded=encoded_features,
    )
    timings_ms["candidate"] = round(
''',
    '''    raw_candidates = candidate_result.candidates
    raw_acq_value = candidate_result.acq_value
    candidates = _postprocess_candidates(
        raw_candidates,
        request=processing_request,
        encoded=encoded_features,
    )
    batch_acq_value = _batch_acq_value(
        raw_acq_value,
        int(candidates.shape[0]),
    )
    unique_candidate_count = count_unique_candidate_rows(
        candidates,
        tolerance=opt_config.duplicate_tolerance,
        tolerances=duplicate_tolerances,
    )
    uniqueness_warning = (
        None
        if unique_candidate_count == int(request.optimizer.q)
        else (
            f"要求した{request.optimizer.q}件のうち、最終実験条件で異なる候補は"
            f"{unique_candidate_count}件です。探索範囲、step、制約、または最小距離を"
            "見直してください。"
        )
    )
    timings_ms["candidate"] = round(
''',
)
replace_exact(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''        "raw_acq_value": to_serializable(raw_acq_value),
        "candidates": rows,
''',
    '''        "raw_acq_value": to_serializable(raw_acq_value),
        "batch_acq_value": batch_acq_value,
        "candidates": rows,
''',
)
replace_exact(
    "src/bochan/serving/webapp/workflows_tabular.py",
    '''            "repair_enabled": repair_config is not None,
            "n_feature_constraints": len(feature_constraints),
''',
    '''            "repair_enabled": repair_config is not None,
            "candidate_uniqueness": {
                "requested_q": int(request.optimizer.q),
                "unique_count": unique_candidate_count,
                "sequential": bool(request.optimizer.sequential),
                "minimum_distance_ratio": minimum_distance_ratio,
                "per_feature_tolerances": dict(
                    zip(feature_columns, duplicate_tolerances, strict=True)
                ),
                "warning": uniqueness_warning,
            },
            "n_feature_constraints": len(feature_constraints),
''',
)

# Frontend result types.
replace_exact(
    "web/src/types.ts",
    '''  candidates: CandidateRow[];
  visualizations: ResultVisualization[];
''',
    '''  candidates: CandidateRow[];
  /** Scalar acquisition value for the complete joint q-batch. */
  batch_acq_value?: number | null;
  visualizations: ResultVisualization[];
''',
)

# Browser request transport.
replace_exact(
    "web/src/api.ts",
    '''  q: number;
  numRestarts: number;
  rawSamples: number;
''',
    '''  q: number;
  sequential?: boolean;
  minimumCandidateDistanceRatio?: number;
  numRestarts: number;
  rawSamples: number;
''',
)
replace_exact(
    "web/src/api.ts",
    '''  if (!Number.isFinite(input.perturbationStd) || input.perturbationStd <= 0) {
    throw new Error("入力摂動のばらつきは0より大きくしてください。");
  }

  const noiseAlpha''',
    '''  if (!Number.isFinite(input.perturbationStd) || input.perturbationStd <= 0) {
    throw new Error("入力摂動のばらつきは0より大きくしてください。");
  }
  const minimumCandidateDistanceRatio = input.minimumCandidateDistanceRatio ?? 1e-3;
  if (
    !Number.isFinite(minimumCandidateDistanceRatio)
    || minimumCandidateDistanceRatio < 0
    || minimumCandidateDistanceRatio > 1
  ) {
    throw new Error("最小候補間距離は探索範囲比0〜100%で指定してください。");
  }

  const noiseAlpha''',
)
replace_exact(
    "web/src/api.ts",
    '''        sequential:
          input.searchSpace.some((variable) => variable.type === "categorical") ||
          searchMethod === "cmaes"
''',
    '''        sequential:
          Boolean(input.sequential ?? true) ||
          input.searchSpace.some((variable) => variable.type === "categorical") ||
          searchMethod === "cmaes",
        minimum_candidate_distance_ratio: minimumCandidateDistanceRatio
''',
)

# Workbench state and request assembly.
replace_exact(
    "web/src/context/WorkbenchContext.tsx",
    '''  q: number;
  setQ: (q: number) => void;
  numRestarts: number;
''',
    '''  q: number;
  setQ: (q: number) => void;
  sequential: boolean;
  setSequential: (sequential: boolean) => void;
  minimumCandidateDistanceRatio: number;
  setMinimumCandidateDistanceRatio: (ratio: number) => void;
  numRestarts: number;
''',
)
replace_exact(
    "web/src/context/WorkbenchContext.tsx",
    '''  const [q, setQ] = useState(3);
  const [numRestarts, setNumRestarts] = useState(10);
''',
    '''  const [q, setQ] = useState(3);
  const [sequential, setSequential] = useState(true);
  const [minimumCandidateDistanceRatio, setMinimumCandidateDistanceRatio] = useState(1e-3);
  const [numRestarts, setNumRestarts] = useState(10);
''',
)
replace_exact(
    "web/src/context/WorkbenchContext.tsx",
    '''    q,
    numRestarts,
    rawSamples,
''',
    '''    q,
    sequential,
    minimumCandidateDistanceRatio,
    numRestarts,
    rawSamples,
''',
)
replace_exact(
    "web/src/context/WorkbenchContext.tsx",
    '''      setAcquisition("EI");
      setStepState("prepare");
''',
    '''      setAcquisition("EI");
      setSequential(true);
      setMinimumCandidateDistanceRatio(1e-3);
      setStepState("prepare");
''',
)
replace_exact(
    "web/src/context/WorkbenchContext.tsx",
    '''      setQ(restored.q);
      setNumRestarts(restored.numRestarts);
''',
    '''      setQ(restored.q);
      setSequential(restored.sequential);
      setMinimumCandidateDistanceRatio(restored.minimumCandidateDistanceRatio);
      setNumRestarts(restored.numRestarts);
''',
)
replace_exact(
    "web/src/context/WorkbenchContext.tsx",
    '''    q,
    setQ,
    numRestarts,
''',
    '''    q,
    setQ,
    sequential,
    setSequential,
    minimumCandidateDistanceRatio,
    setMinimumCandidateDistanceRatio,
    numRestarts,
''',
)

# Artifact restoration.
replace_exact(
    "web/src/modelArtifactRestore.ts",
    '''  q: number;
  numRestarts: number;
''',
    '''  q: number;
  sequential: boolean;
  minimumCandidateDistanceRatio: number;
  numRestarts: number;
''',
)
replace_exact(
    "web/src/modelArtifactRestore.ts",
    '''  const q = Math.max(1, Math.trunc(finiteNumber(optimizerSettings.q, 3)));
  const numRestarts = Math.max(1, Math.trunc(finiteNumber(optimizerSettings.num_restarts, 10)));
''',
    '''  const q = Math.max(1, Math.trunc(finiteNumber(optimizerSettings.q, 3)));
  const sequential = optimizerSettings.sequential === undefined
    ? true
    : Boolean(optimizerSettings.sequential);
  const minimumCandidateDistanceRatio = Math.min(1, Math.max(
    0,
    finiteNumber(optimizerSettings.minimum_candidate_distance_ratio, 1e-3)
  ));
  const numRestarts = Math.max(1, Math.trunc(finiteNumber(optimizerSettings.num_restarts, 10)));
''',
)
replace_exact(
    "web/src/modelArtifactRestore.ts",
    '''    q,
    numRestarts,
    rawSamples,
''',
    '''    q,
    sequential,
    minimumCandidateDistanceRatio,
    numRestarts,
    rawSamples,
''',
    count=2,
)

# Candidate configuration UI.
replace_exact(
    "web/src/pages/OptimizePage.tsx",
    '''    q,
    setQ,
    numRestarts,
''',
    '''    q,
    setQ,
    sequential,
    setSequential,
    minimumCandidateDistanceRatio,
    setMinimumCandidateDistanceRatio,
    numRestarts,
''',
)
replace_exact(
    "web/src/pages/OptimizePage.tsx",
    '''  const projectedModel = modelType === "pca" || modelType === "rembo";

  const acquisitionOptions''',
    '''  const projectedModel = modelType === "pca" || modelType === "rembo";
  const sequentialForced = q > 1 && (
    selectedVariables.some((variable) => variable.type === "categorical")
    || searchMethod === "cmaes"
  );

  const acquisitionOptions''',
)
replace_exact(
    "web/src/pages/OptimizePage.tsx",
    '''    if (rawSamples < 1) errors.push("raw_samplesは1以上にしてください。");
    return errors;
''',
    '''    if (rawSamples < 1) errors.push("raw_samplesは1以上にしてください。");
    if (
      !Number.isFinite(minimumCandidateDistanceRatio)
      || minimumCandidateDistanceRatio < 0
      || minimumCandidateDistanceRatio > 1
    ) {
      errors.push("最小候補間距離は探索範囲比0〜100%で指定してください。");
    }
    return errors;
''',
)
replace_exact(
    "web/src/pages/OptimizePage.tsx",
    '''    modelType,
    multiObjective,
    numRestarts,
''',
    '''    minimumCandidateDistanceRatio,
    modelType,
    multiObjective,
    numRestarts,
''',
)
replace_exact(
    "web/src/pages/OptimizePage.tsx",
    '''          <label>q<input type="number" min={1} max={20} step={1} value={q} onChange={(event) => setQ(Number(event.target.value))} /></label>
          <label>num_restarts<input type="number" min={1} step={1} value={numRestarts} onChange={(event) => setNumRestarts(Number(event.target.value))} /></label>
          <label>raw_samples<input type="number" min={1} step={1} value={rawSamples} onChange={(event) => setRawSamples(Number(event.target.value))} /></label>
''',
    '''          <label>q<input type="number" min={1} max={20} step={1} value={q} onChange={(event) => setQ(Number(event.target.value))} /></label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={q > 1 && (sequential || sequentialForced)}
              disabled={q <= 1 || sequentialForced}
              onChange={(event) => setSequential(event.target.checked)}
            />
            逐次候補生成
          </label>
          <p className="settings-note">
            q &gt; 1で有効にすると、選択済み候補をpendingとして次候補を順番に探索します。
            カテゴリ変数とCMA-ESでは自動的に有効になります。
          </p>
          <label>
            最小候補間距離（探索範囲比 %）
            <input
              type="number"
              min={0}
              max={100}
              step={0.01}
              value={minimumCandidateDistanceRatio * 100}
              onChange={(event) => setMinimumCandidateDistanceRatio(Number(event.target.value) / 100)}
            />
          </label>
          <p className="settings-note">
            連続変数は探索範囲比、step指定変数は実験分解能、カテゴリ変数はカテゴリ一致で重複を判定します。
          </p>
          <label>num_restarts<input type="number" min={1} step={1} value={numRestarts} onChange={(event) => setNumRestarts(Number(event.target.value))} /></label>
          <label>raw_samples<input type="number" min={1} step={1} value={rawSamples} onChange={(event) => setRawSamples(Number(event.target.value))} /></label>
''',
)

# Results display: six decimal places, one batch value, optional warning.
replace_exact(
    "web/src/pages/ResultsPage.tsx",
    '''  return Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)
    ? value.toExponential(4)
    : value.toFixed(4).replace(/\\.0+$/, "").replace(/(\\.\\d*?)0+$/, "$1");
''',
    '''  return Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.000001)
    ? value.toExponential(6)
    : value.toFixed(6).replace(/\\.0+$/, "").replace(/(\\.\\d*?)0+$/, "$1");
''',
)
replace_exact(
    "web/src/pages/ResultsPage.tsx",
    '''  const candidates = [...completedResult.candidates].sort((left, right) => left.rank - right.rank);

  function downloadCandidates()''',
    '''  const candidates = [...completedResult.candidates].sort((left, right) => left.rank - right.rank);
  const batchAcqValue = completedResult.batch_acq_value;
  const showPerCandidateAcq = candidates.some((candidate) => candidate.acq_value !== null);
  const candidateUniqueness = completedResult.metadata?.candidate_uniqueness as
    | Record<string, unknown>
    | undefined;
  const candidateUniquenessWarning = typeof candidateUniqueness?.warning === "string"
    ? candidateUniqueness.warning
    : null;

  function downloadCandidates()''',
)
replace_exact(
    "web/src/pages/ResultsPage.tsx",
    '''            <span className={`status-chip ${staleAfterAppend ? "warning" : "success"}`}>
              {candidates.length} candidates{staleAfterAppend ? " · stale" : ""}
            </span>
          </div>
          <div className="table-wrap">
''',
    '''            <div>
              <span className={`status-chip ${staleAfterAppend ? "warning" : "success"}`}>
                {candidates.length} candidates{staleAfterAppend ? " · stale" : ""}
              </span>
              {batchAcqValue !== null && batchAcqValue !== undefined && (
                <span className="status-chip">
                  {candidates.length}候補全体の獲得値 {formatNumber(batchAcqValue)}
                </span>
              )}
            </div>
          </div>
          {candidateUniquenessWarning && (
            <div className="alert warning">{candidateUniquenessWarning}</div>
          )}
          <div className="table-wrap">
''',
)
replace_exact(
    "web/src/pages/ResultsPage.tsx",
    '''                  <th>獲得値</th>
                  <th>条件</th>
''',
    '''                  {showPerCandidateAcq && <th>候補別獲得値</th>}
                  <th>条件</th>
''',
)
replace_exact(
    "web/src/pages/ResultsPage.tsx",
    '''                    <td>{formatNumber(candidate.acq_value)}</td>
                    <td>
''',
    '''                    {showPerCandidateAcq && <td>{formatNumber(candidate.acq_value)}</td>}
                    <td>
''',
)

write(
    "tests/test_regression_final_candidate_uniqueness_e2e.py",
    '''from __future__ import annotations

import torch

from bochan.api import OptimizeConfig
from bochan.api.candidate_uniqueness import count_unique_candidate_rows
from bochan.api.optimizer_dispatch import optimize_candidates
from bochan.serving.webapp.target_results import (
    _batch_acq_value,
    _broadcast_acq_values,
)
from bochan.serving.webapp.workflows_tabular import _candidate_distance_tolerances


class _PendingAwareRegressionAcquisition(torch.nn.Module):
    """Smooth acquisition whose unconstrained q points share one optimum."""

    def __init__(self) -> None:
        super().__init__()
        self.X_pending = None

    def set_X_pending(self, X_pending=None) -> None:
        self.X_pending = X_pending

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        score = -((X - 0.5) ** 2).sum(dim=(-1, -2))
        if self.X_pending is None or self.X_pending.numel() == 0:
            return score
        pending = self.X_pending.to(device=X.device, dtype=X.dtype).reshape(
            -1, X.shape[-1]
        )
        distance_squared = (
            X.unsqueeze(-2) - pending.view(*([1] * (X.ndim - 2)), 1, *pending.shape)
        ).pow(2).sum(dim=-1)
        repulsion = torch.exp(-distance_squared / 0.0025).sum(dim=(-1, -2))
        return score - 10.0 * repulsion


def test_actual_optimize_acqf_q3_refills_unique_final_regression_candidates() -> None:
    torch.manual_seed(0)
    acquisition = _PendingAwareRegressionAcquisition()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def final_postprocess(X: torch.Tensor) -> torch.Tensor:
        return (torch.round(X / 0.1) * 0.1).clamp(0.0, 1.0)

    candidates, acq_value = optimize_candidates(
        acqf=acquisition,
        bounds=bounds,
        config=OptimizeConfig(
            q=3,
            num_restarts=8,
            raw_samples=128,
            sequential=False,
            duplicate_tolerances=[0.049],
            duplicate_pool_restarts=8,
            duplicate_refill_attempts=4,
            final_candidate_postprocess=final_postprocess,
            optimizer_kwargs={"options": {"maxiter": 100}},
        ),
    )

    assert candidates.shape == (3, 1)
    assert torch.allclose(candidates, final_postprocess(candidates))
    assert count_unique_candidate_rows(
        candidates,
        tolerances=[0.049],
    ) == 3
    assert torch.isfinite(torch.as_tensor(acq_value)).all()


def test_web_minimum_distances_follow_range_step_and_category_resolution() -> None:
    encoded = {
        "bounds": [[0.0, 0.0, 0.0, 0.0], [10.0, 4.0, 2.0, 1.0]],
        "cat_dims": [2],
        "fixed_features": {3: 1.0},
        "steps": {1: 1.0},
    }

    tolerances = _candidate_distance_tolerances(
        encoded,
        relative_distance=1e-3,
    )

    assert tolerances == [0.01, 0.5, 0.0, 0.0]


def test_joint_batch_acquisition_value_is_not_repeated_per_candidate() -> None:
    joint_value = torch.tensor(0.3304, dtype=torch.double)

    assert _batch_acq_value(joint_value, 3) == 0.3304
    assert _broadcast_acq_values(joint_value, 3) == [None, None, None]

    point_values = torch.tensor([0.3, 0.2, 0.1], dtype=torch.double)
    assert _batch_acq_value(point_values, 3) is None
    assert _broadcast_acq_values(point_values, 3) == [0.3, 0.2, 0.1]


def test_candidate_config_validates_featurewise_tolerances_and_postprocess() -> None:
    config = OptimizeConfig(
        q=3,
        duplicate_tolerances=[0.01, 0.0],
        final_candidate_postprocess=lambda X: X,
    )

    assert config.duplicate_tolerances == (0.01, 0.0)
    assert callable(config.final_candidate_postprocess)
''',
)

print("Applied regression final-candidate uniqueness changes.")
