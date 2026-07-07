"""LLM candidate-set generation and acquisition reranking helpers."""

from __future__ import annotations

from typing import Any

from .client import make_llm_client
from .configs import coerce_llm_context
from .parser import candidates_to_tensor, clip_to_bounds, remove_duplicate_rows, remove_nonfinite
from .prompt_builder import build_candidate_prompt


def _apply_fixed_features(candidates: Any, fixed_features: dict[int, float] | None) -> Any:
    if not fixed_features:
        return candidates
    candidates = candidates.clone()
    for index, value in fixed_features.items():
        candidates[..., int(index)] = float(value)
    return candidates


def _linear_constraints_mask(
    candidates: Any,
    *,
    inequality_constraints: Any | None = None,
    equality_constraints: Any | None = None,
    tolerance: float = 1e-6,
) -> Any:
    import torch

    mask = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)
    for constraint in inequality_constraints or []:
        indices, coefficients, rhs = constraint
        lhs = (candidates[..., indices] * coefficients).sum(dim=-1)
        mask &= lhs >= float(rhs) - tolerance
    for constraint in equality_constraints or []:
        indices, coefficients, rhs = constraint
        lhs = (candidates[..., indices] * coefficients).sum(dim=-1)
        mask &= (lhs - float(rhs)).abs() <= tolerance
    return mask


def _sobol_candidates(bounds: Any, n: int, *, seed: int | None = None) -> Any:
    import torch

    if n <= 0:
        return bounds[:0].T
    engine = torch.quasirandom.SobolEngine(dimension=int(bounds.shape[-1]), scramble=True, seed=seed)
    unit = engine.draw(n).to(dtype=bounds.dtype, device=bounds.device)
    return bounds[0] + (bounds[1] - bounds[0]) * unit


def build_llm_candidate_set(
    *,
    bounds: Any,
    n_candidates: int,
    llm_config: Any | None = None,
    llm_context: Any | None = None,
    goal: Any | None = None,
    candidate_set: Any | None = None,
    prompt: str | None = None,
    acquisition_name: str | None = None,
    history_summary: dict[str, Any] | None = None,
    pending_candidates: Any | None = None,
    fallback_sobol: bool = True,
    seed: int | None = None,
) -> Any:
    """LLM または明示 candidate_set から候補集合 tensor を作る。

    ``candidate_set`` が指定された場合は LLM を呼びません。テストやオフライン利用では
    この経路を使うと API key なしで同じ reranking 処理を確認できます。
    """

    context = coerce_llm_context(llm_context)
    target_n = int(n_candidates)

    if candidate_set is None:
        if prompt is None:
            prompt = build_candidate_prompt(
                bounds=bounds,
                n_candidates=target_n,
                llm_context=context,
                goal=goal,
                acquisition_name=acquisition_name,
                history_summary=history_summary,
                pending_candidates=pending_candidates,
            )
        client = make_llm_client(llm_config)
        response = client.generate_json(prompt)
        candidate_set = response.text

    candidates = candidates_to_tensor(candidate_set, bounds=bounds, variable_names=context.variable_names)
    candidates = remove_nonfinite(candidates)
    candidates = clip_to_bounds(candidates, bounds)
    candidates = remove_duplicate_rows(candidates)

    if fallback_sobol and candidates.shape[0] < target_n:
        supplement = _sobol_candidates(bounds, target_n - candidates.shape[0], seed=seed)
        candidates = remove_duplicate_rows(candidates=torch.cat([candidates, supplement], dim=0))
    return candidates


def sanitize_candidate_set(
    candidates: Any,
    *,
    bounds: Any,
    fixed_features: dict[int, float] | None = None,
    post_processing_func: Any | None = None,
    inequality_constraints: Any | None = None,
    equality_constraints: Any | None = None,
    constraint_tolerance: float = 1e-6,
    duplicate_tolerance: float = 1e-9,
) -> Any:
    """bounds / fixed features / post-processing / constraints を適用する。"""

    candidates = remove_nonfinite(candidates)
    candidates = clip_to_bounds(candidates, bounds)
    candidates = _apply_fixed_features(candidates, fixed_features)
    if post_processing_func is not None:
        candidates = post_processing_func(candidates)
        candidates = candidates.to(dtype=bounds.dtype, device=bounds.device)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
    candidates = clip_to_bounds(candidates, bounds)
    if inequality_constraints is not None or equality_constraints is not None:
        mask = _linear_constraints_mask(
            candidates,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            tolerance=constraint_tolerance,
        )
        candidates = candidates[mask]
    return remove_duplicate_rows(candidates, tolerance=duplicate_tolerance)


def score_candidate_set(acq_function: Any, candidates: Any) -> Any:
    """各候補を q=1 acquisition value でスコアリングする。"""

    import torch

    if candidates.numel() == 0:
        return torch.empty(0, dtype=candidates.dtype, device=candidates.device)
    with torch.no_grad():
        values = acq_function(candidates.unsqueeze(-2))
    if not torch.is_tensor(values):
        values = torch.as_tensor(values, dtype=candidates.dtype, device=candidates.device)
    values = values.detach()
    if values.ndim == 0:
        values = values.reshape(1)
    return values.reshape(candidates.shape[0], -1).mean(dim=-1)


def select_top_candidates(acq_function: Any, candidates: Any, *, q: int) -> tuple[Any, Any]:
    """候補集合から acquisition value 上位 q 件を返す。"""

    import torch

    scores = score_candidate_set(acq_function, candidates)
    if scores.numel() == 0:
        raise ValueError("No valid candidates remained after LLM candidate validation and repair.")
    k = min(int(q), int(scores.numel()))
    top = torch.topk(scores, k=k).indices
    return candidates[top], scores[top]
