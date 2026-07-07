"""LLM candidate-set optimizer.

LLM は候補集合を広めに生成し、最終選抜は既存 acquisition function が行います。
この optimizer は LLM を最終裁定者にしないための軽量 backend です。
"""

from __future__ import annotations

from typing import Any

from bochan.llm import build_llm_candidate_set, sanitize_candidate_set, select_top_candidates


def optimize_acqf_llm_candidate_set(
    acq_function: Any,
    bounds: Any,
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 256,
    return_best_only: bool = True,
    sequential: bool = False,
    post_processing_func: Any | None = None,
    fixed_features: dict[int, float] | None = None,
    inequality_constraints: Any | None = None,
    equality_constraints: Any | None = None,
    llm_config: Any | None = None,
    llm_context: Any | None = None,
    goal: Any | None = None,
    n_llm_candidates: int | None = None,
    candidate_set: Any | None = None,
    prompt: str | None = None,
    history_summary: dict[str, Any] | None = None,
    pending_candidates: Any | None = None,
    fallback_sobol: bool = True,
    seed: int | None = None,
    constraint_tolerance: float = 1e-6,
    duplicate_tolerance: float = 1e-9,
    options: dict[str, Any] | None = None,
    **_: Any,
) -> tuple[Any, Any]:
    """Generate many LLM candidates and rerank them with ``acq_function``.

    Args:
        acq_function: 既存の BoTorch / bochan acquisition function。
        bounds: 探索範囲 ``[2, d]``。
        q: 返す候補数。
        raw_samples: ``n_llm_candidates`` 未指定時の候補集合サイズの下限に使います。
        llm_config: OpenAI / Gemini などの provider 設定。
        llm_context: 変数説明・目的変数説明・ドメインノート。
        goal: 自然言語の探索目的。候補生成 prompt の補足文脈です。
        candidate_set: 明示候補集合。指定時は LLM を呼ばず acquisition reranking のみ行います。
        post_processing_func: 既存の grid / k-sparse / 制約補修関数。

    Notes:
        初期実装では、LLM 候補を q=1 acquisition value で個別評価し、上位 q 件を返します。
        joint q-batch acquisition の厳密最適化ではなく、candidate-set reranking として使います。
    """

    merged_options = dict(options or {})
    llm_config = merged_options.pop("llm_config", llm_config)
    llm_context = merged_options.pop("llm_context", llm_context)
    goal = merged_options.pop("goal", goal)
    n_llm_candidates = merged_options.pop("n_llm_candidates", n_llm_candidates)
    candidate_set = merged_options.pop("candidate_set", candidate_set)
    prompt = merged_options.pop("prompt", prompt)
    history_summary = merged_options.pop("history_summary", history_summary)
    pending_candidates = merged_options.pop("pending_candidates", pending_candidates)
    fallback_sobol = bool(merged_options.pop("fallback_sobol", fallback_sobol))
    seed = merged_options.pop("seed", seed)
    constraint_tolerance = float(merged_options.pop("constraint_tolerance", constraint_tolerance))
    duplicate_tolerance = float(merged_options.pop("duplicate_tolerance", duplicate_tolerance))

    target_n = int(n_llm_candidates or max(int(raw_samples), int(num_restarts) * max(int(q), 1), int(q)))
    candidates = build_llm_candidate_set(
        bounds=bounds,
        n_candidates=target_n,
        llm_config=llm_config,
        llm_context=llm_context,
        goal=goal,
        candidate_set=candidate_set,
        prompt=prompt,
        acquisition_name=getattr(acq_function, "__class__", type(acq_function)).__name__,
        history_summary=history_summary,
        pending_candidates=pending_candidates,
        fallback_sobol=fallback_sobol,
        seed=seed,
    )
    candidates = sanitize_candidate_set(
        candidates,
        bounds=bounds,
        fixed_features=fixed_features,
        post_processing_func=post_processing_func,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        constraint_tolerance=constraint_tolerance,
        duplicate_tolerance=duplicate_tolerance,
    )
    return select_top_candidates(acq_function, candidates, q=int(q))
