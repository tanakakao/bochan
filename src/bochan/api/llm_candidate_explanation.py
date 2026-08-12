"""Candidate explanation mixin for the canonical Bayesian optimizer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _as_2d_tensor(value: Any) -> torch.Tensor:
    tensor = (
        value.detach().cpu()
        if hasattr(value, "detach")
        else torch.as_tensor(value)
    )
    tensor = tensor.to(dtype=torch.double)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(
            f"candidates must be a 2D matrix. Got shape={tuple(tensor.shape)}."
        )
    return tensor


def _select_rows(value: Any, indices: Sequence[int]) -> Any:
    if hasattr(value, "index_select"):
        index = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=getattr(value, "device", None),
        )
        return value.index_select(0, index)
    return [value[index] for index in indices]


def _safe_bounds_tensor(
    bounds: Any,
    X: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve finite column bounds without torch.nanmin / torch.nanmax."""

    if bounds is not None:
        try:
            bounds_tensor = torch.as_tensor(bounds, dtype=torch.double)
            if bounds_tensor.shape == (2, X.shape[-1]):
                return bounds_tensor[0], bounds_tensor[1]
        except (TypeError, ValueError, RuntimeError):
            pass

    lower: list[float] = []
    upper: list[float] = []
    for column in range(X.shape[-1]):
        values = X[:, column]
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            lower.append(0.0)
            upper.append(1.0)
        else:
            lower.append(float(finite.min().item()))
            upper.append(float(finite.max().item()))
    return (
        torch.tensor(lower, dtype=torch.double),
        torch.tensor(upper, dtype=torch.double),
    )


def _per_candidate_acquisition(
    acq_value: Any,
    n_candidates: int,
) -> torch.Tensor | None:
    if acq_value is None:
        return None
    try:
        values = torch.as_tensor(acq_value, dtype=torch.double).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return None
    if values.numel() != n_candidates:
        return None
    return values


def _select_representative_candidates(
    candidates: Any,
    *,
    acq_value: Any | None = None,
    max_representatives: int = 5,
    bounds: Any | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Select best, central and diverse representatives without global patches."""

    X = _as_2d_tensor(candidates)
    n_candidates = int(X.shape[0])
    if n_candidates == 0:
        raise ValueError("candidates must contain at least one row.")
    limit = min(max(int(max_representatives), 1), n_candidates)
    if n_candidates <= limit:
        indices = list(range(n_candidates))
        return indices, {index: "all_candidates" for index in indices}

    lower, upper = _safe_bounds_tensor(bounds, X)
    scale = (upper - lower).abs().clamp_min(1e-12)
    normalized = torch.nan_to_num(
        (X - lower) / scale,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    selected: list[int] = []
    roles: dict[int, str] = {}
    per_candidate_acq = _per_candidate_acquisition(acq_value, n_candidates)
    if per_candidate_acq is not None and torch.isfinite(per_candidate_acq).any():
        safe_values = torch.where(
            torch.isfinite(per_candidate_acq),
            per_candidate_acq,
            torch.full_like(per_candidate_acq, -torch.inf),
        )
        first = int(torch.argmax(safe_values).item())
        selected.append(first)
        roles[first] = "highest_acquisition"
    else:
        selected.append(0)
        roles[0] = "optimizer_first"

    if len(selected) < limit:
        centroid = normalized.mean(dim=0)
        distances = torch.linalg.vector_norm(normalized - centroid, dim=-1)
        for index in selected:
            distances[index] = torch.inf
        central = int(torch.argmin(distances).item())
        if central not in selected:
            selected.append(central)
            roles[central] = "central_candidate"

    while len(selected) < limit:
        chosen = normalized[selected]
        pairwise = torch.cdist(normalized, chosen)
        min_distance = pairwise.min(dim=1).values
        for index in selected:
            min_distance[index] = -torch.inf
        next_index = int(torch.argmax(min_distance).item())
        selected.append(next_index)
        roles[next_index] = "diverse_candidate"

    return selected, roles


def _coerce_config(value: Any) -> Any:
    from bochan.llm.candidate_explainer_overall import CandidateExplanationConfig

    if value is None:
        return CandidateExplanationConfig()
    if isinstance(value, CandidateExplanationConfig):
        return value
    return CandidateExplanationConfig(**dict(value))


class LLMCandidateExplanationMixin:
    """Domain-aware explanation methods for final candidate batches."""

    def explain_candidates(
        self,
        result: Any | None = None,
        *,
        candidates: Any | None = None,
        acq_value: Any | None = None,
        config: Any | None = None,
        max_representatives: int | None = None,
        perspectives: Sequence[str] | None = None,
        prompt: str | None = None,
        goal: Any | None = None,
        llm_config: Any | None = None,
        llm_context: Any | None = None,
        explanation_response: Any | None = None,
    ) -> Any:
        """Explain final candidates without regenerating or mutating them."""

        from bochan.llm.candidate_explainer_overall import (
            build_candidate_explanation_prompt,
            explanation_from_payload,
        )
        from bochan.llm.client import make_llm_client
        from bochan.llm.parser import parse_json_payload

        resolved_config = _coerce_config(config)
        if max_representatives is not None:
            resolved_config.max_representatives = int(max_representatives)
            if resolved_config.max_representatives <= 0:
                raise ValueError("max_representatives must be positive.")
        if perspectives is not None:
            resolved_config.perspectives = tuple(
                str(item) for item in perspectives
            )
        if prompt is not None:
            resolved_config.prompt = str(prompt)

        candidate_result = result
        if candidate_result is None and candidates is None:
            history = getattr(self, "history", None) or []
            if not history:
                raise ValueError(
                    "No candidate result is available. Pass result/candidates or "
                    "call candidate() first."
                )
            candidate_result = history[-1]
        if candidate_result is not None:
            candidates = candidate_result.candidates
            acq_value = candidate_result.acq_value
        if candidates is None:
            raise ValueError("candidates are required for explanation.")

        bounds = getattr(self, "bounds", None)
        if candidate_result is not None:
            context = getattr(candidate_result, "data_context", None)
            context_bounds = getattr(context, "bounds", None)
            if context_bounds is not None:
                bounds = context_bounds

        representative_indices, roles = _select_representative_candidates(
            candidates,
            acq_value=acq_value,
            max_representatives=resolved_config.max_representatives,
            bounds=bounds,
        )
        selected_candidates = _select_rows(
            candidates,
            representative_indices,
        )

        prediction_mean = None
        prediction_variance = None
        local_warnings: list[str] = []
        if resolved_config.include_predictions:
            try:
                prediction = self.predict(
                    selected_candidates,
                    return_result=True,
                )
                prediction_mean = getattr(prediction, "mean", None)
                if resolved_config.include_uncertainty:
                    prediction_variance = getattr(prediction, "variance", None)
            except Exception as exc:
                local_warnings.append(
                    "Candidate prediction summary was unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )

        settings = getattr(self, "llm_settings", None)
        resolved_goal = goal
        resolved_llm_config = llm_config
        resolved_llm_context = llm_context
        if settings is not None:
            if resolved_goal is None:
                resolved_goal = settings.goal
            if resolved_llm_config is None:
                resolved_llm_config = settings.llm_config
            if resolved_llm_context is None:
                resolved_llm_context = settings.llm_context
        if resolved_goal is None:
            resolved_goal = (
                "Explain why the proposed experimental or process conditions "
                "are informative."
            )

        acq_config = getattr(candidate_result, "acq_config", None)
        opt_config = getattr(candidate_result, "opt_config", None)
        explanation_prompt = build_candidate_explanation_prompt(
            goal=resolved_goal,
            candidates=candidates,
            representative_indices=representative_indices,
            representative_roles=roles,
            llm_context=resolved_llm_context,
            train_X=getattr(self, "train_X", None),
            train_Y=getattr(self, "train_Y", None),
            prediction_mean=prediction_mean,
            prediction_variance=prediction_variance,
            model_config=getattr(self, "model_config", None),
            acquisition_config=acq_config,
            optimize_config=opt_config,
            perspectives=resolved_config.perspectives,
            language=resolved_config.language,
            additional_prompt=resolved_config.prompt,
        )
        if explanation_response is None:
            client = make_llm_client(resolved_llm_config)
            payload = parse_json_payload(
                client.generate_json(explanation_prompt).text
            )
        else:
            payload = parse_json_payload(explanation_response)
        if not isinstance(payload, Mapping):
            raise ValueError("Candidate explanation response must be a JSON object.")

        total_candidates = int(_as_2d_tensor(candidates).shape[0])
        explanation = explanation_from_payload(
            payload,
            total_candidates=total_candidates,
            representative_indices=representative_indices,
        )
        explanation.warnings = local_warnings + explanation.warnings
        self.last_candidate_explanation = explanation
        self.last_candidate_explanation_prompt = explanation_prompt
        if candidate_result is not None:
            candidate_result.explanation = explanation
        return explanation

    def explain_last_candidates(self, **kwargs: Any) -> Any:
        """Explain the most recent candidate result."""

        return self.explain_candidates(None, **kwargs)


def install_bayesian_optimizer_candidate_explanation_api(
    optimizer_cls: type[Any],
) -> None:
    """Deprecated no-op retained for source compatibility."""

    del optimizer_cls


__all__ = [
    "LLMCandidateExplanationMixin",
    "install_bayesian_optimizer_candidate_explanation_api",
]
