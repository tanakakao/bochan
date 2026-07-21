"""Install domain-aware candidate explanation methods on BayesianOptimizer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from bochan.llm import candidate_explainer as _candidate_explainer_base
from bochan.llm import candidate_explainer_overall as _candidate_explainer
from bochan.llm.candidate_explainer_overall import (
    CandidateExplanation,
    CandidateExplanationConfig,
    build_candidate_explanation_prompt,
    explanation_from_payload,
    select_representative_candidates,
)
from bochan.llm.client import make_llm_client
from bochan.llm.parser import parse_json_payload


def _safe_bounds_tensor(bounds: Any, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve finite column bounds without relying on unavailable torch.nanmin."""

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
    return torch.tensor(lower, dtype=torch.double), torch.tensor(upper, dtype=torch.double)


_candidate_explainer_base._bounds_tensor = _safe_bounds_tensor
_candidate_explainer._bounds_tensor = _safe_bounds_tensor


def _coerce_config(
    value: CandidateExplanationConfig | Mapping[str, Any] | None,
) -> CandidateExplanationConfig:
    if value is None:
        return CandidateExplanationConfig()
    if isinstance(value, CandidateExplanationConfig):
        return value
    return CandidateExplanationConfig(**dict(value))


def install_bayesian_optimizer_candidate_explanation_api(optimizer_cls: type[Any]) -> None:
    """Attach candidate explanation methods to ``optimizer_cls`` once."""

    if getattr(optimizer_cls, "_bochan_candidate_explanation_api_installed", False):
        return

    def explain_candidates(
        self: Any,
        result: Any | None = None,
        *,
        candidates: Any | None = None,
        acq_value: Any | None = None,
        config: CandidateExplanationConfig | Mapping[str, Any] | None = None,
        max_representatives: int | None = None,
        perspectives: Sequence[str] | None = None,
        prompt: str | None = None,
        goal: Any | None = None,
        llm_config: Any | None = None,
        llm_context: Any | None = None,
        explanation_response: Any | None = None,
    ) -> CandidateExplanation:
        """Explain final candidates from model and engineering perspectives.

        This method never regenerates candidates. For large batches it selects a
        highest-ranked, central, and diverse subset before the LLM call. Each
        representative point includes both specialist perspectives and one
        integrated, decision-oriented explanation.
        """

        resolved_config = _coerce_config(config)
        if max_representatives is not None:
            resolved_config.max_representatives = int(max_representatives)
            if resolved_config.max_representatives <= 0:
                raise ValueError("max_representatives must be positive.")
        if perspectives is not None:
            resolved_config.perspectives = tuple(str(item) for item in perspectives)
        if prompt is not None:
            resolved_config.prompt = str(prompt)

        candidate_result = result
        if candidate_result is None and candidates is None:
            history = getattr(self, "history", None) or []
            if not history:
                raise ValueError(
                    "No candidate result is available. Pass result/candidates or call candidate() first."
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

        representative_indices, roles = select_representative_candidates(
            candidates,
            acq_value=acq_value,
            max_representatives=resolved_config.max_representatives,
            bounds=bounds,
        )
        selected_candidates = _candidate_explainer._select_rows(
            candidates,
            representative_indices,
        )

        prediction_mean = None
        prediction_variance = None
        local_warnings: list[str] = []
        if resolved_config.include_predictions:
            try:
                prediction = self.predict(selected_candidates, return_result=True)
                prediction_mean = getattr(prediction, "mean", None)
                if resolved_config.include_uncertainty:
                    prediction_variance = getattr(prediction, "variance", None)
            except Exception as exc:  # custom models may not expose a standard posterior
                local_warnings.append(
                    f"Candidate prediction summary was unavailable: {type(exc).__name__}: {exc}"
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
            resolved_goal = "Explain why the proposed experimental or process conditions are informative."

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
            payload = parse_json_payload(client.generate_json(explanation_prompt).text)
        else:
            payload = parse_json_payload(explanation_response)
        if not isinstance(payload, Mapping):
            raise ValueError("Candidate explanation response must be a JSON object.")

        total_candidates = int(_candidate_explainer._as_2d_tensor(candidates).shape[0])
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

    def explain_last_candidates(self: Any, **kwargs: Any) -> CandidateExplanation:
        """Explain the most recent candidate result."""

        return self.explain_candidates(None, **kwargs)

    optimizer_cls.explain_candidates = explain_candidates
    optimizer_cls.explain_last_candidates = explain_last_candidates
    optimizer_cls._bochan_candidate_explanation_api_installed = True


__all__ = ["install_bayesian_optimizer_candidate_explanation_api"]
