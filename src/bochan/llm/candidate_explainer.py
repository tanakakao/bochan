"""Domain-aware explanations for final BayesianOptimizer candidates.

The explainer keeps candidate generation and domain interpretation separate. It
summarizes model evidence and asks an LLM to discuss representative candidates
from physical, chemical, manufacturing, and development perspectives without
presenting speculative mechanisms as established facts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import torch

from .client import make_llm_client
from .configs import coerce_goal_config, coerce_llm_context
from .parser import parse_json_payload

_DEFAULT_PERSPECTIVES = (
    "physics",
    "chemistry",
    "manufacturing",
    "development",
)


@dataclass
class CandidateExplanationConfig:
    """Control representative selection and the domain explanation prompt."""

    max_representatives: int = 5
    perspectives: Sequence[str] = _DEFAULT_PERSPECTIVES
    representative_strategy: str = "acquisition_diverse"
    include_predictions: bool = True
    include_uncertainty: bool = True
    language: str = "ja"
    prompt: str | None = None

    def __post_init__(self) -> None:
        self.max_representatives = int(self.max_representatives)
        if self.max_representatives <= 0:
            raise ValueError("max_representatives must be positive.")
        self.perspectives = tuple(str(item) for item in self.perspectives)
        if not self.perspectives:
            raise ValueError("perspectives must not be empty.")


@dataclass
class CandidatePointExplanation:
    """Explanation for one representative candidate."""

    candidate_index: int
    representative_role: str = "representative"
    headline: str = ""
    model_evidence: list[str] = field(default_factory=list)
    physical_interpretation: list[str] = field(default_factory=list)
    chemical_interpretation: list[str] = field(default_factory=list)
    manufacturing_interpretation: list[str] = field(default_factory=list)
    development_interpretation: list[str] = field(default_factory=list)
    risks_and_tradeoffs: list[str] = field(default_factory=list)
    recommended_checks: list[str] = field(default_factory=list)
    confidence: str = "unknown"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidatePointExplanation":
        data = dict(value)
        return cls(
            candidate_index=int(data.get("candidate_index", -1)),
            representative_role=str(data.get("representative_role") or "representative"),
            headline=str(data.get("headline") or ""),
            model_evidence=_string_list(data.get("model_evidence")),
            physical_interpretation=_string_list(data.get("physical_interpretation")),
            chemical_interpretation=_string_list(data.get("chemical_interpretation")),
            manufacturing_interpretation=_string_list(data.get("manufacturing_interpretation")),
            development_interpretation=_string_list(data.get("development_interpretation")),
            risks_and_tradeoffs=_string_list(data.get("risks_and_tradeoffs")),
            recommended_checks=_string_list(data.get("recommended_checks")),
            confidence=str(data.get("confidence") or "unknown"),
        )


@dataclass
class CandidateExplanation:
    """Structured explanation returned by ``BayesianOptimizer.explain_candidates``."""

    total_candidates: int
    representative_indices: list[int]
    omitted_count: int
    summary: str = ""
    selection_note: str = ""
    common_patterns: list[str] = field(default_factory=list)
    candidate_explanations: list[CandidatePointExplanation] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "total_candidates": self.total_candidates,
            "representative_indices": list(self.representative_indices),
            "omitted_count": self.omitted_count,
            "summary": self.summary,
            "selection_note": self.selection_note,
            "common_patterns": list(self.common_patterns),
            "candidate_explanations": [asdict(item) for item in self.candidate_explanations],
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "raw_response": _to_jsonable(self.raw_response),
        }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def _as_2d_tensor(value: Any) -> torch.Tensor:
    tensor = value.detach().cpu() if hasattr(value, "detach") else torch.as_tensor(value)
    tensor = tensor.to(dtype=torch.double)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(f"candidates must be a 2D matrix. Got shape={tuple(tensor.shape)}.")
    return tensor


def _bounds_tensor(bounds: Any, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if bounds is not None:
        try:
            bounds_tensor = torch.as_tensor(bounds, dtype=torch.double)
            if bounds_tensor.shape == (2, X.shape[-1]):
                return bounds_tensor[0], bounds_tensor[1]
        except (TypeError, ValueError, RuntimeError):
            pass
    return torch.nanmin(X, dim=0).values, torch.nanmax(X, dim=0).values


def _per_candidate_acquisition(acq_value: Any, n_candidates: int) -> torch.Tensor | None:
    if acq_value is None:
        return None
    try:
        values = torch.as_tensor(acq_value, dtype=torch.double).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return None
    if values.numel() != n_candidates:
        return None
    return values


def select_representative_candidates(
    candidates: Any,
    *,
    acq_value: Any | None = None,
    max_representatives: int = 5,
    bounds: Any | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Select a best/central/diverse subset without sending every point to the LLM."""

    X = _as_2d_tensor(candidates)
    n_candidates = int(X.shape[0])
    if n_candidates == 0:
        raise ValueError("candidates must contain at least one row.")
    limit = min(max(int(max_representatives), 1), n_candidates)
    if n_candidates <= limit:
        indices = list(range(n_candidates))
        return indices, {index: "all_candidates" for index in indices}

    lower, upper = _bounds_tensor(bounds, X)
    scale = (upper - lower).abs().clamp_min(1e-12)
    normalized = torch.nan_to_num((X - lower) / scale, nan=0.0, posinf=1.0, neginf=0.0)

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


def _select_rows(value: Any, indices: Sequence[int]) -> Any:
    if hasattr(value, "index_select"):
        index = torch.as_tensor(indices, dtype=torch.long, device=getattr(value, "device", None))
        return value.index_select(0, index)
    return [value[index] for index in indices]


def _matrix_summary(value: Any, names: Sequence[str] | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    try:
        matrix = _as_2d_tensor(value)
    except (TypeError, ValueError, RuntimeError):
        return []
    resolved_names = list(names or [])
    summary: list[dict[str, Any]] = []
    for column in range(matrix.shape[-1]):
        values = matrix[:, column]
        finite = values[torch.isfinite(values)]
        name = resolved_names[column] if column < len(resolved_names) else str(column)
        if finite.numel() == 0:
            summary.append({"name": name, "index": column, "finite_count": 0})
            continue
        summary.append(
            {
                "name": name,
                "index": column,
                "finite_count": int(finite.numel()),
                "min": float(finite.min().item()),
                "max": float(finite.max().item()),
                "mean": float(finite.mean().item()),
            }
        )
    return summary


def _row_dicts(value: Any, names: Sequence[str] | None, n_rows: int) -> list[dict[str, Any]]:
    if value is None:
        return [{} for _ in range(n_rows)]
    try:
        tensor = value.detach().cpu() if hasattr(value, "detach") else torch.as_tensor(value)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.shape[0] != n_rows:
            return [{} for _ in range(n_rows)]
        data = tensor.tolist()
    except (TypeError, ValueError, RuntimeError):
        return [{} for _ in range(n_rows)]
    resolved_names = list(names or [])
    rows: list[dict[str, Any]] = []
    for row in data:
        row_values = row if isinstance(row, list) else [row]
        rows.append(
            {
                resolved_names[index] if index < len(resolved_names) else str(index): item
                for index, item in enumerate(row_values)
            }
        )
    return rows


def _candidate_payload(
    *,
    candidates: Any,
    representative_indices: Sequence[int],
    roles: Mapping[int, str],
    variable_names: Sequence[str] | None,
    target_names: Sequence[str] | None,
    prediction_mean: Any | None,
    prediction_variance: Any | None,
    train_X_summary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = _as_2d_tensor(_select_rows(candidates, representative_indices))
    names = list(variable_names or [])
    mean_rows = _row_dicts(prediction_mean, target_names, len(representative_indices))
    variance_rows = _row_dicts(prediction_variance, target_names, len(representative_indices))
    training_by_index = {int(item["index"]): item for item in train_X_summary if "index" in item}
    payload: list[dict[str, Any]] = []
    for local_index, original_index in enumerate(representative_indices):
        values = {
            names[column] if column < len(names) else str(column): float(selected[local_index, column].item())
            for column in range(selected.shape[-1])
        }
        range_position: dict[str, str] = {}
        for column, item in training_by_index.items():
            if column >= selected.shape[-1] or "min" not in item or "max" not in item:
                continue
            name = names[column] if column < len(names) else str(column)
            value = float(selected[local_index, column].item())
            if value < float(item["min"]):
                range_position[name] = "below_training_range"
            elif value > float(item["max"]):
                range_position[name] = "above_training_range"
            else:
                range_position[name] = "within_training_range"
        payload.append(
            {
                "candidate_index": int(original_index),
                "representative_role": roles.get(int(original_index), "representative"),
                "variables": values,
                "relative_to_training_range": range_position,
                "predicted_mean": mean_rows[local_index],
                "predicted_variance": variance_rows[local_index],
            }
        )
    return payload


def build_candidate_explanation_prompt(
    *,
    goal: Any,
    candidates: Any,
    representative_indices: Sequence[int],
    representative_roles: Mapping[int, str],
    llm_context: Any | None = None,
    train_X: Any | None = None,
    train_Y: Any | None = None,
    prediction_mean: Any | None = None,
    prediction_variance: Any | None = None,
    model_config: Any | None = None,
    acquisition_config: Any | None = None,
    optimize_config: Any | None = None,
    perspectives: Sequence[str] = _DEFAULT_PERSPECTIVES,
    language: str = "ja",
    additional_prompt: str | None = None,
) -> str:
    """Build a grounded JSON-only prompt for domain interpretation."""

    goal_config = coerce_goal_config(goal)
    context = coerce_llm_context(llm_context)
    variable_names = list(context.variable_names or [])
    target_names = list(context.target_names or [])
    train_X_summary = _matrix_summary(train_X, variable_names)
    train_Y_summary = _matrix_summary(train_Y, target_names)
    candidate_rows = _candidate_payload(
        candidates=candidates,
        representative_indices=representative_indices,
        roles=representative_roles,
        variable_names=variable_names,
        target_names=target_names,
        prediction_mean=prediction_mean,
        prediction_variance=prediction_variance,
        train_X_summary=train_X_summary,
    )
    payload = {
        "role": (
            "You explain Bayesian-optimization candidates to materials, process, "
            "manufacturing, and development engineers."
        ),
        "language": language,
        "goal": None if goal_config is None else goal_config.text,
        "requested_perspectives": list(perspectives),
        "additional_prompt": additional_prompt,
        "important_rules": [
            "Return JSON only.",
            "Separate model evidence from domain interpretation.",
            "Do not present correlations, model predictions, or plausible mechanisms as proven causality.",
            "Use variable_descriptions, target_descriptions, domain_notes, and candidate_policy as the primary domain context.",
            "When domain information is insufficient, state the uncertainty instead of inventing a mechanism.",
            "Identify practical manufacturability, controllability, safety, scale-up, measurement, and validation concerns when relevant.",
            "Explain only the supplied representative candidates; do not fabricate unlisted candidates.",
            "Candidate indices in the response must match the supplied candidate_index values.",
            "Recommended checks should be actionable experiments, measurements, or process confirmations.",
        ],
        "domain_context": {
            "variable_names": variable_names,
            "target_names": target_names,
            "variable_descriptions": dict(context.variable_descriptions),
            "target_descriptions": dict(context.target_descriptions),
            "domain_notes": list(context.domain_notes),
            "candidate_policy": context.candidate_policy,
            "metadata": _to_jsonable(context.metadata),
        },
        "model_context": {
            "model_config": _to_jsonable(model_config),
            "acquisition_config": _to_jsonable(acquisition_config),
            "optimize_config": _to_jsonable(optimize_config),
            "train_X_summary": train_X_summary,
            "train_Y_summary": train_Y_summary,
        },
        "candidate_context": {
            "total_candidates": int(_as_2d_tensor(candidates).shape[0]),
            "representative_count": len(representative_indices),
            "representative_candidates": candidate_rows,
        },
        "output_schema": {
            "summary": "overall interpretation of the proposed batch",
            "selection_note": "why only these representative points are described",
            "common_patterns": ["pattern shared by multiple proposed candidates"],
            "candidate_explanations": [
                {
                    "candidate_index": 0,
                    "representative_role": "highest_acquisition or central_candidate or diverse_candidate",
                    "headline": "short description",
                    "model_evidence": ["facts directly supported by supplied predictions/configuration"],
                    "physical_interpretation": ["plausible physical interpretation with uncertainty wording"],
                    "chemical_interpretation": ["plausible chemical interpretation with uncertainty wording"],
                    "manufacturing_interpretation": ["operability, controllability, scale-up, safety, and quality points"],
                    "development_interpretation": ["learning value, hypothesis discrimination, and next-step value"],
                    "risks_and_tradeoffs": ["risk or trade-off"],
                    "recommended_checks": ["specific measurement or experiment"],
                    "confidence": "low or medium or high",
                }
            ],
            "assumptions": ["assumption used in the interpretation"],
            "warnings": ["missing context or extrapolation warning"],
        },
    }
    return "Explain the representative Bayesian-optimization candidates from the following JSON payload.\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def explanation_from_payload(
    payload: Mapping[str, Any],
    *,
    total_candidates: int,
    representative_indices: Sequence[int],
) -> CandidateExplanation:
    """Convert an LLM JSON response into typed explanation objects."""

    data = dict(payload)
    allowed_indices = {int(index) for index in representative_indices}
    explanations: list[CandidatePointExplanation] = []
    warnings = _string_list(data.get("warnings"))
    for item in data.get("candidate_explanations") or []:
        if not isinstance(item, Mapping):
            warnings.append("Ignored a non-object candidate_explanation entry.")
            continue
        explanation = CandidatePointExplanation.from_mapping(item)
        if explanation.candidate_index not in allowed_indices:
            warnings.append(
                f"Ignored explanation for unselected candidate_index={explanation.candidate_index}."
            )
            continue
        explanations.append(explanation)
    explained_indices = {item.candidate_index for item in explanations}
    missing = [index for index in representative_indices if int(index) not in explained_indices]
    if missing:
        warnings.append(f"No LLM explanation was returned for representative indices {missing}.")
    return CandidateExplanation(
        total_candidates=int(total_candidates),
        representative_indices=[int(index) for index in representative_indices],
        omitted_count=max(int(total_candidates) - len(representative_indices), 0),
        summary=str(data.get("summary") or ""),
        selection_note=str(data.get("selection_note") or ""),
        common_patterns=_string_list(data.get("common_patterns")),
        candidate_explanations=explanations,
        assumptions=_string_list(data.get("assumptions")),
        warnings=warnings,
        raw_response=data,
    )


def _coerce_explanation_config(
    value: CandidateExplanationConfig | Mapping[str, Any] | None,
) -> CandidateExplanationConfig:
    if value is None:
        return CandidateExplanationConfig()
    if isinstance(value, CandidateExplanationConfig):
        return value
    return CandidateExplanationConfig(**dict(value))


def install_bayesian_optimizer_candidate_explanation_api(optimizer_cls: type[Any]) -> None:
    """Install ``explain_candidates`` and ``explain_last_candidates`` once."""

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
        """Explain final candidates using model evidence and domain context.

        Candidate generation is not repeated. When many points are present, the
        method selects a highest-ranked/central/diverse representative subset.
        """

        settings = getattr(self, "llm_settings", None)
        resolved_config = _coerce_explanation_config(config)
        if max_representatives is not None:
            resolved_config.max_representatives = int(max_representatives)
        elif settings is not None and getattr(settings, "candidate_explanation_max_points", None) is not None:
            resolved_config.max_representatives = int(settings.candidate_explanation_max_points)
        if perspectives is not None:
            resolved_config.perspectives = tuple(str(item) for item in perspectives)
        elif settings is not None and getattr(settings, "candidate_explanation_perspectives", None):
            resolved_config.perspectives = tuple(settings.candidate_explanation_perspectives)
        if prompt is not None:
            resolved_config.prompt = prompt

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
            bounds = getattr(context, "bounds", None) or bounds
        representative_indices, roles = select_representative_candidates(
            candidates,
            acq_value=acq_value,
            max_representatives=resolved_config.max_representatives,
            bounds=bounds,
        )
        selected_candidates = _select_rows(candidates, representative_indices)

        prediction_mean = None
        prediction_variance = None
        local_warnings: list[str] = []
        if resolved_config.include_predictions:
            try:
                prediction = self.predict(selected_candidates, return_result=True)
                prediction_mean = getattr(prediction, "mean", None)
                if resolved_config.include_uncertainty:
                    prediction_variance = getattr(prediction, "variance", None)
            except Exception as exc:  # explanation should remain available for custom models
                local_warnings.append(
                    f"Candidate prediction summary was unavailable: {type(exc).__name__}: {exc}"
                )

        resolved_goal = goal
        resolved_llm_config = llm_config
        resolved_llm_context = llm_context
        resolved_response = explanation_response
        if settings is not None:
            if resolved_goal is None:
                resolved_goal = settings.goal
            if resolved_llm_config is None:
                resolved_llm_config = settings.llm_config
            if resolved_llm_context is None:
                resolved_llm_context = settings.llm_context
            if resolved_response is None:
                resolved_response = getattr(settings, "candidate_explanation_response", None)
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
        if resolved_response is None:
            client = make_llm_client(resolved_llm_config)
            payload = parse_json_payload(client.generate_json(explanation_prompt).text)
        else:
            payload = parse_json_payload(resolved_response)
        if not isinstance(payload, Mapping):
            raise ValueError("Candidate explanation response must be a JSON object.")
        explanation = explanation_from_payload(
            payload,
            total_candidates=int(_as_2d_tensor(candidates).shape[0]),
            representative_indices=representative_indices,
        )
        explanation.warnings = local_warnings + explanation.warnings
        self.last_candidate_explanation = explanation
        self.last_candidate_explanation_prompt = explanation_prompt
        if candidate_result is not None:
            candidate_result.explanation = explanation
        return explanation

    def explain_last_candidates(self: Any, **kwargs: Any) -> CandidateExplanation:
        """Explain the most recent ``candidate()`` result."""

        return self.explain_candidates(None, **kwargs)

    optimizer_cls.explain_candidates = explain_candidates
    optimizer_cls.explain_last_candidates = explain_last_candidates
    optimizer_cls._bochan_candidate_explanation_api_installed = True


__all__ = [
    "CandidateExplanation",
    "CandidateExplanationConfig",
    "CandidatePointExplanation",
    "build_candidate_explanation_prompt",
    "explanation_from_payload",
    "install_bayesian_optimizer_candidate_explanation_api",
    "select_representative_candidates",
]
