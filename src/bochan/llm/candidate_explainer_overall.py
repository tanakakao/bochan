"""Integrated domain explanations for final BayesianOptimizer candidates.

This module extends the base candidate explainer with a typed
``overall_interpretation`` for each representative point. The integrated text
combines model evidence, physical and chemical hypotheses, manufacturing
practicality, development value, risks, and the recommended next action without
removing the individual perspective fields.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from . import candidate_explainer as _base

CandidateExplanationConfig = _base.CandidateExplanationConfig
_as_2d_tensor = _base._as_2d_tensor
_select_rows = _base._select_rows
_bounds_tensor = _base._bounds_tensor
select_representative_candidates = _base.select_representative_candidates


@dataclass
class CandidatePointExplanation:
    """Explanation for one representative candidate.

    ``overall_interpretation`` is a decision-oriented synthesis. The remaining
    fields retain the evidence and each specialist perspective separately so the
    synthesis can be audited.
    """

    candidate_index: int
    representative_role: str = "representative"
    headline: str = ""
    overall_interpretation: str = ""
    model_evidence: list[str] = field(default_factory=list)
    physical_interpretation: list[str] = field(default_factory=list)
    chemical_interpretation: list[str] = field(default_factory=list)
    manufacturing_interpretation: list[str] = field(default_factory=list)
    development_interpretation: list[str] = field(default_factory=list)
    risks_and_tradeoffs: list[str] = field(default_factory=list)
    recommended_checks: list[str] = field(default_factory=list)
    confidence: str = "unknown"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CandidatePointExplanation:
        data = dict(value)
        return cls(
            candidate_index=int(data.get("candidate_index", -1)),
            representative_role=str(data.get("representative_role") or "representative"),
            headline=str(data.get("headline") or ""),
            overall_interpretation=str(data.get("overall_interpretation") or ""),
            model_evidence=_base._string_list(data.get("model_evidence")),
            physical_interpretation=_base._string_list(data.get("physical_interpretation")),
            chemical_interpretation=_base._string_list(data.get("chemical_interpretation")),
            manufacturing_interpretation=_base._string_list(
                data.get("manufacturing_interpretation")
            ),
            development_interpretation=_base._string_list(
                data.get("development_interpretation")
            ),
            risks_and_tradeoffs=_base._string_list(data.get("risks_and_tradeoffs")),
            recommended_checks=_base._string_list(data.get("recommended_checks")),
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
        """Return a JSON-friendly representation including integrated explanations."""

        return {
            "total_candidates": self.total_candidates,
            "representative_indices": list(self.representative_indices),
            "omitted_count": self.omitted_count,
            "summary": self.summary,
            "selection_note": self.selection_note,
            "common_patterns": list(self.common_patterns),
            "candidate_explanations": [
                asdict(item) for item in self.candidate_explanations
            ],
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "raw_response": _base._to_jsonable(self.raw_response),
        }


def build_candidate_explanation_prompt(*args: Any, **kwargs: Any) -> str:
    """Build the base prompt and require a synthesis for every candidate."""

    prompt = _base.build_candidate_explanation_prompt(*args, **kwargs)
    prefix, encoded = prompt.split("\n", maxsplit=1)
    payload = json.loads(encoded)
    payload["important_rules"].append(
        "For every candidate, provide overall_interpretation as a concise, "
        "decision-oriented synthesis of model evidence, all requested domain "
        "perspectives, risks, trade-offs, and the recommended next action."
    )
    candidate_schema = payload["output_schema"]["candidate_explanations"][0]
    candidate_schema["overall_interpretation"] = (
        "integrated explanation combining evidence, physical and chemical "
        "hypotheses, manufacturing practicality, development value, major risks, "
        "and what should be done next"
    )
    payload["output_schema"]["summary"] = (
        "overall interpretation of the complete proposed batch, including its "
        "combined technical intent, practical value, and principal risks"
    )
    return prefix + "\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def explanation_from_payload(
    payload: Mapping[str, Any],
    *,
    total_candidates: int,
    representative_indices: Sequence[int],
) -> CandidateExplanation:
    """Convert an LLM JSON response into typed integrated explanation objects."""

    data = dict(payload)
    allowed_indices = {int(index) for index in representative_indices}
    explanations: list[CandidatePointExplanation] = []
    warnings = _base._string_list(data.get("warnings"))
    for item in data.get("candidate_explanations") or []:
        if not isinstance(item, Mapping):
            warnings.append("Ignored a non-object candidate_explanation entry.")
            continue
        explanation = CandidatePointExplanation.from_mapping(item)
        if explanation.candidate_index not in allowed_indices:
            warnings.append(
                "Ignored explanation for unselected "
                f"candidate_index={explanation.candidate_index}."
            )
            continue
        explanations.append(explanation)

    explained_indices = {item.candidate_index for item in explanations}
    missing = [
        int(index)
        for index in representative_indices
        if int(index) not in explained_indices
    ]
    if missing:
        warnings.append(
            f"No LLM explanation was returned for representative indices {missing}."
        )

    return CandidateExplanation(
        total_candidates=int(total_candidates),
        representative_indices=[int(index) for index in representative_indices],
        omitted_count=max(int(total_candidates) - len(representative_indices), 0),
        summary=str(data.get("summary") or ""),
        selection_note=str(data.get("selection_note") or ""),
        common_patterns=_base._string_list(data.get("common_patterns")),
        candidate_explanations=explanations,
        assumptions=_base._string_list(data.get("assumptions")),
        warnings=warnings,
        raw_response=data,
    )


__all__ = [
    "CandidateExplanation",
    "CandidateExplanationConfig",
    "CandidatePointExplanation",
    "build_candidate_explanation_prompt",
    "explanation_from_payload",
    "select_representative_candidates",
]
