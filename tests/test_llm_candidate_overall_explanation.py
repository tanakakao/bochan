from __future__ import annotations

import json

import torch

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    CandidateResult,
    DataContext,
    ModelConfig,
    OptimizeConfig,
    PredictionResult,
)
from bochan.llm import (
    CandidateExplanation,
    CandidatePointExplanation,
    LLMContextConfig,
    LLMSettings,
    build_candidate_explanation_prompt,
)


def _prompt_payload(prompt: str) -> dict:
    return json.loads(prompt.split("\n", maxsplit=1)[1])


def test_prompt_requires_integrated_explanation_for_each_candidate():
    prompt = build_candidate_explanation_prompt(
        goal="導電率を高くし、収縮率を低くする",
        candidates=torch.tensor([[800.0, 2.0]], dtype=torch.double),
        representative_indices=[0],
        representative_roles={0: "all_candidates"},
        llm_context=LLMContextConfig(
            variable_names=["temperature", "time"],
            target_names=["conductivity", "shrinkage"],
        ),
    )
    payload = _prompt_payload(prompt)
    schema = payload["output_schema"]["candidate_explanations"][0]

    assert "overall_interpretation" in schema
    assert any(
        "overall_interpretation" in rule
        and "decision-oriented synthesis" in rule
        for rule in payload["important_rules"]
    )
    assert "complete proposed batch" in payload["output_schema"]["summary"]


def test_bayesian_optimizer_returns_typed_overall_interpretation():
    candidates = torch.tensor(
        [[0.2, 0.3], [0.8, 0.7], [0.5, 0.4]],
        dtype=torch.double,
    )
    result = CandidateResult(
        candidates=candidates,
        acq_value=torch.tensor([0.2, 1.0, 0.4], dtype=torch.double),
        acqf=object(),
        acq_config=AcquisitionConfig(name="UCB"),
        opt_config=OptimizeConfig(q=3),
        data_context=DataContext(
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        ),
    )
    optimizer = BayesianOptimizer(
        model_config=ModelConfig(),
        bounds=result.data_context.bounds,
        llm_settings=LLMSettings(
            goal="propertyを最大化する",
            llm_context=LLMContextConfig(
                variable_names=["composition", "temperature"],
                target_names=["property"],
            ),
        ),
    )
    optimizer.train_X = candidates
    optimizer.train_Y = candidates[:, :1]
    optimizer.predict = lambda X, **kwargs: PredictionResult(
        posterior=None,
        mean=X.sum(dim=-1, keepdim=True),
        variance=torch.full((X.shape[0], 1), 0.05, dtype=X.dtype),
    )

    explanation = optimizer.explain_candidates(
        result,
        max_representatives=1,
        explanation_response={
            "summary": "候補群全体では性能向上と工程成立性を同時に確認する構成である。",
            "candidate_explanations": [
                {
                    "candidate_index": 1,
                    "representative_role": "highest_acquisition",
                    "headline": "性能有望だが工程確認が必要な候補",
                    "overall_interpretation": (
                        "モデル上は最も有望で、物理化学的にも妥当な可能性がある一方、"
                        "量産時の温度制御余裕を確認してから採用判断すべき候補である。"
                    ),
                    "model_evidence": ["候補中で獲得関数値が最大である。"],
                    "physical_interpretation": ["構造変化が寄与する可能性がある。"],
                    "chemical_interpretation": ["反応進行度が変化する可能性がある。"],
                    "manufacturing_interpretation": ["温度ばらつきの影響確認が必要である。"],
                    "development_interpretation": ["仮説識別価値が高い。"],
                    "risks_and_tradeoffs": ["制御幅が狭い可能性がある。"],
                    "recommended_checks": ["温度を振った再現実験を行う。"],
                    "confidence": "medium",
                }
            ],
            "warnings": [],
        },
    )

    assert isinstance(explanation, CandidateExplanation)
    point = explanation.candidate_explanations[0]
    assert isinstance(point, CandidatePointExplanation)
    assert "採用判断" in point.overall_interpretation
    assert explanation.summary.startswith("候補群全体")
    serialized = explanation.to_dict()
    assert serialized["candidate_explanations"][0]["overall_interpretation"]
    assert result.explanation is explanation


def test_missing_overall_interpretation_is_backward_compatible():
    point = CandidatePointExplanation.from_mapping(
        {
            "candidate_index": 0,
            "headline": "旧形式の応答",
            "model_evidence": ["予測値を確認した。"],
        }
    )

    assert point.overall_interpretation == ""
