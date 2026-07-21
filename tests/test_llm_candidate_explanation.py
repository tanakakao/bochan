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
    LLMContextConfig,
    LLMSettings,
    build_candidate_explanation_prompt,
    select_representative_candidates,
)


def _payload(prompt: str) -> dict:
    return json.loads(prompt.split("\n", maxsplit=1)[1])


def _candidate_result(candidates: torch.Tensor, acq_value: torch.Tensor) -> CandidateResult:
    return CandidateResult(
        candidates=candidates,
        acq_value=acq_value,
        acqf=object(),
        acq_config=AcquisitionConfig(name="UCB"),
        opt_config=OptimizeConfig(q=int(candidates.shape[0])),
        data_context=DataContext(bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]])),
    )


def test_representative_selection_keeps_best_central_and_diverse_points():
    candidates = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.5, 0.5],
            [0.9, 0.1],
            [1.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.double,
    )
    acq_value = torch.tensor([0.1, 0.2, 0.3, 0.4, 1.2, 0.5], dtype=torch.double)

    indices, roles = select_representative_candidates(
        candidates,
        acq_value=acq_value,
        max_representatives=4,
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
    )

    assert len(indices) == 4
    assert indices[0] == 4
    assert roles[4] == "highest_acquisition"
    assert "central_candidate" in roles.values()
    assert "diverse_candidate" in roles.values()


def test_small_candidate_batches_are_all_explained():
    candidates = torch.tensor([[0.2, 0.3], [0.7, 0.8]], dtype=torch.double)

    indices, roles = select_representative_candidates(
        candidates,
        max_representatives=5,
    )

    assert indices == [0, 1]
    assert roles == {0: "all_candidates", 1: "all_candidates"}


def test_candidate_explanation_prompt_separates_evidence_and_domain_hypotheses():
    candidates = torch.tensor(
        [[800.0, 2.0], [900.0, 4.0], [850.0, 3.0]],
        dtype=torch.double,
    )
    prompt = build_candidate_explanation_prompt(
        goal="導電率を高くし、収縮率を低くする",
        candidates=candidates,
        representative_indices=[1, 2],
        representative_roles={1: "highest_acquisition", 2: "central_candidate"},
        llm_context=LLMContextConfig(
            variable_names=["temperature", "time"],
            target_names=["conductivity", "shrinkage"],
            variable_descriptions={
                "temperature": "焼成温度。",
                "time": "保持時間。",
            },
            target_descriptions={
                "conductivity": "高いほど望ましい。",
                "shrinkage": "低いほど望ましい。",
            },
            domain_notes=["高温長時間では粒成長に注意する。"],
        ),
        train_X=torch.tensor([[780.0, 1.0], [880.0, 3.5]], dtype=torch.double),
        train_Y=torch.tensor([[10.0, 5.0], [12.0, 6.0]], dtype=torch.double),
        prediction_mean=torch.tensor([[13.0, 5.5], [12.5, 5.0]], dtype=torch.double),
        prediction_variance=torch.tensor([[0.2, 0.3], [0.1, 0.2]], dtype=torch.double),
        model_config=ModelConfig(task_type="multi_objective"),
        acquisition_config=AcquisitionConfig(name="NEHVI"),
        optimize_config=OptimizeConfig(q=3),
    )
    data = _payload(prompt)

    assert data["requested_perspectives"] == [
        "physics",
        "chemistry",
        "manufacturing",
        "development",
    ]
    assert len(data["candidate_context"]["representative_candidates"]) == 2
    assert data["candidate_context"]["representative_candidates"][0]["candidate_index"] == 1
    assert any("Separate model evidence" in rule for rule in data["important_rules"])
    assert any("proven causality" in rule for rule in data["important_rules"])
    assert "recommended_checks" in data["output_schema"]["candidate_explanations"][0]


def test_bayesian_optimizer_explains_and_attaches_representative_candidates():
    candidates = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.4],
            [0.6, 0.6],
            [0.8, 0.2],
            [0.9, 0.9],
            [0.1, 0.8],
        ],
        dtype=torch.double,
    )
    acq_value = torch.tensor([0.1, 0.2, 0.3, 1.5, 0.5, 0.6, 0.4], dtype=torch.double)
    result = _candidate_result(candidates, acq_value)
    indices, roles = select_representative_candidates(
        candidates,
        acq_value=acq_value,
        max_representatives=3,
        bounds=result.data_context.bounds,
    )
    response = {
        "summary": "高獲得値条件と条件空間を広く確認する候補を含む。",
        "selection_note": "代表3点を説明した。",
        "common_patterns": ["既存範囲内と境界条件を組み合わせている。"],
        "candidate_explanations": [
            {
                "candidate_index": index,
                "representative_role": roles[index],
                "headline": f"候補{index}",
                "model_evidence": ["予測平均と不確かさを確認した。"],
                "physical_interpretation": ["物理機構としては仮説段階である。"],
                "chemical_interpretation": ["反応進行の可能性を確認する。"],
                "manufacturing_interpretation": ["制御余裕を確認する。"],
                "development_interpretation": ["仮説識別に有用である。"],
                "risks_and_tradeoffs": ["外挿リスクを確認する。"],
                "recommended_checks": ["再現実験を行う。"],
                "confidence": "medium",
            }
            for index in indices
        ],
        "assumptions": ["変数説明が正しい。"],
        "warnings": [],
    }
    optimizer = BayesianOptimizer(
        model_config=ModelConfig(),
        bounds=result.data_context.bounds,
        llm_settings=LLMSettings(
            goal="propertyを最大化する",
            llm_context=LLMContextConfig(
                variable_names=["material", "temperature"],
                target_names=["property"],
                variable_descriptions={
                    "material": "原料配合。",
                    "temperature": "処理温度。",
                },
            ),
        ),
    )
    optimizer.train_X = torch.tensor([[0.1, 0.2], [0.7, 0.6]], dtype=torch.double)
    optimizer.train_Y = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    optimizer.history.append(result)

    def fake_predict(X, *, return_result=False, **kwargs):
        mean = X.sum(dim=-1, keepdim=True)
        variance = torch.full_like(mean, 0.05)
        prediction = PredictionResult(
            posterior=None,
            mean=mean,
            variance=variance,
            task_type="regression",
        )
        return prediction if return_result else mean

    optimizer.predict = fake_predict
    explanation = optimizer.explain_candidates(
        result,
        max_representatives=3,
        explanation_response=response,
    )

    assert isinstance(explanation, CandidateExplanation)
    assert explanation.total_candidates == 7
    assert explanation.representative_indices == indices
    assert explanation.omitted_count == 4
    assert len(explanation.candidate_explanations) == 3
    assert result.explanation is explanation
    assert optimizer.last_candidate_explanation is explanation
    prompt_data = _payload(optimizer.last_candidate_explanation_prompt)
    assert prompt_data["domain_context"]["variable_names"] == [
        "material",
        "temperature",
    ]
    assert prompt_data["candidate_context"]["representative_count"] == 3


def test_unselected_candidate_explanations_are_ignored_with_warning():
    candidates = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    result = CandidateResult(
        candidates=candidates,
        acq_value=torch.tensor([0.1, 1.0, 0.2]),
        acqf=object(),
        acq_config=AcquisitionConfig(name="UCB"),
        opt_config=OptimizeConfig(q=3),
        data_context=DataContext(bounds=torch.tensor([[0.0], [1.0]])),
    )
    optimizer = BayesianOptimizer(
        model_config=ModelConfig(),
        bounds=result.data_context.bounds,
        llm_settings=LLMSettings(goal="高い値を探索する"),
    )
    optimizer.train_X = candidates
    optimizer.train_Y = candidates
    optimizer.predict = lambda X, **kwargs: PredictionResult(
        posterior=None,
        mean=X,
        variance=torch.ones_like(X),
    )

    explanation = optimizer.explain_candidates(
        result,
        max_representatives=1,
        explanation_response={
            "candidate_explanations": [
                {"candidate_index": 99, "headline": "invalid"},
            ]
        },
    )

    assert explanation.candidate_explanations == []
    assert any("unselected candidate_index=99" in warning for warning in explanation.warnings)
    assert any("No LLM explanation" in warning for warning in explanation.warnings)
