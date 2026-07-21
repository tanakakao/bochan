from __future__ import annotations

import json

from bochan.acquisition.feasible import (
    FeasibilityConstraintSpec,
    OrdinalRankConstraintSpec,
)
from bochan.api import (
    BayesianOptimizer,
    CandidateRepairConfig,
    ModelConfig,
    MultiOutputConfig,
    OutcomeConstraintConfig,
)
from bochan.llm import LLMContextConfig, build_config_planner_prompt, plan_configs


def _planner_payload(prompt: str) -> dict:
    return json.loads(prompt.split("\n", maxsplit=1)[1])


def test_prompt_exposes_dedicated_hybrid_and_constraint_schemas():
    prompt = build_config_planner_prompt(
        goal="propertyを最大化し、合格確率とordinal rankを制約にする",
        mode="full",
        llm_context=LLMContextConfig(
            variable_names=["raw material 1", "raw material 2", "temperature"],
            target_names=["property", "pass_probability", "quality_rank"],
        ),
    )
    payload = _planner_payload(prompt)

    schemas = payload["dedicated_config_schemas"]
    assert set(schemas) == {
        "hybrid_model",
        "outcome_constraints",
        "candidate_constraints_and_repair",
    }
    hybrid = schemas["hybrid_model"]["model_config"]
    assert hybrid["task_type"] == "hybrid"
    assert hybrid["multi_output_config"]["use_hybrid"] is True
    assert len(hybrid["multi_output_config"]["output_configs"]) == 3

    outcome = schemas["outcome_constraints"]
    assert outcome["classification_probability_constraint"]["target_class"] == 1
    assert outcome["ordinal_rank_probability_constraint"]["rank"] == 2

    candidate = schemas["candidate_constraints_and_repair"]["optimize_config"]
    assert candidate["repair_config"]["final_sum_constraint"]["rhs"] == 1.0
    assert "constraint_semantics" in payload
    assert "multi_output_config" in payload["output_schema"]["model_config"]
    assert (
        "outcome_constraint_config"
        in payload["output_schema"]["acquisition_config"]
    )
    assert "repair_config" in payload["output_schema"]["optimize_config"]


def test_plan_configs_normalizes_named_candidate_constraints_to_indices():
    context = LLMContextConfig(
        variable_names=[
            "raw material 1",
            "raw material 2",
            "raw material 3",
            "atmosphere",
        ],
        target_names=["property"],
    )
    plan = plan_configs(
        goal="3成分の合計を1にして候補を作る",
        llm_context=context,
        planner_response={
            "optimize_config": {
                "optimizer": "optimize_acqf",
                "q": 3,
                "equality_constraints": [
                    {
                        "indices": [
                            "raw material 1",
                            "raw material 2",
                            "raw material 3",
                        ],
                        "coefficients": [1.0, 1.0, 1.0],
                        "rhs": 1.0,
                    }
                ],
                "fixed_features": {"atmosphere": 1.0},
                "repair_config": {
                    "numeric_indices": [
                        "raw material 1",
                        "raw material 2",
                        "raw material 3",
                    ],
                    "steps": {
                        "raw material 1": 0.05,
                        "raw material 2": 0.05,
                        "raw material 3": 0.05,
                    },
                    "comp_idx": [
                        "raw material 1",
                        "raw material 2",
                        "raw material 3",
                    ],
                    "fixed_features": {"atmosphere": 1.0},
                    "final_sum_constraint": {
                        "indices": [
                            "raw material 1",
                            "raw material 2",
                            "raw material 3",
                        ],
                        "rhs": 1.0,
                    },
                },
            }
        },
    )

    optimize = plan["optimize_config"]
    assert optimize["equality_constraints"][0] == (
        [0, 1, 2],
        [1.0, 1.0, 1.0],
        1.0,
    )
    assert optimize["fixed_features"] == {3: 1.0}
    repair = optimize["repair_config"]
    assert repair["numeric_indices"] == [0, 1, 2]
    assert repair["steps"] == {0: 0.05, 1: 0.05, 2: 0.05}
    assert repair["comp_idx"] == [0, 1, 2]
    assert repair["fixed_features"] == {3: 1.0}
    assert repair["final_sum_constraint"] == ([0, 1, 2], 1.0)


def test_suggest_all_coerces_hybrid_outcome_and_repair_configs():
    context = LLMContextConfig(
        variable_names=[
            "raw material 1",
            "raw material 2",
            "raw material 3",
            "atmosphere",
        ],
        target_names=["property", "pass_probability", "quality_rank"],
    )
    optimizer = BayesianOptimizer(model_config=ModelConfig())

    suggestion = optimizer.suggest_all(
        train_X=[[0.2, 0.3, 0.5, 0.0], [0.4, 0.1, 0.5, 1.0]],
        train_Y=[[1.2, 1.0, 2.0], [1.5, 0.0, 3.0]],
        llm_context=context,
        planner_response={
            "model_config": {
                "task_type": "regression",
                "model_type": "base",
                "multi_output_config": {
                    "output_configs": [
                        {
                            "task_type": "regression",
                            "model_type": "base",
                        },
                        {
                            "task_type": "binary",
                            "model_type": "base",
                        },
                        {
                            "task_type": "ordinal",
                            "model_type": "base",
                            "model_kwargs": {"num_classes": 4},
                        },
                    ]
                },
            },
            "fit_config": {"method": "auto", "skip_fit": True},
            "acquisition_config": {
                "name": "NEHVI",
                "objective_config": {
                    "mode": "multi_output",
                    "outputs": ["property"],
                    "directions": ["maximize"],
                    "weights": [1.0],
                },
                "outcome_constraint_config": {
                    "constraints": [
                        {
                            "kind": "feasibility",
                            "output": "pass_probability",
                            "threshold": 0.8,
                            "sense": "ge",
                            "target_class": 1,
                        },
                        {
                            "kind": "ordinal_rank",
                            "output": "quality_rank",
                            "rank": 2,
                            "sense": "ge",
                            "probability_threshold": 0.8,
                        },
                    ]
                },
            },
            "optimize_config": {
                "optimizer": "optimize_acqf",
                "q": 3,
                "repair_config": {
                    "comp_idx": [
                        "raw material 1",
                        "raw material 2",
                        "raw material 3",
                    ],
                    "fixed_features": {"atmosphere": 1.0},
                    "final_sum_constraint": {
                        "indices": [
                            "raw material 1",
                            "raw material 2",
                            "raw material 3",
                        ],
                        "rhs": 1.0,
                    },
                },
            },
        },
    )

    assert suggestion.model_config.task_type == "hybrid"
    multi_output = suggestion.model_config.multi_output_config
    assert isinstance(multi_output, MultiOutputConfig)
    assert multi_output.use_hybrid is True
    assert multi_output.output_names == [
        "property",
        "pass_probability",
        "quality_rank",
    ]
    assert [item["name"] for item in multi_output.output_configs] == [
        "property",
        "pass_probability",
        "quality_rank",
    ]

    outcome = suggestion.acq_config.outcome_constraint_config
    assert isinstance(outcome, OutcomeConstraintConfig)
    assert isinstance(outcome.constraints[0], FeasibilityConstraintSpec)
    assert isinstance(outcome.constraints[1], OrdinalRankConstraintSpec)
    assert outcome.constraints[0].target_class == 1
    assert outcome.constraints[1].probability_threshold == 0.8

    repair = suggestion.opt_config.repair_config
    assert isinstance(repair, CandidateRepairConfig)
    assert repair.comp_idx == [0, 1, 2]
    assert repair.fixed_features == {3: 1.0}
    assert repair.final_sum_constraint == ([0, 1, 2], 1.0)
