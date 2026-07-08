from __future__ import annotations

from bochan.tabular import TabularBayesianOptimizer
from bochan.tabular import optimizer_api
from bochan.tabular.builders import (
    make_acquisition_config,
    make_model_config,
    make_optimize_config,
)


def test_make_model_config_accepts_dict_with_nested_input_transform() -> None:
    config = make_model_config(
        {
            "task_type": "regression",
            "model_type": "base",
            "input_transform_config": {
                "normalize": False,
                "perturbation": True,
                "n_w": 6,
                "std": 0.05,
            },
        }
    )

    assert config.task_type == "regression"
    assert config.input_transform_config is not None
    assert config.input_transform_config.normalize is False
    assert config.input_transform_config.perturbation is True
    assert config.input_transform_config.n_w == 6
    assert config.input_transform_config.std == 0.05


def test_make_acquisition_config_accepts_dict_with_nested_objective_config() -> None:
    config = make_acquisition_config(
        {
            "name": "EI",
            "objective_config": {
                "mode": "scalar",
                "output": 0,
                "n_w": 4,
            },
        },
        objective_risk_type="cvar",
        objective_alpha=0.2,
    )

    assert config.name == "EI"
    assert config.objective_config is not None
    assert config.objective_config.mode == "scalar"
    assert config.objective_config.output == 0
    assert config.objective_config.n_w == 4
    assert config.objective_config.risk_type == "cvar"
    assert config.objective_config.alpha == 0.2


def test_make_optimize_config_accepts_dict_with_nested_repair_config() -> None:
    config = make_optimize_config(
        {
            "q": 3,
            "optimizer": "pso",
            "repair_config": {
                "comp_idx": [0, 1, 2],
                "k": 2,
            },
        },
        steps=0.1,
        final_priority="constraints",
    )

    assert config.q == 3
    assert config.optimizer == "evo"
    assert config.evo_method == "pso"
    assert config.repair_config is not None
    assert config.repair_config.comp_idx == [0, 1, 2]
    assert config.repair_config.k == 2
    assert config.repair_config.steps == 0.1
    assert config.repair_config.final_priority == "constraints"


def test_make_optimize_config_preserves_dict_repair_config_when_direct_none_is_synthetic() -> None:
    config = make_optimize_config(
        {
            "q": 2,
            "repair_config": {
                "numeric_indices": [0, 1, 2],
                "steps": [0.01, 0.01, 0.01],
                "comp_idx": [0, 1, 2],
                "k": 2,
            },
        },
        repair_config=None,
    )

    assert config.q == 2
    assert config.repair_config is not None
    assert config.repair_config.numeric_indices == [0, 1, 2]
    assert config.repair_config.steps == [0.01, 0.01, 0.01]
    assert config.repair_config.comp_idx == [0, 1, 2]
    assert config.repair_config.k == 2


def test_make_optimize_config_routes_shared_direct_fields_to_optimizer_config() -> None:
    equality_constraints = [([0, 1, 2], [1.0, 1.0, 1.0], 1.0)]
    inequality_constraints = [([3], [1.0], 100.0)]

    config = make_optimize_config(
        q=5,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        fixed_features={4: 10.0},
        numeric_indices=[0, 1, 2],
        steps=[0.01, 0.01, 0.01],
        comp_idx=[0, 1, 2],
        k=2,
    )

    assert config.q == 5
    assert config.equality_constraints is equality_constraints
    assert config.inequality_constraints is inequality_constraints
    assert config.fixed_features == {4: 10.0}
    assert config.repair_config is not None
    assert config.repair_config.equality_constraints is None
    assert config.repair_config.inequality_constraints is None
    assert config.repair_config.fixed_features is None
    assert config.repair_config.numeric_indices == [0, 1, 2]
    assert config.repair_config.steps == [0.01, 0.01, 0.01]
    assert config.repair_config.comp_idx == [0, 1, 2]
    assert config.repair_config.k == 2


def test_make_optimize_config_accepts_repair_prefixed_shared_field_overrides() -> None:
    optimizer_equality_constraints = [([0, 1], [1.0, 1.0], 1.0)]
    repair_equality_constraints = [([0, 1, 2], [1.0, 1.0, 1.0], 1.0)]

    config = make_optimize_config(
        equality_constraints=optimizer_equality_constraints,
        repair_equality_constraints=repair_equality_constraints,
        repair_fixed_features={2: 0.0},
        steps=[0.01, 0.01, 0.01],
    )

    assert config.equality_constraints is optimizer_equality_constraints
    assert config.repair_config is not None
    assert config.repair_config.equality_constraints is repair_equality_constraints
    assert config.repair_config.fixed_features == {2: 0.0}
    assert config.repair_config.steps == [0.01, 0.01, 0.01]


def test_tabular_constructor_accepts_dict_configs() -> None:
    bo = TabularBayesianOptimizer(
        model_config={
            "task_type": "regression",
            "model_type": "base",
            "input_transform_config": {
                "perturbation": True,
                "n_w": 4,
            },
        },
        fit_config={"maxiter": 32},
        input_cols=["x1"],
        target_cols="y",
    )

    assert bo.model_config.task_type == "regression"
    assert bo.model_config.input_transform_config is not None
    assert bo.model_config.input_transform_config.perturbation is True
    assert bo.model_config.input_transform_config.n_w == 4
    assert bo.fit_config is not None
    assert bo.fit_config.maxiter == 32


def test_tabular_candidate_accepts_dict_configs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_candidate(self, acq_config=None, opt_config=None, **kwargs):
        captured["acq_config"] = acq_config
        captured["opt_config"] = opt_config
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(optimizer_api._BaseTabularBayesianOptimizer, "candidate", fake_candidate)

    bo = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x1"],
        target_cols="y",
    )

    result = bo.candidate(
        acq_config={"name": "EI", "objective_config": {"n_w": 4}},
        opt_config={"q": 2, "optimizer": "evo"},
        objective_risk_type="cvar",
        evo_method="ga",
    )

    assert result == "ok"
    acq_config = captured["acq_config"]
    opt_config = captured["opt_config"]

    assert acq_config.name == "EI"
    assert acq_config.objective_config is not None
    assert acq_config.objective_config.n_w == 4
    assert acq_config.objective_config.risk_type == "cvar"
    assert opt_config.q == 2
    assert opt_config.optimizer == "evo"
    assert opt_config.evo_method == "ga"
