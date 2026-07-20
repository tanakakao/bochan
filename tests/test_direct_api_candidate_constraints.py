from __future__ import annotations

import math

import pandas as pd
import torch

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    CandidateRepairConfig,
    FitConfig,
    ModelConfig,
    MultiOutputConfig,
    OptimizeConfig,
    OutputConfig,
)
from bochan.tabular import TabularBayesianOptimizer


def _hybrid_frame() -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    x1_values = [0.0, 0.1, 0.2, 0.4, 0.6, 1.0]
    x2_values = [0.0, 0.1, 0.2, 0.4]
    x3_values = [0.0, 0.3, 0.6, 0.9]
    temperature_values = [50.0, 65.0, 80.0, 100.0]
    time_values = [10.0, 40.0, 80.0, 120.0]
    for index in range(24):
        x1 = x1_values[index % len(x1_values)]
        x2 = x2_values[(index * 2) % len(x2_values)]
        x3 = x3_values[(index * 3) % len(x3_values)]
        temperature = temperature_values[index % len(temperature_values)]
        time = time_values[(index * 3) % len(time_values)]
        property_value = 0.25 * x1 + 0.35 * x2 + 0.4 * x3 + 0.001 * temperature
        records.append(
            {
                "raw material 1": x1,
                "raw material 2": x2,
                "raw material 3": x3,
                "temperature": temperature,
                "time": time,
                "property": property_value,
                "y_cat_str": int(property_value > 0.35),
            }
        )
    return pd.DataFrame.from_records(records)


def _assert_candidate_constraints(candidate) -> None:
    components = [float(candidate[index]) for index in range(3)]
    component_sum = sum(components)
    first_two_sum = sum(components[:2])
    process_sum = float(candidate[3]) + float(candidate[4])
    zero_count = sum(abs(value) <= 1e-8 for value in components)

    assert math.isclose(component_sum, 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert first_two_sum <= 0.4 + 1e-6
    assert process_sum >= 100.0 - 1e-6
    assert zero_count >= 1


def test_tabular_bayesian_optimizer_applies_named_constraints_and_repair() -> None:
    frame = _hybrid_frame()
    optimizer = TabularBayesianOptimizer(
        model_config={"task_type": "hybrid"},
        fit_config={"skip_fit": True},
        input_cols=[
            "raw material 1",
            "raw material 2",
            "raw material 3",
            "temperature",
            "time",
        ],
        target_cols=["property", "y_cat_str"],
        multi_output_config={
            "output_configs": [
                {
                    "task_type": "regression",
                    "model_type": "base",
                    "name": "property",
                },
                {
                    "task_type": "binary",
                    "model_type": "base",
                    "name": "y_cat_str",
                },
            ],
            "use_hybrid": True,
        },
    )
    optimizer.fit(frame)

    candidates, _ = optimizer.candidate(
        acq_config={"name": "ehvi"},
        opt_config={
            "q": 2,
            "optimizer": "optimize_acqf",
            "num_restarts": 4,
            "raw_samples": 64,
            "optimizer_kwargs": {
                "options": {
                    "maxiter": 12,
                    "batch_limit": 4,
                }
            },
            "constraints": [
                (
                    ["raw material 1", "raw material 2", "raw material 3"],
                    [1.0, 1.0, 1.0],
                    "=",
                    1.0,
                ),
                (
                    ["temperature", "time"],
                    [1.0, 1.0],
                    ">=",
                    100.0,
                ),
                (
                    ["raw material 1", "raw material 2"],
                    [1.0, 1.0],
                    "<=",
                    0.4,
                ),
            ],
            "repair_config": {
                "steps": {
                    "raw material 1": 0.01,
                    "raw material 2": 0.01,
                    "raw material 3": 0.01,
                    "temperature": 1.0,
                    "time": 2.0,
                },
                "comp_idx": [
                    "raw material 1",
                    "raw material 2",
                    "raw material 3",
                ],
                "k": 2,
                "final_priority": "constraints",
                "max_iters": 24,
                "num_alternations": 3,
            },
        },
    )

    assert len(candidates) == 2
    for row in candidates.itertuples(index=False, name=None):
        _assert_candidate_constraints(row)


def test_bayesian_optimizer_repair_uses_top_level_botorch_inequality_sense() -> None:
    frame = _hybrid_frame()
    input_columns = [
        "raw material 1",
        "raw material 2",
        "raw material 3",
        "temperature",
        "time",
    ]
    train_x = torch.as_tensor(
        frame[input_columns].to_numpy(dtype=float),
        dtype=torch.double,
    )
    train_y = torch.as_tensor(
        frame[["property", "y_cat_str"]].to_numpy(dtype=float),
        dtype=torch.double,
    )
    bounds = torch.tensor(
        [
            [0.0, 0.0, 0.0, 50.0, 10.0],
            [1.0, 0.4, 0.9, 100.0, 120.0],
        ],
        dtype=torch.double,
    )

    optimizer = BayesianOptimizer(
        model_config=ModelConfig(
            task_type="hybrid",
            multi_output_config=MultiOutputConfig(
                output_configs=[
                    OutputConfig(
                        task_type="regression",
                        model_type="base",
                        name="property",
                    ),
                    OutputConfig(
                        task_type="binary",
                        model_type="base",
                        name="y_cat_str",
                    ),
                ],
                use_hybrid=True,
            ),
        ),
        fit_config=FitConfig(skip_fit=True),
        bounds=bounds,
    )
    optimizer.fit(train_x, train_y)

    equality_constraints = [
        (
            torch.tensor([0, 1, 2], dtype=torch.long),
            torch.tensor([1.0, 1.0, 1.0], dtype=torch.double),
            1.0,
        )
    ]
    inequality_constraints = [
        (
            torch.tensor([3, 4], dtype=torch.long),
            torch.tensor([1.0, 1.0], dtype=torch.double),
            100.0,
        ),
        (
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([-1.0, -1.0], dtype=torch.double),
            -0.4,
        ),
    ]
    candidates, _ = optimizer.candidate(
        AcquisitionConfig(name="ehvi"),
        OptimizeConfig(
            q=2,
            optimizer="optimize_acqf",
            num_restarts=4,
            raw_samples=64,
            optimizer_kwargs={
                "options": {
                    "maxiter": 12,
                    "batch_limit": 4,
                }
            },
            equality_constraints=equality_constraints,
            inequality_constraints=inequality_constraints,
            repair_config=CandidateRepairConfig(
                numeric_indices=[0, 1, 2, 3, 4],
                steps=[0.01, 0.01, 0.01, 1.0, 2.0],
                comp_idx=[0, 1, 2],
                k=2,
                final_priority="constraints",
                max_iters=24,
                num_alternations=3,
            ),
        ),
    )

    assert candidates.shape == torch.Size([2, 5])
    for candidate in candidates.detach().cpu():
        _assert_candidate_constraints(candidate)
