from __future__ import annotations

import torch

from bochan.tabular.builders import make_optimize_config
from bochan.tabular.converter import resolve_optimize_config_columns


def test_optimize_config_constraints_accept_column_names() -> None:
    config = make_optimize_config(
        q=3,
        equality_constraints=[(["raw_1", "raw_2", "raw_3"], [1.0, 1.0, 1.0], 1.0)],
        inequality_constraints=[(["temperature"], [1.0], 100.0)],
        fixed_features={"time": 10.0},
        numeric_indices=["raw_1", "raw_2", "raw_3"],
        steps={"raw_1": 0.01, "raw_2": 0.01, "raw_3": 0.01},
        comp_idx=["raw_1", "raw_2", "raw_3"],
        k=2,
        final_sum_constraint=(["raw_1", "raw_2", "raw_3"], 1.0),
    )

    resolved = resolve_optimize_config_columns(
        config,
        ["raw_1", "raw_2", "raw_3", "temperature", "time"],
        dtype=torch.double,
        device=None,
    )

    assert resolved.equality_constraints == [([0, 1, 2], [1.0, 1.0, 1.0], 1.0)]
    assert resolved.inequality_constraints == [([3], [1.0], 100.0)]
    assert resolved.fixed_features == {4: 10.0}

    assert resolved.repair_config is not None
    assert resolved.repair_config.numeric_indices == [0, 1, 2]
    assert resolved.repair_config.steps == [0.01, 0.01, 0.01]
    assert resolved.repair_config.comp_idx == [0, 1, 2]
    assert resolved.repair_config.k == 2
    assert resolved.repair_config.final_sum_constraint == ([0, 1, 2], 1.0)


def test_repair_prefixed_constraints_accept_column_names() -> None:
    config = make_optimize_config(
        equality_constraints=([(["raw_1", "raw_2"], [1.0, 1.0], 1.0)]),
        repair_equality_constraints=[(["raw_1", "raw_2", "raw_3"], [1.0, 1.0, 1.0], 1.0)],
        repair_inequality_constraints=[(["temperature"], [1.0], 120.0)],
        repair_fixed_features={"time": 5.0},
        steps={"raw_1": 0.01, "raw_2": 0.01, "raw_3": 0.01},
        numeric_indices=["raw_1", "raw_2", "raw_3"],
    )

    resolved = resolve_optimize_config_columns(
        config,
        ["raw_1", "raw_2", "raw_3", "temperature", "time"],
        dtype=torch.double,
        device=None,
    )

    assert resolved.equality_constraints == [([0, 1], [1.0, 1.0], 1.0)]
    assert resolved.repair_config is not None
    assert resolved.repair_config.equality_constraints == [([0, 1, 2], [1.0, 1.0, 1.0], 1.0)]
    assert resolved.repair_config.inequality_constraints == [([3], [1.0], 120.0)]
    assert resolved.repair_config.fixed_features == {4: 5.0}
    assert resolved.repair_config.numeric_indices == [0, 1, 2]
    assert resolved.repair_config.steps == [0.01, 0.01, 0.01]
