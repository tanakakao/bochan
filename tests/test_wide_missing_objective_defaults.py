from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import bochan.api.automatic_best_f as automatic_best_f
import bochan.api.automatic_multiobjective as automatic_multiobjective
from bochan.models.wide_multitask import wide_to_long


def test_wide_to_long_omits_only_nan_target_cells() -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor(
        [[0.0, 1.0], [0.5, float("nan")], [1.0, 0.0]],
        dtype=torch.double,
    )

    X_long, Y_long, num_tasks = wide_to_long(train_X, train_Y)

    assert num_tasks == 2
    torch.testing.assert_close(
        X_long,
        torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=torch.double,
        ),
    )
    torch.testing.assert_close(
        Y_long,
        torch.tensor(
            [[0.0], [1.0], [0.5], [1.0], [0.0]],
            dtype=torch.double,
        ),
    )
    assert not torch.isnan(Y_long).any()


def test_default_ref_point_ignores_missing_cells_per_output() -> None:
    values = torch.tensor(
        [
            [1.0, float("nan")],
            [float("nan"), 3.0],
            [2.0, 2.0],
        ],
        dtype=torch.double,
    )

    ref_point = automatic_multiobjective.make_default_ref_point(
        values,
        margin=0.1,
    )

    torch.testing.assert_close(
        ref_point,
        torch.tensor([0.9, 1.9], dtype=torch.double),
    )


def test_partitioning_uses_only_complete_objective_rows(monkeypatch) -> None:
    values = torch.tensor(
        [
            [1.0, float("nan")],
            [float("nan"), 3.0],
            [2.0, 2.0],
        ],
        dtype=torch.double,
    )
    ref_point = torch.tensor([0.9, 1.9], dtype=torch.double)
    captured: dict[str, torch.Tensor] = {}
    sentinel = object()

    def fake_partitioning(*, ref_point, Y):
        captured["ref_point"] = ref_point
        captured["Y"] = Y
        return sentinel

    from botorch.utils.multi_objective.box_decompositions import non_dominated

    monkeypatch.setattr(
        non_dominated,
        "FastNondominatedPartitioning",
        fake_partitioning,
    )

    result = automatic_multiobjective.make_partitioning(ref_point, values)

    assert result is sentinel
    torch.testing.assert_close(captured["ref_point"], ref_point)
    torch.testing.assert_close(
        captured["Y"],
        torch.tensor([[2.0, 2.0]], dtype=torch.double),
    )


def test_partitioning_rejects_data_without_a_jointly_observed_row() -> None:
    values = torch.tensor(
        [[1.0, float("nan")], [float("nan"), 3.0]],
        dtype=torch.double,
    )

    with pytest.raises(ValueError, match="at least one training row"):
        automatic_multiobjective.make_partitioning(
            torch.tensor([0.9, 2.9], dtype=torch.double),
            values,
        )


def test_reference_point_rejects_an_entirely_missing_output() -> None:
    values = torch.tensor(
        [[1.0, float("nan")], [2.0, float("nan")]],
        dtype=torch.double,
    )

    with pytest.raises(ValueError, match="Missing output indices"):
        automatic_multiobjective.make_default_ref_point(values)


def test_regression_best_f_ignores_nan_values(monkeypatch) -> None:
    monkeypatch.setattr(
        automatic_best_f,
        "_regression_observed_values",
        lambda *args, **kwargs: torch.tensor(
            [[1.0, float("nan")], [2.0, 3.0]],
            dtype=torch.double,
        ),
    )

    result = automatic_best_f._compute_regression_best_f(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
    )

    torch.testing.assert_close(
        result,
        torch.tensor(3.0, dtype=torch.double),
    )
