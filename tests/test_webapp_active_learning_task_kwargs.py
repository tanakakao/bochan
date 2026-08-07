from __future__ import annotations

import pytest
import torch

from bochan.serving.webapp.workflows_tabular import _set_active_learning_kwargs


@pytest.mark.parametrize(
    ("task_type", "multi_output", "aggregation_key"),
    [
        ("regression", False, None),
        ("binary", False, None),
        ("multiclass", False, None),
        ("ordinal", False, None),
        ("regression", True, "output_reduction"),
        ("binary", True, "output_mode"),
        ("multiclass", True, "output_mode"),
        ("ordinal", True, "output_mode"),
    ],
)
def test_web_active_learning_kwargs_are_task_and_output_aware(
    task_type: str,
    multi_output: bool,
    aggregation_key: str | None,
) -> None:
    kwargs: dict[str, object] = {}
    train_x = torch.tensor([[0.1], [0.9]], dtype=torch.double)

    _set_active_learning_kwargs(
        kwargs,
        acq_key="variance",
        train_x=train_x,
        task_type=task_type,
        multi_output=multi_output,
        output_weights=[0.25, 0.75],
    )

    assert kwargs["X_observed"] is train_x
    if not multi_output:
        assert "output_weights" not in kwargs
        assert "output_reduction" not in kwargs
        assert "output_mode" not in kwargs
    else:
        assert kwargs["output_weights"] == [0.25, 0.75]
        assert aggregation_key is not None
        assert kwargs[aggregation_key] == "weighted_mean"
        other_key = "output_mode" if aggregation_key == "output_reduction" else "output_reduction"
        assert other_key not in kwargs


@pytest.mark.parametrize(
    ("task_type", "expect_observed"),
    [
        ("regression", False),
        ("binary", True),
        ("multiclass", True),
        ("ordinal", True),
    ],
)
def test_web_nipv_kwargs_are_task_aware(
    task_type: str,
    expect_observed: bool,
) -> None:
    kwargs: dict[str, object] = {}
    train_x = torch.tensor([[0.2], [0.8]], dtype=torch.double)

    _set_active_learning_kwargs(
        kwargs,
        acq_key="nipv",
        train_x=train_x,
        task_type=task_type,
        multi_output=False,
    )

    assert kwargs["mc_points"] is train_x
    assert ("X_observed" in kwargs) is expect_observed
    if expect_observed:
        assert kwargs["X_observed"] is train_x
