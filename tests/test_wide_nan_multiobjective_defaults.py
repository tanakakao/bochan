from __future__ import annotations

import pytest
import torch

from bochan.api import AcquisitionConfig, engine_defaults
from bochan.api.automatic_multiobjective import (
    make_default_ref_point,
    make_partitioning,
)
from bochan.models.wide_multitask import wide_to_long


def test_wide_to_long_omits_only_nan_target_cells() -> None:
    train_X = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double)
    train_Y = torch.tensor(
        [[1.0, float("nan")], [float("nan"), 2.0], [3.0, 4.0]],
        dtype=torch.double,
    )

    X_long, Y_long, num_tasks = wide_to_long(train_X, train_Y)

    assert num_tasks == 2
    torch.testing.assert_close(
        X_long,
        torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
            dtype=torch.double,
        ),
    )
    torch.testing.assert_close(
        Y_long,
        torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.double),
    )
    assert torch.isfinite(Y_long).all()


def test_default_ref_point_uses_finite_values_per_output() -> None:
    values = torch.tensor(
        [[1.0, float("nan")], [float("nan"), 2.0]],
        dtype=torch.double,
    )

    ref_point = make_default_ref_point(values, margin=0.1)

    torch.testing.assert_close(
        ref_point,
        torch.tensor([0.9, 1.9], dtype=torch.double),
    )


def test_ehvi_partitioning_drops_incomplete_rows() -> None:
    values = torch.tensor(
        [
            [1.0, float("nan")],
            [float("nan"), 2.0],
            [0.5, 1.5],
        ],
        dtype=torch.double,
    )
    ref_point = make_default_ref_point(values, margin=0.1)

    partitioning = make_partitioning(ref_point, values)

    assert torch.isfinite(partitioning.pareto_Y).all()
    torch.testing.assert_close(
        partitioning.pareto_Y,
        torch.tensor([[0.5, 1.5]], dtype=torch.double),
    )


def test_ehvi_partitioning_requires_one_complete_objective_row() -> None:
    values = torch.tensor(
        [[1.0, float("nan")], [float("nan"), 2.0]],
        dtype=torch.double,
    )
    ref_point = make_default_ref_point(values, margin=0.1)

    with pytest.raises(ValueError, match="EHVI requires at least one training row"):
        make_partitioning(ref_point, values)


def test_reference_point_requires_each_output_to_have_an_observation() -> None:
    values = torch.tensor(
        [[1.0, float("nan")], [2.0, float("nan")]],
        dtype=torch.double,
    )

    with pytest.raises(ValueError, match="Missing output indices: \\[1\\]"):
        make_default_ref_point(values)


def test_public_api_installs_nan_safe_engine_default_references() -> None:
    # Importing the public AcquisitionConfig installs support for the
    # callables that engine_defaults imported by name.
    assert AcquisitionConfig is not None
    assert engine_defaults.make_default_ref_point is make_default_ref_point
    assert engine_defaults.make_partitioning is make_partitioning
