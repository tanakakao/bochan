from __future__ import annotations

import pytest
import torch

from bochan.models.regression.gaussian.materials import (
    MaterialExplicitTaskSpec,
    normalize_material_task_feature,
    split_material_task_feature,
    stack_material_task_observations,
    validate_explicit_material_task_data,
)


def test_explicit_task_spec_is_serializable() -> None:
    spec = MaterialExplicitTaskSpec(
        task_feature=-1,
        all_tasks=(0, 2, 5),
        output_tasks=(2, 5),
    )
    assert spec.as_dict() == {
        "task_feature": -1,
        "all_tasks": [0, 2, 5],
        "output_tasks": [2, 5],
    }


def test_explicit_task_spec_rejects_unknown_output_task() -> None:
    with pytest.raises(ValueError, match="subset"):
        MaterialExplicitTaskSpec(all_tasks=(0, 1), output_tasks=(2,))


def test_normalize_material_task_feature_supports_negative_index() -> None:
    assert normalize_material_task_feature(-1, 4) == 3
    assert normalize_material_task_feature(-2, 4) == 2
    with pytest.raises(ValueError, match="out of bounds"):
        normalize_material_task_feature(4, 4)


def test_stack_material_task_observations_uses_long_format() -> None:
    base_X = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    train_Y = torch.tensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])

    long_X, long_Y, long_Yvar, task_feature = stack_material_task_observations(
        base_X,
        train_Y,
        task_values=(10, 20, 30),
    )

    assert task_feature == 2
    assert long_Yvar is None
    assert long_X.shape == (6, 3)
    assert long_Y.shape == (6, 1)
    assert long_X[:, -1].tolist() == [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    assert long_Y.squeeze(-1).tolist() == pytest.approx([0.1, 0.2, 0.3, 1.1, 1.2, 1.3])


def test_stack_material_task_observations_can_insert_task_column() -> None:
    base_X = torch.tensor([[1.0, 10.0]])
    train_Y = torch.tensor([[0.1, 0.2]])

    long_X, _, _, task_feature = stack_material_task_observations(
        base_X,
        train_Y,
        task_values=(3, 7),
        task_feature=1,
    )

    assert task_feature == 1
    assert long_X.tolist() == [[1.0, 3.0, 10.0], [1.0, 7.0, 10.0]]


def test_stack_material_task_observations_splits_known_noise() -> None:
    base_X = torch.tensor([[1.0], [2.0]])
    train_Y = torch.tensor([[0.1, 0.2], [1.1, 1.2]])
    train_Yvar = torch.tensor([[0.01, 0.02], [0.11, 0.12]])

    _, _, long_Yvar, _ = stack_material_task_observations(
        base_X,
        train_Y,
        train_Yvar,
    )

    assert long_Yvar is not None
    assert long_Yvar.squeeze(-1).tolist() == pytest.approx([0.01, 0.02, 0.11, 0.12])


def test_stack_material_task_observations_drops_missing_targets() -> None:
    base_X = torch.tensor([[1.0], [2.0]])
    train_Y = torch.tensor([[0.1, float("nan")], [1.1, 1.2]])

    long_X, long_Y, _, task_feature = stack_material_task_observations(base_X, train_Y)

    assert task_feature == 1
    assert long_X.shape[0] == 3
    assert long_X[:, -1].tolist() == [0.0, 0.0, 1.0]
    assert long_Y.squeeze(-1).tolist() == pytest.approx([0.1, 1.1, 1.2])


def test_validate_explicit_material_task_data_rejects_non_integer_tasks() -> None:
    train_X = torch.tensor([[1.0, 0.5], [2.0, 1.0]])
    train_Y = torch.tensor([[0.1], [0.2]])

    with pytest.raises(ValueError, match="integer-valued"):
        validate_explicit_material_task_data(train_X, train_Y, task_feature=-1)


def test_validate_explicit_material_task_data_returns_observed_tasks() -> None:
    train_X = torch.tensor([[1.0, 2.0], [2.0, 0.0], [3.0, 2.0]])
    train_Y = torch.tensor([[0.1], [0.2], [0.3]])

    task_feature, observed = validate_explicit_material_task_data(
        train_X,
        train_Y,
        task_feature=-1,
        all_tasks=(0, 1, 2),
    )

    assert task_feature == 1
    assert observed == (0, 2)


def test_split_material_task_feature_restores_base_inputs() -> None:
    train_X = torch.tensor([[1.0, 3.0, 10.0], [2.0, 7.0, 20.0]])

    base_X, task_ids, task_feature = split_material_task_feature(train_X, task_feature=1)

    assert task_feature == 1
    assert base_X.tolist() == [[1.0, 10.0], [2.0, 20.0]]
    assert task_ids.tolist() == [3, 7]
